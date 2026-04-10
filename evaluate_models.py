#!/usr/bin/env python3
"""
Simple Evaluation Script for SAM3 LoRA Model
Compares Base vs LoRA model and outputs masks
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import pycocotools.mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM3 LoRA model')
    parser.add_argument('--data-dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--weights', type=str, required=True, help='LoRA weights path')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--prompt', type=str, default='clothing', help='Text prompt')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    base_output_dir = output_dir / "base_model"
    lora_output_dir = output_dir / "lora_model"
    gt_output_dir = output_dir / "ground_truth"

    base_output_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    ann_file = data_dir / "_annotations.coco.json"

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    print(f"\nFound {len(images)} images in validation set")
    print(f"Categories: {categories}")

    try:
        from sam3.model_builder import build_sam3_image_model
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
        import yaml

        print("\n" + "="*60)
        print("Loading Base SAM3 Model (without LoRA)...")
        print("="*60)

        base_model = build_sam3_image_model(
            device=device.type,
            compile=False,
            load_from_HF=False,
            checkpoint_path="/mnt/vris-comfy/models/sam3/sam3.pt",
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )
        base_model.to(device)
        base_model.eval()
        print("✓ Base model loaded successfully")

        print("\n" + "="*60)
        print("Loading LoRA Fine-tuned Model...")
        print("="*60)

        lora_model = build_sam3_image_model(
            device=device.type,
            compile=False,
            load_from_HF=False,
            checkpoint_path="/mnt/vris-comfy/models/sam3/sam3.pt",
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        lora_model = apply_lora_to_model(lora_model, lora_config)
        load_lora_weights(lora_model, args.weights)
        lora_model.to(device)
        lora_model.eval()
        print("✓ LoRA model loaded successfully")

        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
        )
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        )
        import torch.nn.functional as F

        def run_inference(model, image_path, prompt_text):
            pil_image = PILImage.open(image_path).convert("RGB")
            w, h = pil_image.size

            sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
            query = FindQueryLoaded(
                query_text=prompt_text,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=0,
                    original_image_id=0,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            datapoint = Datapoint(find_queries=[query], images=[sam_image])

            transform = ComposeAPI(transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            datapoint = transform(datapoint)

            batch = collate_fn_api([datapoint], dict_key="input")["input"]
            batch = copy_data_to_device(batch, device, non_blocking=True)

            with torch.no_grad():
                outputs = model(batch)
                last_output = outputs[-1]

            pred_logits = last_output['pred_logits']
            pred_masks = last_output.get('pred_masks', None)

            scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
            keep = scores > args.threshold

            if pred_masks is not None and keep.sum() > 0:
                masks_small = pred_masks[0, keep].sigmoid() > 0.5
                orig_h, orig_w = pil_image.size[1], pil_image.size[0]
                masks_resized = F.interpolate(
                    masks_small.unsqueeze(0).float(),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) > 0.5
                return masks_resized.cpu().numpy(), keep.sum().item()
            else:
                return None, 0

        print("\n" + "="*60)
        print(f"Running inference on {len(images)} images...")
        print("="*60)

        results_summary = []

        for idx, (img_id, img_info) in enumerate(sorted(images.items()), 1):
            image_name = img_info['file_name']
            image_path = data_dir / image_name

            print(f"\n[{idx}/{len(images)}] Processing: {image_name}")
            print("-" * 50)

            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

            gt_mask_combined = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in annotations:
                seg = ann.get('segmentation', None)
                if seg:
                    if isinstance(seg, dict):
                        mask_np = mask_utils.decode(seg)
                        gt_mask_combined[mask_np > 0] = 255
                    elif isinstance(seg, list):
                        rles = mask_utils.frPyObjects(seg, img_info['height'], img_info['width'])
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                        gt_mask_combined[mask_np > 0] = 255

            gt_mask_img = PILImage.fromarray(gt_mask_combined)
            gt_mask_img.save(gt_output_dir / f"{Path(image_name).stem}_gt_mask.png")
            print(f"  ✓ GT mask saved: {gt_output_dir / f'{Path(image_name).stem}_gt_mask.png'}")

            print(f"  Running Base model inference...")
            base_masks, base_count = run_inference(base_model, image_path, args.prompt)

            if base_masks is not None and len(base_masks) > 0:
                base_mask_combined = np.zeros((base_masks.shape[1], base_masks.shape[2]), dtype=np.uint8)
                for i in range(base_masks.shape[0]):
                    base_mask_combined[base_masks[i]] = 255
                base_mask_img = PILImage.fromarray(base_mask_combined)
                base_mask_img.save(base_output_dir / f"{Path(image_name).stem}_base_mask.png")
                print(f"  ✓ Base model: {base_count} detections → {base_output_dir / f'{Path(image_name).stem}_base_mask.png'}")
            else:
                empty_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                PILImage.fromarray(empty_mask).save(base_output_dir / f"{Path(image_name).stem}_base_mask.png")
                print(f"  ⚠ Base model: No detections")

            print(f"  Running LoRA model inference...")
            lora_masks, lora_count = run_inference(lora_model, image_path, args.prompt)

            if lora_masks is not None and len(lora_masks) > 0:
                lora_mask_combined = np.zeros((lora_masks.shape[1], lora_masks.shape[2]), dtype=np.uint8)
                for i in range(lora_masks.shape[0]):
                    lora_mask_combined[lora_masks[i]] = 255
                lora_mask_img = PILImage.fromarray(lora_mask_combined)
                lora_mask_img.save(lora_output_dir / f"{Path(image_name).stem}_lora_mask.png")
                print(f"  ✓ LoRA model: {lora_count} detections → {lora_output_dir / f'{Path(image_name).stem}_lora_mask.png'}")
            else:
                empty_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                PILImage.fromarray(empty_mask).save(lora_output_dir / f"{Path(image_name).stem}_lora_mask.png")
                print(f"  ⚠ LoRA model: No detections")

            results_summary.append({
                'image': image_name,
                'gt_annotations': len(annotations),
                'base_detections': base_count,
                'lora_detections': lora_count
            })

        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Output directories:")
        print(f"  Ground Truth masks: {gt_output_dir}/")
        print(f"  Base model masks:  {base_output_dir}/")
        print(f"  LoRA model masks:  {lora_output_dir}/")
        print(f"\n📊 Results Summary:")
        print(f"{'Image':<20} {'GT':>5} {'Base':>8} {'LoRA':>8}")
        print("-" * 45)
        for r in results_summary:
            print(f"{r['image']:<20} {r['gt_annotations']:>5} {r['base_detections']:>8} {r['lora_detections']:>8}")
        print("-" * 45)

        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'prompt': args.prompt,
                'threshold': args.threshold,
                'device': str(device),
                'results': results_summary
            }, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_file}")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

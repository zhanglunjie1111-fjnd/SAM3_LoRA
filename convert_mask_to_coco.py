#!/usr/bin/env python3
"""
Mask to COCO Format Converter for SAM3 LoRA Training

This script converts image + mask pairs to COCO format with RLE encoding.
Specifically designed for clothing segmentation tasks.

Author: Auto-generated for SAM3_LoRA project
Date: 2026
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not installed. Will use fallback RLE encoding.")


def mask_to_rle(mask_np: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary numpy mask to COCO RLE format (Compressed RLE).

    Uses pycocotools.mask.frPyObjects() to generate COCO-standard compressed RLE.
    This format is compatible with pycocotools.mask.decode() used in training.

    COCO RLE format (compressed):
    - "counts": Bytes string (compressed run-length encoding)
    - "size": [height, width]

    Args:
        mask_np: Binary mask (H, W) with values 0 or 255 (or 0/1)

    Returns:
        RLE encoded dictionary in COCO format:
        {
            "counts": bytes string (compressed),
            "size": [height, width]
        }
    """
    # Ensure binary mask (0 or 1)
    if mask_np.max() > 1:
        mask_binary = (mask_np > 127).astype(np.uint8)
    else:
        mask_binary = mask_np.astype(np.uint8)

    if HAS_PYCOCOTOOLS:
        # Use pycocotools to generate compressed RLE (COCO standard)
        # Ensure array is Fortran contiguous (required by pycocotools)
        mask_fortran = np.asfortranarray(mask_binary)
        rle = mask_utils.encode(mask_fortran)
        # Convert to JSON-serializable format
        rle_output = {
            'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts'],
            'size': [int(rle['size'][0]), int(rle['size'][1])]
        }
    else:
        # Fallback: uncompressed RLE (may not work with all COCO tools)
        print("Warning: Using uncompressed RLE (pycocotools not available)")
        mask_flat = mask_binary.flatten(order='F')
        counts = []
        if len(mask_flat) > 0:
            diff = np.diff(mask_flat)
            change_indices = np.where(diff != 0)[0] + 1
            prev_idx = 0
            for idx in change_indices:
                counts.append(int(idx - prev_idx))
                prev_idx = idx
            counts.append(int(len(mask_flat) - prev_idx))
            if mask_flat[0] == 1:
                counts.insert(0, 0)
        rle_output = {
            "counts": counts,
            "size": [int(mask_np.shape[0]), int(mask_np.shape[1])]
        }

    return rle_output


def find_image_mask_pairs(data_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find matching image-mask pairs in the directory.

    Naming convention:
    - Images: image_XXX.ext (e.g., image_001.jpg, image_002.png)
    - Masks: mask_XXX.ext (e.g., mask_001.png, mask_002.png)

    Returns:
        List of (image_path, mask_path) tuples
    """
    data_path = Path(data_dir)

    # Find all images and masks
    images = {}
    masks = {}

    # Support common image formats
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    mask_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    for file in data_path.iterdir():
        if file.is_file():
            stem = file.stem.lower()
            ext = file.suffix.lower()

            # Check if follows naming pattern: XXX_001, image_001, etc.
            parts = stem.split('_')
            if len(parts) >= 2:
                number_part = parts[-1]

                if ext in img_extensions and ('image' in stem):
                    images[number_part] = file
                elif ext in mask_extensions and 'mask' in stem:
                    masks[number_part] = file

    # Match pairs by number
    pairs = []
    for num in sorted(masks.keys()):
        if num in images:
            pairs.append((images[num], masks[num]))
        else:
            print(f"Warning: Found mask_{num} but no matching image")

    return pairs


def compute_bbox_from_mask(mask_np: np.ndarray) -> List[int]:
    """
    Compute bounding box from binary mask in COCO format [x, y, width, height].

    Args:
        mask_np: Binary mask (H, W)

    Returns:
        Bounding box [x, y, width, height]
    """
    # Ensure binary
    if mask_np.max() > 1:
        mask_binary = (mask_np > 127)
    else:
        mask_binary = mask_np.astype(bool)

    # Find rows and columns that contain the object
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # COCO format: [x, y, width, height]
    bbox = [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1)
    ]

    return bbox


def compute_area_from_mask(mask_np: np.ndarray) -> int:
    """
    Compute area (number of pixels) from binary mask.

    Args:
        mask_np: Binary mask (H, W)

    Returns:
        Area in pixels
    """
    if mask_np.max() > 1:
        area = int((mask_np > 127).sum())
    else:
        area = int(mask_np.sum())

    return area


def process_single_pair(
    image_path: Path,
    mask_path: Path,
    image_id: int,
    annotation_id_start: int,
    category_id: int,
    category_name: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single image-mask pair.

    Returns:
        Tuple of (image_info_dict, annotation_list)
    """
    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    # Load mask
    pil_mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask_np = np.array(pil_mask)

    # Create image info dict
    image_info = {
        "id": image_id,
        "file_name": image_path.name,
        "height": height,
        "width": width
    }

    # Convert mask to RLE
    rle_segmentation = mask_to_rle(mask_np)

    # Compute bounding box
    bbox = compute_bbox_from_mask(mask_np)

    # Compute area
    area = compute_area_from_mask(mask_np)

    # Create annotation dict (COCO format)
    annotation = {
        "id": annotation_id_start,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "segmentation": rle_segmentation,  # RLE format!
        "iscrowd": 0
    }

    return image_info, [annotation]


def create_coco_dataset(
    input_dir: str,
    output_dir: str,
    category_name: str = "clothing",
    category_id: int = 1,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Create COCO dataset from image-mask pairs.

    Splits data into train/valid sets and generates _annotations.coco.json files.

    Args:
        input_dir: Directory containing image_XXX and mask_XXX files
        output_dir: Output directory (will create train/ and valid/ subdirs)
        category_name: Category name (e.g., "clothing")
        category_id: Category ID (e.g., 1)
        train_ratio: Ratio of training data (0.8 = 80% train, 20% valid)
        seed: Random seed for reproducible splitting
    """
    print("=" * 70)
    print("SAM3 LoRA Data Preparation Tool")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Category: {category_name} (ID: {category_id})")
    print(f"Train/Valid split: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")

    # Find all image-mask pairs
    print("\n[Step 1] Scanning for image-mask pairs...")
    pairs = find_image_mask_pairs(input_dir)
    total_pairs = len(pairs)

    if total_pairs == 0:
        raise ValueError(f"No image-mask pairs found in {input_dir}")

    print(f"Found {total_pairs} image-mask pairs:")
    for i, (img, mask) in enumerate(pairs[:5]):
        print(f"  {i+1}. {img.name} <-> {mask.name}")
    if total_pairs > 5:
        print(f"  ... and {total_pairs - 5} more")

    # Split into train/valid
    np.random.seed(seed)
    indices = np.arange(total_pairs)
    np.random.shuffle(indices)

    split_idx = int(total_pairs * train_ratio)
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    print(f"\n[Step 2] Splitting dataset:")
    print(f"  Training set: {len(train_indices)} images")
    print(f"  Validation set: {len(valid_indices)} images")

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    def process_split(split_name, indices, output_subdir):
        """Process a single split (train or valid)."""
        print(f"\n[Step 3] Processing {split_name} set...")

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": category_id, "name": category_name}
            ]
        }

        ann_id = 1  # Start annotation ID from 1

        for i, idx in enumerate(tqdm(indices, desc=f"Converting {split_name}")):
            image_path, mask_path = pairs[idx]

            # Process this pair
            image_info, annotations = process_single_pair(
                image_path=image_path,
                mask_path=mask_path,
                image_id=i,  # Sequential ID within this split
                annotation_id_start=ann_id,
                category_id=category_id,
                category_name=category_name
            )

            # Copy image to output directory
            dst_image_path = output_subdir / image_path.name
            shutil.copy2(image_path, dst_image_path)

            # Add to COCO data
            coco_data["images"].append(image_info)
            coco_data["annotations"].extend(annotations)

            ann_id += len(annotations)

        # Save COCO JSON
        json_path = output_subdir / "_annotations.coco.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n[OK] Saved {split_name} annotations:")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {[c['name'] for c in coco_data['categories']]}")
        print(f"  JSON file: {json_path}")

        # Verify RLE encoding
        print(f"\n  Sample annotation (first image):")
        if coco_data['annotations']:
            sample_ann = coco_data['annotations'][0]
            print(f"    BBox (COCO format [x,y,w,h]): {sample_ann['bbox']}")
            print(f"    Area: {sample_ann['area']} pixels")
            print(f"    Segmentation type: RLE (uncompressed)")
            print(f"    RLE size [h,w]: {sample_ann['segmentation']['size']}")
            print(f"    RLE counts length: {len(sample_ann['segmentation']['counts'])} runs")

        return coco_data

    # Process both splits
    train_coco = process_split("Training", train_indices, train_dir)
    valid_coco = process_split("Validation", valid_indices, valid_dir)

    # Summary
    print("\n" + "=" * 70)
    print("[OK] Dataset preparation complete!")
    print("=" * 70)
    print(f"\nOutput structure:")
    print(f"{output_dir}/")
    print(f"├── train/")
    print(f"│   ├── *.png (images)")
    print(f"│   └── _annotations.coco.json ({len(train_coco['annotations'])} annotations)")
    print(f"└── valid/")
    print(f"    ├── *.png (images)")
    print(f"    └── _annotations.coco.json ({len(valid_coco['annotations'])} annotations)")
    print(f"\nCategory mapping:")
    print(f"  ID {category_id} → '{category_name}'")
    print(f"\nNext step: Update your config file:")
    print(f'  training.data_dir: "{output_dir}"')
    print(f"\nThen run training:")
    print(f"  python train_sam3_lora_native.py --config configs/clothing_config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image+mask pairs to COCO format for SAM3 LoRA training"
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input directory containing image_XXX and mask_XXX files"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory (will create train/ and valid/ subdirectories)"
    )

    parser.add_argument(
        "--category-name", "-c",
        type=str,
        default="clothing",
        help="Category name (default: clothing)"
    )

    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="Category ID (default: 1)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )

    args = parser.parse_args()

    create_coco_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        category_name=args.category_name,
        category_id=args.category_id,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

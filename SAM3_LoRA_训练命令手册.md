# SAM3 LoRA 微调完整命令手册

> **项目路径**: `/root/code/SAM3_LoRA`  
> **任务**: 服装语义分割（clothing segmentation）  
> **最后更新**: 2026-04-10

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据转换](#2-数据转换)
3. [启动训练](#3-启动训练)
4. [模型验证](#4-模型验证)
5. [常见问题](#5-常见问题)

---

## 1. 环境准备

### 1.1 激活 Conda 环境

```bash
# 激活 sam3 环境（包含所有依赖）
conda activate sam3

# 验证环境
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 1.2 进入项目目录

```bash
cd /root/code/SAM3_LoRA
```

---

## 2. 数据转换

### 2.1 数据组织结构（原始数据）

**原始数据格式**：
```
sam3训练数据/
├── image_001.png    # 原图
├── mask_001.png     # 对应的 mask（二值图：白色=服装，黑色=背景）
├── image_002.png
├── mask_002.png
└── ...
```

**命名规则**：
- 原图：`image_XXX.png`（或 `.jpg`）
- Mask：`mask_XXX.png`（编号需与原图对应）

### 2.2 转换为 COCO 格式

```bash
# 基础命令（80% 训练集，20% 验证集）
python convert_mask_to_coco.py \
  --input-dir "sam3训练数据" \
  --output-dir "data" \
  --category-name "clothing" \
  --category-id 1 \
  --train-ratio 0.8

# 单行版本（方便复制）
python convert_mask_to_coco.py --input-dir "sam3训练数据" --output-dir "data" --category-name "clothing" --category-id 1 --train-ratio 0.8
```

**参数说明**：
- `--input-dir`: 原始数据目录（包含 image_XXX 和 mask_XXX 文件）
- `--output-dir`: 输出目录（会自动创建 train/ 和 valid/ 子目录）
- `--category-name`: 类别名称（用于文本提示，如 "clothing"）
- `--category-id`: 类别 ID（通常从 1 开始）
- `--train-ratio`: 训练集比例（0.8 = 80% 训练，20% 验证）

### 2.3 转换后的数据结构

```
data/
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   ├── ...
│   └── _annotations.coco.json    # COCO 格式标注文件（包含 RLE 编码的 mask）
└── valid/
    ├── image_004.png
    ├── image_007.png
    ├── image_008.png
    └── _annotations.coco.json
```

**验证转换结果**：
```bash
# 查看训练集图片数量
ls -l data/train/*.png | wc -l

# 查看验证集图片数量
ls -l data/valid/*.png | wc -l

# 检查 COCO JSON 文件格式
python -c "import json; data = json.load(open('data/train/_annotations.coco.json')); print('Images:', len(data['images'])); print('Annotations:', len(data['annotations'])); print('Categories:', data['categories'])"
```

---

## 3. 启动训练

### 3.1 检查配置文件

```bash
# 查看配置文件
cat configs/clothing_config.yaml

# 关键参数说明：
# - lora.rank: 16 (LoRA 秩，越大模型容量越大，显存占用越高)
# - training.batch_size: 1 (根据显存调整，建议 1-4)
# - training.num_epochs: 100 (训练轮数)
# - training.learning_rate: 5e-5 (学习率)
# - training.data_dir: "/root/code/SAM3_LoRA/data" (数据路径)
```

### 3.2 启动训练命令

```bash
# 基础训练命令
python train_sam3_lora_native.py --config configs/clothing_config.yaml

# 如果需要指定 GPU（多卡情况）
CUDA_VISIBLE_DEVICES=0 python train_sam3_lora_native.py --config configs/clothing_config.yaml

# 后台运行（推荐用于长时间训练）
nohup python train_sam3_lora_native.py --config configs/clothing_config.yaml > train.log 2>&1 &

# 查看训练日志
tail -f train.log
```

### 3.3 训练输出

**输出目录结构**：
```
outputs/clothing_lora/
├── best_lora_weights.pt    # 最佳模型（验证 loss 最低）⭐ 推荐使用
├── last_lora_weights.pt    # 最后一轮模型
└── val_stats.json          # 每轮训练/验证 loss 记录
```

**训练监控指标**：
- `train_loss`: 训练损失（应逐渐下降）
- `val_loss`: 验证损失（关键指标，越低越好）
- **最佳模型**: `best_lora_weights.pt`（自动保存 val_loss 最低的模型）

**查看训练进度**：
```bash
# 查看最新的 10 轮训练记录
tail -n 10 outputs/clothing_lora/val_stats.json

# 查看最佳验证 loss
python -c "import json; data = [json.loads(line) for line in open('outputs/clothing_lora/val_stats.json')]; best = min(data, key=lambda x: x['val_loss']); print(f\"Best epoch: {best['epoch']}, Val loss: {best['val_loss']:.4f}\")"
```

---

## 4. 模型验证

### 4.1 运行验证命令

```bash
# 完整验证命令（对比 Base vs LoRA）
python evaluate_models.py \
  --data-dir data/valid \
  --weights outputs/clothing_lora/best_lora_weights.pt \
  --config configs/clothing_config.yaml \
  --output-dir outputs/clothing_lora/evaluation_results \
  --prompt "clothing" \
  --threshold 0.5

# 单行版本（方便复制）
python evaluate_models.py --data-dir data/valid --weights outputs/clothing_lora/best_lora_weights.pt --config configs/clothing_config.yaml --output-dir outputs/clothing_lora/evaluation_results --prompt clothing --threshold 0.5
```

**参数说明**：
- `--data-dir`: 验证集目录（包含图片和 `_annotations.coco.json`）
- `--weights`: LoRA 权重文件路径（使用 `best_lora_weights.pt`）
- `--config`: 配置文件路径
- `--output-dir`: 输出目录（保存对比结果）
- `--prompt`: 文本提示词（训练时使用的类别名称）
- `--threshold`: 检测阈值（0.5 = 中等严格度，可调整 0.3-0.7）

### 4.2 验证输出

**输出目录结构**：
```
outputs/clothing_lora/evaluation_results/
├── ground_truth/              # Ground Truth masks（真实标注）
│   ├── image_004_gt_mask.png  # 白色 = 服装区域
│   ├── image_007_gt_mask.png
│   └── image_008_gt_mask.png
├── base_model/                # 原始 SAM3 模型输出（无 LoRA）
│   ├── image_004_base_mask.png
│   ├── image_007_base_mask.png
│   └── image_008_base_mask.png
├── lora_model/                # LoRA 微调后模型输出 ⭐
│   ├── image_004_lora_mask.png
│   ├── image_007_lora_mask.png
│   └── image_008_lora_mask.png
└── evaluation_summary.json    # 数值对比结果
```

### 4.3 查看验证结果

```bash
# 查看数值对比结果
cat outputs/clothing_lora/evaluation_results/evaluation_summary.json

# 查看图片对比（需要图形界面）
# 或将图片下载到本地查看
ls -lh outputs/clothing_lora/evaluation_results/lora_model/
```

### 4.4 调整阈值（可选）

```bash
# 更低的阈值 = 更多检测（可能包含误报，适合召回率优先）
python evaluate_models.py --data-dir data/valid --weights outputs/clothing_lora/best_lora_weights.pt --config configs/clothing_config.yaml --output-dir outputs/clothing_lora/evaluation_results_low_threshold --prompt clothing --threshold 0.3

# 更高的阈值 = 更少检测（更精确，适合精确度优先）
python evaluate_models.py --data-dir data/valid --weights outputs/clothing_lora/best_lora_weights.pt --config configs/clothing_config.yaml --output-dir outputs/clothing_lora/evaluation_results_high_threshold --prompt clothing --threshold 0.7
```

---

## 5. 常见问题

### 5.1 数据转换问题

**Q: 提示 "No image-mask pairs found"**
```bash
# 检查文件命名是否正确
ls sam3训练数据/ | head -10

# 确保命名格式为：image_XXX.png 和 mask_XXX.png
# 如果命名不同，需要修改 convert_mask_to_coco.py 中的 find_image_mask_pairs() 函数
```

**Q: RLE 编码错误 "Expected bytes, got list"**
```bash
# 确保使用最新版本的 convert_mask_to_coco.py（已修复）
# 重新运行数据转换
python convert_mask_to_coco.py --input-dir "sam3训练数据" --output-dir "data" --category-name "clothing" --category-id 1 --train-ratio 0.8
```

### 5.2 训练问题

**Q: CUDA out of memory**
```bash
# 方案 1: 减小 batch_size（在 configs/clothing_config.yaml 中修改）
# training:
#   batch_size: 1

# 方案 2: 使用轻量配置
python train_sam3_lora_native.py --config configs/light_lora_config.yaml

# 方案 3: 降低 LoRA rank
# 在配置文件中修改：
# lora:
#   rank: 8  # 从 16 降到 8
```

**Q: 训练 loss 不下降**
```bash
# 检查学习率是否过大或过小
# 在配置文件中调整：
# training:
#   learning_rate: 1e-5  # 尝试更小的学习率

# 检查数据是否正确加载
python -c "import json; data = json.load(open('data/train/_annotations.coco.json')); print('Images:', len(data['images'])); print('Annotations:', len(data['annotations']))"
```

### 5.3 验证问题

**Q: "unrecognized arguments" 错误**
```bash
# 确保命令格式正确（不要有多余空格或换行符）
# 使用单行命令：
python evaluate_models.py --data-dir data/valid --weights outputs/clothing_lora/best_lora_weights.pt --config configs/clothing_config.yaml --output-dir outputs/clothing_lora/evaluation_results --prompt clothing --threshold 0.5
```

**Q: 模型加载失败**
```bash
# 检查权重文件是否存在
ls -lh outputs/clothing_lora/best_lora_weights.pt

# 检查配置文件路径
ls -lh configs/clothing_config.yaml

# 检查 SAM3 基础模型路径（在配置文件中）
grep "checkpoint_path" configs/clothing_config.yaml
```

---

## 📊 快速参考卡片

### 完整工作流程（一键复制）

```bash
# 1. 进入项目目录
cd /root/code/SAM3_LoRA

# 2. 激活环境
conda activate sam3

# 3. 数据转换
python convert_mask_to_coco.py --input-dir "sam3训练数据" --output-dir "data" --category-name "clothing" --category-id 1 --train-ratio 0.8

# 4. 启动训练
python train_sam3_lora_native.py --config configs/clothing_config.yaml

# 5. 模型验证
python evaluate_models.py --data-dir data/valid --weights outputs/clothing_lora/best_lora_weights.pt --config configs/clothing_config.yaml --output-dir outputs/clothing_lora/evaluation_results --prompt clothing --threshold 0.5
```

### 关键文件路径

| 文件类型 | 路径 |
|---------|------|
| **配置文件** | `configs/clothing_config.yaml` |
| **原始数据** | `sam3训练数据/` |
| **转换后数据** | `data/train/`, `data/valid/` |
| **训练权重** | `outputs/clothing_lora/best_lora_weights.pt` |
| **训练日志** | `outputs/clothing_lora/val_stats.json` |
| **验证结果** | `outputs/clothing_lora/evaluation_results/` |

---

## 🎯 预期结果

### 训练成功标志
- ✅ `val_loss` 逐渐下降（从 ~2.0 降到 < 1.0）
- ✅ 生成 `best_lora_weights.pt` 文件（约 70 MB）
- ✅ 训练日志正常记录

### 验证成功标志
- ✅ LoRA 模型检测数量 ≈ GT 数量
- ✅ LoRA mask 边界更精确
- ✅ Base 模型可能检测失败，LoRA 成功检测

---

## 📝 备注

- **训练时长**: 11 张图片，100 epochs，约 1-2 小时（取决于 GPU）
- **显存需求**: 建议 ≥ 8 GB（batch_size=1）
- **数据建议**: 建议 ≥ 50 张图片以获得更好效果
- **提示词**: 训练和验证时使用相同的提示词（"clothing"）

---

**文档版本**: v1.0  
**最后更新**: 2026-04-10  
**作者**: SAM3 LoRA 微调项目

# FER-ResNet50 人脸表情识别项目

这是一个基于 `ResNet50` 的人脸表情识别（FER, Facial Expression Recognition）项目模板，包含：

- 训练脚本：`train.py`
- 单图推理：`infer.py`
- 实时摄像头识别：`webcam_demo.py`


## 1. 环境安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2. 数据集组织格式

项目使用按类别分文件夹的形式：

```text
data/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  val/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

每个类别目录下放对应图片。

## 3. 开始训练

### 基础训练（使用最佳默认配置）

```bash
python train.py ^
  --train_dir data/train ^
  --val_dir data/val ^
  --save_dir checkpoints ^
  --epochs 20 ^
  --batch_size 32 ^
  --lr 0.01
```

**说明**：默认使用 SGD + Nesterov 动量 + 余弦退火学习率调度器 + SE 注意力机制，这是获得最佳性能的配置。

### 使用 AdamW 优化器（快速上手）

```bash
python train.py ^
  --train_dir data/train ^
  --val_dir data/val ^
  --save_dir checkpoints ^
  --epochs 20 ^
  --batch_size 32 ^
  --lr 1e-4 ^
  --optimizer adamw ^
  --scheduler none
```

### 使用注意力机制（提升性能）

```bash
python train.py ^
  --train_dir data/train ^
  --val_dir data/val ^
  --save_dir checkpoints ^
  --epochs 30 ^
  --batch_size 32 ^
  --lr 0.01 ^
  --attention se
```

**说明**：添加 SE 注意力机制可以让模型更关注重要的特征通道，通常能提升 1-3% 的准确率。

### 高级训练配置（完整参数示例）

```bash
python train.py ^
  --train_dir data/train ^
  --val_dir data/val ^
  --save_dir checkpoints ^
  --epochs 50 ^
  --batch_size 32 ^
  --lr 0.01 ^
  --optimizer sgd ^
  --momentum 0.9 ^
  --weight_decay 1e-4 ^
  --scheduler cosine ^
  --freeze_backbone_epochs 5 ^
  --use_amp
```

**参数说明：**
- `--optimizer`: 优化器类型
  - `sgd`（默认，推荐）：带动量的随机梯度下降，通常能获得更好的泛化性能
  - `adamw`：自适应学习率优化器，适合快速上手和调试
- `--momentum`: SGD 动量系数（默认 0.9），使用 Nesterov 动量加速收敛（仅在使用 SGD 时生效）
- `--scheduler`: 学习率调度器
  - `cosine`（默认，推荐）：余弦退火，学习率按余弦曲线逐渐衰减至初始值的 1%
  - `plateau`：基于验证损失自适应调整，当损失不再下降时降低学习率至 50%
  - `step`：固定步长衰减，每 7 个 epoch 降低为原来的 10%
  - `none`：不使用学习率调整
- `--weight_decay`: 权重衰减系数，L2 正则化防止过拟合（默认 1e-4）
- `--attention`: 注意力机制类型
  - `se`（默认，推荐）：Squeeze-and-Excitation 注意力，学习通道间的重要性关系，通常提升 1-3% 准确率
  - `cbam`：Convolutional Block Attention Module，同时考虑通道和空间注意力
  - `none`：不使用注意力机制

训练输出：

- `checkpoints/best.pt`：验证集最优模型
- `checkpoints/last.pt`：最后一个 epoch 的模型
- `checkpoints/classes.json`：类别映射

## 4. 单张图片推理

```bash
python infer.py --image demo.jpg --ckpt checkpoints/best.pt --classes checkpoints/classes.json
```

## 5. 摄像头实时识别

### 基础使用（默认优化配置）

```bash
python webcam_demo.py --ckpt checkpoints/best.pt --classes checkpoints/classes.json
```

**默认启用**：
- ✅ 帧跳过策略（每3帧推理一次，提升 FPS）
- ✅ 预测结果平滑（消除闪烁）
- ✅ 置信度过滤（只显示高置信度结果）
- ✅ Haar Cascade 人脸检测（轻量级）

### 高级配置

```bash
python webcam_demo.py ^
  --ckpt checkpoints/best.pt ^
  --classes checkpoints/classes.json ^
  --camera_id 0 ^
  --confidence_threshold 0.6 ^
  --frame_skip 3 ^
  --smooth_window 7 ^
  --use_mediapipe
```

**参数说明**：
- `--confidence_threshold`: 置信度阈值（默认 0.5），低于此值的预测不显示
- `--frame_skip`: 帧跳过数（默认 2），值越大 FPS 越高但延迟增加
- `--smooth_window`: 平滑窗口大小（默认 5），值越大越稳定但响应变慢

按 `q` 退出窗口。

## 6. 说明与可扩展项

- **已支持**：
  - 动量优化器（SGD + Nesterov）
  - 学习率自动调整（余弦退火/自适应/固定步长）
  - 注意力机制（SE/CBAM）
  - 预测结果平滑和置信度过滤
  - 实时摄像头演示（Haar Cascade 人脸检测）
- 可升级为 `RetinaFace` 或 `MediaPipe 0.9.x` 来提升检测精度。
- 可增加早停、混淆矩阵等功能。


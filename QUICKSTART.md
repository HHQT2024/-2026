# 🚀 FER-ResNet50 快速开始指南

## ⚡ 5分钟快速上手

### 1️⃣ 环境安装

```bash
# 克隆或进入项目目录
cd fer-resnet50

# 创建虚拟环境
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

### 2️⃣ 准备数据集

确保你的数据集按以下结构组织：

```
data/
├── train/
│   ├── angry/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── happy/
│   ├── sad/
│   └── ... (其他类别)
└── val/
    ├── angry/
    ├── happy/
    └── ...
```

**支持的表情类别**: angry, disgust, fear, happy, neutral, sad, surprise

---

### 3️⃣ 开始训练

#### 方式 A: 使用默认配置（推荐新手）

```bash
python train.py \
  --train_dir data/train \
  --val_dir data/val \
  --save_dir checkpoints \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.01
```

#### 方式 B: 快速测试（验证环境）

```bash
python train.py \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 5 \
  --batch_size 16 \
  --max_train_batches 10 \
  --max_val_batches 5
```

#### 方式 C: 高性能训练

```bash
python train.py \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.01 \
  --optimizer sgd \
  --scheduler cosine \
  --attention se \
  --freeze_backbone_epochs 5 \
  --use_amp \
  --early_stopping_patience 10
```

**训练输出**:
- `checkpoints/best.pt` - 验证集最优模型
- `checkpoints/last.pt` - 最后一个 epoch 的模型
- `checkpoints/classes.json` - 类别映射

---

### 4️⃣ 单张图片推理

```bash
python infer.py \
  --image test.jpg \
  --ckpt checkpoints/best.pt \
  --classes checkpoints/classes.json
```

**输出示例**:
```
预测结果：happy
各类别概率：
  angry: 0.0012
  disgust: 0.0003
  fear: 0.0008
  happy: 0.9876
  neutral: 0.0045
  sad: 0.0021
  surprise: 0.0035
```

---

### 5️⃣ 摄像头实时识别

```bash
python webcam_demo.py \
  --ckpt checkpoints/best.pt \
  --classes checkpoints/classes.json
```

**快捷键**:
- `q` - 退出

**提示**: 
- 确保光线充足
- 正对摄像头
- 保持适当距离（30-50cm）

---

## 📊 进阶用法

### 批量推理

对文件夹中所有图片进行识别并生成报告：

```bash
python batch_infer.py \
  --input_dir test_images/ \
  --ckpt checkpoints/best.pt \
  --classes checkpoints/classes.json \
  --output_csv results.csv
```

**输出**: CSV 文件包含每张图片的预测结果和所有类别的概率

---

### 模型导出

导出为 ONNX 或 TorchScript 格式用于部署：

```bash
# 导出两种格式
python export_model.py \
  --ckpt checkpoints/best.pt \
  --classes checkpoints/classes.json \
  --output_dir exported_models

# 仅导出 ONNX
python export_model.py \
  --ckpt checkpoints/best.pt \
  --classes checkpoints/classes.json \
  --format onnx
```

---

### 测试模型

验证模型是否正常工作：

```bash
python test_model.py
```

---

## 🔧 常用参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--optimizer` | sgd | 优化器 (sgd/adamw) |
| `--scheduler` | cosine | 学习率调度器 (cosine/plateau/step/none) |
| `--attention` | se | 注意力机制 (se/cbam/none) |
| `--freeze_backbone_epochs` | 0 | 冻结骨干网络的轮数 |
| `--use_amp` | False | 启用混合精度训练 |
| `--early_stopping_patience` | 7 | 早停耐心值 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence_threshold` | 0.5 | 置信度阈值 |
| `--frame_skip` | 2 | 帧跳过数（提升 FPS） |
| `--smooth_window` | 5 | 平滑窗口大小 |

---

## 💡 性能调优建议

### 如果显存不足
```bash
python train.py \
  --batch_size 16 \
  --use_amp \
  --num_workers 2
```

### 如果想加快训练速度
```bash
python train.py \
  --batch_size 64 \
  --use_amp \
  --num_workers 8 \
  --optimizer adamw
```

### 如果想获得更高准确率
```bash
python train.py \
  --epochs 50 \
  --lr 0.01 \
  --optimizer sgd \
  --scheduler cosine \
  --attention cbam \
  --weight_decay 5e-4 \
  --early_stopping_patience 10
```

---

## ❓ 常见问题

### Q: 训练时出现 "CUDA out of memory"？
**A**: 减小 batch_size 或启用混合精度：
```bash
--batch_size 16 --use_amp
```

### Q: 验证准确率远低于训练准确率？
**A**: 可能过拟合，尝试：
- 增加权重衰减：`--weight_decay 5e-4`
- 使用早停：`--early_stopping_patience 7`
- 减少训练轮数

### Q: 摄像头检测不到人脸？
**A**: 
- 确保光线充足
- 正对摄像头
- 调整距离（30-50cm）
- 检查摄像头权限

### Q: 如何查看训练曲线？
**A**: 可以添加 TensorBoard 支持（见 IMPROVEMENTS.md）

---

## 📚 更多资源

- **详细改进建议**: 查看 [IMPROVEMENTS.md](IMPROVEMENTS.md)
- **项目文档**: 查看 [README.md](README.md)
- **代码结构**: 
  - `src/model.py` - 模型定义
  - `src/dataset.py` - 数据加载
  - `train.py` - 训练脚本
  - `infer.py` - 单图推理
  - `webcam_demo.py` - 实时演示

---

## 🎯 下一步

1. ✅ 完成基础训练和推理
2. 📈 尝试不同的超参数配置
3. 🔍 分析混淆矩阵，了解模型弱点
4. 🚀 导出模型并部署到生产环境
5. 🌐 创建 Web 界面（可选）

---

**祝你使用愉快！** 🎉

如有问题，请查看详细文档或提交 Issue。

"""
批量推理工具 - 对文件夹中的所有图片进行表情识别
生成 CSV 报告文件
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.dataset import build_transforms
from src.model import FERResNet50, load_checkpoint
from src.utils import configure_utf8_stdio


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser("批量表情识别工具")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--classes", type=str, required=True, help="类别文件路径")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="输出 CSV 文件路径")
    parser.add_argument(
        "--attention",
        type=str,
        default="auto",
        choices=["auto", "none", "se", "cbam"],
        help="注意力机制类型（auto=自动检测，需与训练时一致，默认：auto）"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小（默认：32）")
    return parser.parse_args()


def find_images(input_dir: Path):
    """查找目录下所有图片文件"""
    images = []
    for ext in IMG_EXTS:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def main():
    configure_utf8_stdio()
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载类别信息
    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)["classes"]
    
    # 自动检测注意力机制类型
    attention_type = args.attention
    if attention_type == "auto":
        try:
            ckpt_temp = torch.load(args.ckpt, map_location=device, weights_only=False)
            if "model_config" in ckpt_temp:
                attention_type = ckpt_temp["model_config"].get("attention_type", "none")
                print(f"📋 从检查点读取配置: attention_type={attention_type}")
            else:
                # 旧版本检查点，需要自动检测
                state_dict = ckpt_temp.get("model_state_dict", {})
                has_se = any("attention.fc" in key for key in state_dict.keys())
                has_cbam = any("attention.channel_attention" in key for key in state_dict.keys())
                if has_se:
                    attention_type = "se"
                elif has_cbam:
                    attention_type = "cbam"
                else:
                    attention_type = "none"
                print(f"🔍 自动检测到: attention_type={attention_type}")
            del ckpt_temp
        except Exception as e:
            print(f"⚠️  无法自动检测配置，使用默认值 'none': {e}")
            attention_type = "none"
    
    # 加载模型
    model = FERResNet50(
        num_classes=len(class_names),
        use_pretrained=False,
        attention_type=attention_type
    ).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()
    
    transform = build_transforms(is_train=False)
    
    # 查找图片
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    images = find_images(input_dir)
    if not images:
        print(f"⚠️  在 {input_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(images)} 张图片")
    print(f"开始批量推理...\n")
    
    # 批量处理
    results = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(images), batch_size), desc="处理进度"):
        batch_paths = images[i:i + batch_size]
        batch_tensors = []
        valid_paths = []
        
        # 加载和预处理图片
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)
                batch_tensors.append(tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"\n⚠️  跳过无效图片 {img_path}: {e}")
        
        if not batch_tensors:
            continue
        
        # 堆叠成批次
        batch = torch.stack(batch_tensors).to(device)
        
        # 推理
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # 收集结果
        for j, img_path in enumerate(valid_paths):
            pred_idx = preds[j].item()
            pred_name = class_names[pred_idx]
            pred_conf = probs[j][pred_idx].item()
            
            # 获取所有类别的概率
            all_probs = {class_names[k]: probs[j][k].item() for k in range(len(class_names))}
            
            results.append({
                "image_path": str(img_path),
                "predicted_emotion": pred_name,
                "confidence": f"{pred_conf:.4f}",
                **{f"prob_{k}": f"{v:.4f}" for k, v in all_probs.items()}
            })
    
    # 保存结果为 CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if results:
        fieldnames = ["image_path", "predicted_emotion", "confidence"]
        fieldnames.extend([f"prob_{name}" for name in class_names])
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ 推理完成！")
        print(f"   总图片数: {len(images)}")
        print(f"   成功处理: {len(results)}")
        print(f"   结果已保存到: {output_path}")
        
        # 统计各类别数量
        emotion_counts = {}
        for r in results:
            emotion = r["predicted_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n📊 表情分布统计:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            print(f"   {emotion:12s}: {count:4d} ({percentage:5.1f}%)")
    else:
        print("\n❌ 没有成功处理任何图片")


if __name__ == "__main__":
    main()

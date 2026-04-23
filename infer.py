import argparse
import json

import torch
from PIL import Image

from src.dataset import build_transforms
from src.model import FERResNet50, load_checkpoint
from src.utils import configure_utf8_stdio


def parse_args():
    parser = argparse.ArgumentParser("人脸表情识别单图推理")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--classes", type=str, required=True, help="类别文件 classes.json 路径")
    parser.add_argument(
        "--attention",
        type=str,
        default="auto",
        choices=["auto", "none", "se", "cbam"],
        help="注意力机制类型（auto=自动检测，需与训练时一致，默认：auto）",
    )
    return parser.parse_args()


def main():
    configure_utf8_stdio()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)["classes"]

    # 自动检测注意力机制类型（如果用户未指定）
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
            print(f"⚠️  无法自动检测配置，使用默认值 'se': {e}")
            attention_type = "se"

    model = FERResNet50(
        num_classes=len(class_names), 
        use_pretrained=False,
        attention_type=attention_type
    ).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    x = build_transforms(is_train=False)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    print(f"预测结果：{class_names[pred]}")
    print("各类别概率：")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i].item():.4f}")


if __name__ == "__main__":
    main()


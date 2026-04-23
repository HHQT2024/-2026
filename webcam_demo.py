import argparse
import json
import time
from collections import deque

import cv2
import torch
from PIL import Image

from src.dataset import build_transforms
from src.model import FERResNet50, load_checkpoint
from src.utils import configure_utf8_stdio


def parse_args():
    parser = argparse.ArgumentParser("人脸表情识别实时摄像头演示")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="模型权重路径（默认：checkpoints/best.pt）")
    parser.add_argument("--classes", type=str, default="checkpoints/classes.json", help="类别文件 classes.json 路径（默认：checkpoints/classes.json）")
    parser.add_argument("--camera_id", type=int, default=0, help="摄像头ID（默认：0）")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="置信度阈值，低于此值的预测不显示（默认：0.5）",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=2,
        help="帧跳过数，每 N+1 帧推理一次，提升性能（默认：2，即每3帧推理一次）",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="平滑窗口大小，用于稳定预测结果（默认：5）",
    )
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

    model = FERResNet50(
        num_classes=len(class_names), 
        use_pretrained=False,
        attention_type=attention_type
    ).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    transform = build_transforms(is_train=False)
    
    # 初始化人脸检测器
    # 注意：MediaPipe 0.10+ 已移除 solutions 模块，暂时使用 Haar Cascade
    # 如需更高精度的人脸检测，可以安装旧版 MediaPipe (0.9.x) 或使用 RetinaFace
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("使用 Haar Cascade 人脸检测")
    print("提示：如需更高精度，可安装 mediapipe==0.9.3 或升级为 RetinaFace")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头。")
    
    # 设置摄像头分辨率（可选，提升性能）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 用于平滑预测的历史记录
    prediction_history = deque(maxlen=args.smooth_window)
    frame_count = 0
    last_prediction = None
    
    # FPS 计算相关
    prev_time = time.time()
    fps = 0.0
    
    print(f"配置信息：")
    print(f"  - 置信度阈值: {args.confidence_threshold}")
    print(f"  - 帧跳过: 每 {args.frame_skip + 1} 帧推理一次")
    print(f"  - 平滑窗口: {args.smooth_window} 帧")
    print(f"  - 按 'q' 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 计算 FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        frame_count += 1
        display_frame = frame.copy()
        
        # 帧跳过策略：只在特定帧进行推理
        should_infer = (frame_count % (args.frame_skip + 1) == 0)
        
        if should_infer:
            faces_detected = []
            
            # Haar Cascade 人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60)
            )
            faces_detected = [(x, y, w, h) for (x, y, w, h) in faces]
            
            # 对每个人脸进行表情识别
            current_predictions = []
            for (x, y, w, h) in faces_detected:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                    
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                inp = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_name = class_names[pred_idx]
                    pred_conf = probs[pred_idx].item()
                    
                    # 置信度过滤
                    if pred_conf >= args.confidence_threshold:
                        current_predictions.append({
                            'bbox': (x, y, w, h),
                            'name': pred_name,
                            'confidence': pred_conf
                        })
            
            # 更新预测历史
            if current_predictions:
                prediction_history.append(current_predictions)
                last_prediction = current_predictions
        
        # 使用最近的预测结果进行显示（平滑效果）
        if prediction_history:
            # 获取最近的预测
            recent_preds = prediction_history[-1]
            
            for pred in recent_preds:
                x, y, w, h = pred['bbox']
                pred_name = pred['name']
                pred_conf = pred['confidence']
                
                # 绘制边界框
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                
                # 绘制标签背景
                label = f"{pred_name} {pred_conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(
                    display_frame,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    (0, 220, 0),
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    display_frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
        
        # 显示 FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            display_frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("人脸表情识别 ResNet50", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


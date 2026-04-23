import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EmotionFolderDataset, build_transforms, discover_classes
from src.model import FERResNet50
from src.utils import accuracy, configure_utf8_stdio, save_json, set_seed


def set_backbone_trainable(model: FERResNet50, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    # 分类头始终保持可训练
    for p in model.backbone.fc.parameters():
        p.requires_grad = True


def build_optimizer(model: FERResNet50, lr: float, optimizer_type: str = "adamw", momentum: float = 0.9, weight_decay: float = 1e-4):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            trainable_params, 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            nesterov=True  # 使用 Nesterov 动量加速收敛
        )
    else:  # adamw (default)
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


from typing import Optional, Union

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    max_batches: Optional[int] = None,
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: Optional[torch.amp.GradScaler] = None,
):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    iterator = tqdm(loader, desc="训练中" if train else "验证中", leave=False)
    for batch_idx, (images, labels) in enumerate(iterator):
        images = images.to(device)
        labels = labels.to(device)

        amp_ctx = (
            torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(train):
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        total_count += bs
        iterator.set_postfix(loss=total_loss / total_count, acc=total_acc / total_count)

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return total_loss / total_count, total_acc / total_count


def build_scheduler(optimizer, scheduler_type: str, epochs: int, lr: float):
    """
    构建学习率调度器
    - cosine: 余弦退火（推荐）
    - plateau: 基于验证损失自适应调整
    - step: 固定步长衰减
    """
    if scheduler_type.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type.lower() == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    elif scheduler_type.lower() == "step":
        return StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser("基于 ResNet50 的人脸表情识别训练")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/train",
        help="训练集目录（默认：data/train）",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="data/val",
        help="验证/测试集目录（默认：data/val）",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="前 N 个 epoch 仅训练分类层，之后自动解冻全量微调（0 表示不冻结）",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="启用混合精度训练（CPU 默认 bfloat16，若不支持会自动回退）",
    )
    parser.add_argument("--max_train_batches", type=int, default=0, help="每个 epoch 最多训练多少个 batch（0 表示不限制）")
    parser.add_argument("--max_val_batches", type=int, default=0, help="每个 epoch 最多验证多少个 batch（0 表示不限制）")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adamw", "sgd"],
        help="优化器类型：sgd（默认，推荐）或 adamw",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD 动量系数（仅在使用 SGD 优化器时生效，默认：0.9）",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "step", "none"],
        help="学习率调度器：cosine（默认，推荐）、plateau（自适应）、step（固定步长）、none（不使用）",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减系数，用于防止过拟合（默认：1e-4）",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="se",
        choices=["none", "se", "cbam"],
        help="注意力机制类型：se（默认，推荐）、cbam（Convolutional Block Attention）、none（不使用）",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=7,
        help="早停耐心值，连续 N 个 epoch 验证准确率无提升则停止训练（0 表示禁用，默认：7）",
    )
    return parser.parse_args()


def resolve_data_path(path_str: str) -> str:
    """
    若传入相对路径，则按 train.py 所在目录解析，避免受当前终端 cwd 影响。
    """
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    project_root = Path(__file__).resolve().parent
    return str((project_root / p).resolve())


def main():
    configure_utf8_stdio()
    args = parse_args()
    set_seed(args.seed)

    args.train_dir = resolve_data_path(args.train_dir)
    args.val_dir = resolve_data_path(args.val_dir)
    args.save_dir = resolve_data_path(args.save_dir)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"训练集目录：{args.train_dir}")
    print(f"验证/测试集目录：{args.val_dir}")
    print(f"模型保存目录：{args.save_dir}")
    print(f"运行设备：{device}")

    class_names = discover_classes(args.train_dir)
    save_json({"classes": class_names}, str(save_dir / "classes.json"))

    train_ds = EmotionFolderDataset(args.train_dir, class_names, transform=build_transforms(is_train=True))
    val_ds = EmotionFolderDataset(args.val_dir, class_names, transform=build_transforms(is_train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FERResNet50(
        num_classes=len(class_names), 
        use_pretrained=not args.no_pretrained,
        attention_type=args.attention
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = bool(args.use_amp)
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    if amp_enabled:
        if device.type == "cpu":
            print("已启用混合精度：CPU bfloat16 autocast。")
        else:
            print("已启用混合精度：CUDA float16 + GradScaler。")

    if args.freeze_backbone_epochs > 0:
        print(f"将冻结骨干网络前 {args.freeze_backbone_epochs} 个 epoch，仅训练分类层。")

    # 早停机制初始化
    early_stopping_enabled = args.early_stopping_patience > 0
    if early_stopping_enabled:
        print(f"已启用早停机制：耐心值={args.early_stopping_patience}")
    
    best_val_acc = -1.0
    no_improve_count = 0  # 连续无提升的 epoch 计数
    backbone_frozen = False
    optimizer = None
    scheduler = None
    for epoch in range(1, args.epochs + 1):
        should_freeze = epoch <= args.freeze_backbone_epochs
        if should_freeze != backbone_frozen or optimizer is None:
            set_backbone_trainable(model, trainable=not should_freeze)
            optimizer = build_optimizer(model, args.lr, args.optimizer, args.momentum, args.weight_decay)
            
            # 重新构建调度器（如果使用了 plateau，需要传入 optimizer）
            if args.scheduler.lower() != "none":
                scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.lr)
                if epoch == 1:
                    print(f"已启用学习率调度器：{args.scheduler}")
            
            backbone_frozen = should_freeze
            if backbone_frozen:
                print(f"轮次 {epoch}: 骨干网络已冻结，仅训练分类层。")
            else:
                print(f"轮次 {epoch}: 骨干网络已解冻，开始全量微调。")
            
            # 打印优化器信息
            print(f"优化器: {args.optimizer.upper()}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
            if args.optimizer.lower() == "sgd":
                print(f"动量系数: {args.momentum} (Nesterov)")

        max_train = None if args.max_train_batches <= 0 else args.max_train_batches
        max_val = None if args.max_val_batches <= 0 else args.max_val_batches

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            max_batches=max_train,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            max_batches=max_val,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
            amp_dtype=amp_dtype,
            scaler=None,
        )

        print(
            f"轮次 [{epoch}/{args.epochs}] "
            f"训练损失={train_loss:.4f} 训练准确率={train_acc:.4f} "
            f"验证损失={val_loss:.4f} 验证准确率={val_acc:.4f}"
        )

        # 更新学习率调度器
        if scheduler is not None:
            if args.scheduler.lower() == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": class_names,
            "model_config": {
                "attention_type": args.attention,
                "num_classes": len(class_names),
            },
        }
        torch.save(ckpt, save_dir / "last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0  # 重置计数器
            torch.save(ckpt, save_dir / "best.pt")
            print(f"✨ 已保存最优模型：验证准确率={best_val_acc:.4f}")
        else:
            no_improve_count += 1
            if early_stopping_enabled:
                print(f"⚠️  验证准确率未提升 ({no_improve_count}/{args.early_stopping_patience})")
                
                # 检查是否触发早停
                if no_improve_count >= args.early_stopping_patience:
                    print(f"\n🛑 早停触发！连续 {args.early_stopping_patience} 个 epoch 无提升")
                    print(f"最佳验证准确率: {best_val_acc:.4f} (第 {epoch - args.early_stopping_patience} 轮)")
                    break

    print("\n✅ 训练完成。")
    if early_stopping_enabled and no_improve_count > 0:
        print(f"   总轮次: {epoch}, 最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()


import os
import random
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vaipe_detection_dataset import VaipeDetectionDataset, collate_fn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = 2):
    """
    num_classes = 2 gồm:
    - class 0: background
    - class 1: pill
    """
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception:
        model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return train_idx, val_idx


def move_targets_to_device(targets, device):
    moved = []
    for t in targets:
        moved.append({k: v.to(device) for k, v in t.items()})
    return moved


def train_one_epoch(model, loader, optimizer, device, log_interval: int = 100) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        running_loss += loss_value
        num_batches += 1

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            avg_so_far = running_loss / max(1, num_batches)
            print(
                f"  train batch {batch_idx}/{total_batches} | "
                f"loss={loss_value:.4f} | avg={avg_so_far:.4f}",
                flush=True,
            )

    return running_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_loss(model, loader, device, log_interval: int = 100) -> float:
    """
    TorchVision detection model chỉ trả về loss khi model ở train mode và có targets.
    Vì vậy validation loss sẽ được tính trong train mode nhưng dưới no_grad().
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        loss_value = float(loss.item())
        running_loss += loss_value
        num_batches += 1

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            avg_so_far = running_loss / max(1, num_batches)
            print(
                f"  val   batch {batch_idx}/{total_batches} | "
                f"loss={loss_value:.4f} | avg={avg_so_far:.4f}",
                flush=True,
            )

    return running_loss / max(1, num_batches)


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": vars(args),
        "num_classes": 2,
        "class_names": ["background", "pill"],
    }
    torch.save(ckpt, path)


def main(args):
    set_seed(args.seed)

    if not os.path.isdir(args.train_root):
        raise FileNotFoundError(f"Không tìm thấy train_root: {args.train_root}")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    dataset = VaipeDetectionDataset(args.train_root)
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.out_dir, "fasterrcnn_vaipe_best.pth")
    last_ckpt_path = os.path.join(args.out_dir, "fasterrcnn_vaipe_last.pth")
    history_path = os.path.join(args.out_dir, "train_log.txt")

    print(f"Total samples : {len(dataset)}", flush=True)
    print(f"Train samples : {len(train_set)}", flush=True)
    print(f"Val samples   : {len(val_set)}", flush=True)
    print(f"Train batches : {len(train_loader)}", flush=True)
    print(f"Val batches   : {len(val_loader)}", flush=True)
    print(f"Output dir    : {args.out_dir}", flush=True)
    print(f"Log interval  : every {args.log_interval} batches", flush=True)

    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 80, flush=True)
        print(f"Starting epoch {epoch}/{args.epochs} | lr={current_lr:.6f}", flush=True)
        print("=" * 80, flush=True)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            log_interval=args.log_interval,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            log_interval=args.log_interval,
        )

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            args=args,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
            )
            print(f"Saved best checkpoint -> {best_ckpt_path}", flush=True)

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.8f}\n")

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}",
            flush=True,
        )

        scheduler.step()

    print("\nTraining detector completed.", flush=True)
    print("Best checkpoint:", best_ckpt_path, flush=True)
    print("Last checkpoint:", last_ckpt_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN detector for VAIPE pill localization"
    )
    parser.add_argument("--train_root", type=str, required=True, help="Path to public_train")
    parser.add_argument("--out_dir", type=str, default="outputs_faster_rcnn_vaipe")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=4)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()
    main(args)

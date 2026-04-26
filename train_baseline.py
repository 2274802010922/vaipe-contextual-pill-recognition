import os
import json
import argparse
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import timm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_column(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col is not None and user_col != "":
        if user_col not in df.columns:
            raise RuntimeError(
                f"Không tìm thấy image column '{user_col}'. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )
        return user_col

    candidates = [
        "pill_crop_path",
        "crop_path",
        "pill_image_path",
        "image_path",
        "pill_image",
        "image",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Không tự tìm được cột ảnh. "
        f"Các cột hiện có: {df.columns.tolist()}\n"
        "Hãy truyền thêm --image_col tên_cột_ảnh."
    )


def resolve_image_path(path_value: str, image_root: Optional[str] = None) -> str:
    path_value = str(path_value)

    if os.path.isabs(path_value):
        return path_value

    if image_root is not None and image_root != "":
        return os.path.join(image_root, path_value)

    return path_value


def get_train_transform(image_size: int = 224):
    # Theo hướng paper PIKA: resize 224, random rotation 10 độ, horizontal flip.
    # Không dùng ColorJitter để bám paper sát hơn.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(image_size: int = 224):
    # Val không augmentation.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class PillBaselineDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_to_idx: Dict[int, int],
        transform,
        image_col: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path).copy()
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.image_root = image_root

        if "pill_label" not in self.df.columns:
            raise RuntimeError(
                f"{csv_path} thiếu cột 'pill_label'. "
                f"Các cột hiện có: {self.df.columns.tolist()}"
            )

        self.image_col = find_image_column(self.df, image_col)

        self.df["pill_label"] = self.df["pill_label"].astype(int)
        self.df["image_path_resolved"] = self.df[self.image_col].astype(str).apply(
            lambda x: resolve_image_path(x, self.image_root)
        )

        before = len(self.df)
        self.df = self.df[self.df["image_path_resolved"].apply(os.path.exists)].copy()
        after = len(self.df)

        if after < before:
            print(f"[Warning] {csv_path}: bỏ {before - after} dòng vì image path không tồn tại.")

        self.df["label_idx"] = self.df["pill_label"].map(self.label_to_idx)

        missing_label = self.df["label_idx"].isna().sum()
        if missing_label > 0:
            print(f"[Warning] {csv_path}: bỏ {missing_label} dòng vì label không có trong label_to_idx.")
            self.df = self.df[self.df["label_idx"].notna()].copy()

        self.df["label_idx"] = self.df["label_idx"].astype(int)
        self.df.reset_index(drop=True, inplace=True)

        if len(self.df) == 0:
            raise RuntimeError(f"Dataset rỗng sau khi lọc: {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path_resolved"]
        label = int(row["label_idx"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label


def build_label_mapping(train_csv: str, val_csv: str, test_csv: Optional[str] = None):
    dfs = [pd.read_csv(train_csv), pd.read_csv(val_csv)]

    if test_csv is not None and test_csv != "" and os.path.exists(test_csv):
        dfs.append(pd.read_csv(test_csv))

    full_df = pd.concat(dfs, ignore_index=True)

    if "pill_label" not in full_df.columns:
        raise RuntimeError("Metadata thiếu cột pill_label.")

    labels = sorted(full_df["pill_label"].astype(int).unique().tolist())

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    return label_to_idx, idx_to_label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    precision, recall, _, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return avg_loss, acc, precision, recall, macro_f1


def main(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    if args.test_csv:
        print("Test CSV :", args.test_csv)

    label_to_idx, idx_to_label = build_label_mapping(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )

    num_classes = len(label_to_idx)
    print("Num classes:", num_classes)

    with open(os.path.join(args.output_dir, "label_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_to_idx.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "idx_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=2)

    train_dataset = PillBaselineDataset(
        csv_path=args.train_csv,
        label_to_idx=label_to_idx,
        transform=get_train_transform(args.image_size),
        image_col=args.image_col,
        image_root=args.image_root,
    )

    val_dataset = PillBaselineDataset(
        csv_path=args.val_csv,
        label_to_idx=label_to_idx,
        transform=get_eval_transform(args.image_size),
        image_col=args.image_col,
        image_root=args.image_root,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples  :", len(val_dataset))
    print("Image column :", train_dataset.image_col)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = timm.create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=num_classes,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_f1 = -1.0
    history = []

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_precision": val_precision,
            "val_macro_recall": val_recall,
            "val_macro_f1": val_f1,
        }
        history.append(row)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   Macro F1: {val_f1:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "model_name": args.model_name,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
            },
            last_path,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "model_name": args.model_name,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                },
                best_path,
            )

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1: {best_val_f1:.4f}")

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.output_dir, "train_history.csv"), index=False)

    print("\nTraining done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_val_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M1 baseline using predefined train/val split CSV")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")

    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/model/M1_split_baseline")
    parser.add_argument("--best_name", type=str, default="M1_baseline_split_best.pth")
    parser.add_argument("--last_name", type=str, default="M1_baseline_split_last.pth")

    parser.add_argument("--image_col", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")

    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

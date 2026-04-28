import os
import json
import random
import argparse
from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def add_mapped_columns(df: pd.DataFrame, label_to_idx: Dict[int, int]) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "pill_crop_path",
        "prescription_image_path",
        "pill_label",
        "context_labels",
    ]

    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"Metadata thiếu cột '{c}'. Các cột hiện có: {df.columns.tolist()}"
            )

    df["pill_label"] = df["pill_label"].astype(int)
    df["mapped_label"] = df["pill_label"].map(label_to_idx)

    missing = df["mapped_label"].isna().sum()
    if missing > 0:
        print(f"[Warning] Bỏ {missing} dòng vì pill_label không có trong label_to_idx.")
        df = df[df["mapped_label"].notna()].copy()

    df["mapped_label"] = df["mapped_label"].astype(int)

    def remap_context(s):
        try:
            vals = json.loads(s)
        except Exception:
            vals = []

        mapped = []
        for v in vals:
            v = int(v)
            if v in label_to_idx:
                mapped.append(int(label_to_idx[v]))

        mapped = sorted(list(set(mapped)))
        return json.dumps(mapped)

    df["context_labels_mapped"] = df["context_labels"].astype(str).apply(remap_context)

    return df.reset_index(drop=True)


def check_image_paths(df: pd.DataFrame, name: str) -> pd.DataFrame:
    pill_exists = df["pill_crop_path"].astype(str).apply(os.path.exists)
    pres_exists = df["prescription_image_path"].astype(str).apply(os.path.exists)

    print(f"{name} pill images existing:", pill_exists.sum(), "/", len(df))
    print(f"{name} prescription images existing:", pres_exists.sum(), "/", len(df))

    missing = (~pill_exists) | (~pres_exists)

    if missing.sum() > 0:
        print(f"[Warning] {name}: bỏ {missing.sum()} dòng vì thiếu ảnh.")
        df = df[~missing].copy()

    if len(df) == 0:
        raise RuntimeError(f"{name} dataset rỗng sau khi lọc ảnh.")

    return df.reset_index(drop=True)


def build_transforms(image_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, val_tfms


class PIKAV3Dataset(Dataset):
    def __init__(self, df, num_classes, pill_transform=None, pres_transform=None):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        self.pill_transform = pill_transform
        self.pres_transform = pres_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pill_img = Image.open(row["pill_crop_path"]).convert("RGB")
        pres_img = Image.open(row["prescription_image_path"]).convert("RGB")

        if self.pill_transform:
            pill_img = self.pill_transform(pill_img)

        if self.pres_transform:
            pres_img = self.pres_transform(pres_img)

        label = int(row["mapped_label"])

        context_labels = json.loads(row["context_labels_mapped"])
        context_vector = np.zeros(self.num_classes, dtype=np.float32)

        for mapped_label in context_labels:
            mapped_label = int(mapped_label)
            if 0 <= mapped_label < self.num_classes:
                context_vector[mapped_label] = 1.0

        context_vector = torch.tensor(context_vector, dtype=torch.float32)

        return pill_img, pres_img, context_vector, label


class TripleContextPIKA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pill_model_name: str,
        pres_model_name: str,
        hidden_dim: int = 256,
        context_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()

        self.pill_encoder = timm.create_model(
            pill_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.pres_encoder = timm.create_model(
            pres_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        pill_dim = self.pill_encoder.num_features
        pres_dim = self.pres_encoder.num_features

        self.context_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
        )

        self.pill_proj = nn.Linear(pill_dim, hidden_dim)
        self.pres_proj = nn.Linear(pres_dim, hidden_dim)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, pill_img, pres_img, context_vector):
        pill_feat = self.pill_proj(self.pill_encoder(pill_img))
        pres_feat = self.pres_proj(self.pres_encoder(pres_img))
        ctx_feat = self.ctx_proj(self.context_encoder(context_vector))

        combined = torch.cat([pill_feat, pres_feat, ctx_feat], dim=1)
        gates = self.gate(combined)

        pill_feat = pill_feat * gates[:, 0].unsqueeze(1)
        pres_feat = pres_feat * gates[:, 1].unsqueeze(1)
        ctx_feat = ctx_feat * gates[:, 2].unsqueeze(1)

        fused = torch.cat([pill_feat, pres_feat, ctx_feat], dim=1)
        fused = self.fusion(fused)

        out = self.classifier(fused)
        return out


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for pill_imgs, pres_imgs, context_vectors, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_vectors = context_vectors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(pill_imgs, pres_imgs, context_vectors)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * pill_imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    for pill_imgs, pres_imgs, context_vectors, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_vectors = context_vectors.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(pill_imgs, pres_imgs, context_vectors)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * pill_imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = accuracy_score(all_labels, all_preds)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_acc, precision, recall, macro_f1


def build_class_weights(train_df: pd.DataFrame, num_classes: int, device: str):
    train_label_counts = Counter(train_df["mapped_label"].tolist())

    class_weights = []
    for class_id in range(num_classes):
        count = train_label_counts.get(class_id, 1)
        class_weights.append(1.0 / count)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights.to(device)


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    print("Using device:", device)

    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    print("Test CSV :", args.test_csv)

    label_to_idx, idx_to_label = build_label_mapping(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )

    num_classes = len(label_to_idx)
    print("Num classes:", num_classes)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_df = add_mapped_columns(train_df, label_to_idx)
    val_df = add_mapped_columns(val_df, label_to_idx)

    train_df = check_image_paths(train_df, "Train")
    val_df = check_image_paths(val_df, "Val")

    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))

    with open(os.path.join(args.output_dir, "label_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_to_idx.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "idx_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=2)

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = PIKAV3Dataset(
        train_df,
        num_classes=num_classes,
        pill_transform=train_tfms,
        pres_transform=train_tfms,
    )

    val_dataset = PIKAV3Dataset(
        val_df,
        num_classes=num_classes,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

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

    model = TripleContextPIKA(
        num_classes=num_classes,
        pill_model_name=args.pill_model_name,
        pres_model_name=args.pres_model_name,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = build_class_weights(train_df, num_classes, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f} | Val Macro F1  : {val_f1:.4f}")

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
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(args.output_dir, "train_history.csv"),
            index=False,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "pill_model_name": args.pill_model_name,
            "pres_model_name": args.pres_model_name,
            "hidden_dim": args.hidden_dim,
            "context_dim": args.context_dim,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "TripleContextPIKA",
        }

        torch.save(checkpoint, last_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(checkpoint, best_path)

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1: {best_f1:.4f}")

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nTraining done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M4 PIKA v3 triple-context model using predefined train/val split CSV")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")

    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/model/M4_split_pika_v3")
    parser.add_argument("--best_name", type=str, default="M4_pika_v3_split_best.pth")
    parser.add_argument("--last_name", type=str, default="M4_pika_v3_split_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--context_dim", type=int, default=128)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")

    args = parser.parse_args()
    main(args)

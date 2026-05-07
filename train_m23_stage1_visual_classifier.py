import os
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    add_mapped_columns,
    build_transforms,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label_to_idx(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def normalize_idx_to_label(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def check_pill_paths(df, split_name):
    df = df.copy()

    if "pill_crop_path" not in df.columns:
        raise ValueError(f"{split_name} CSV must contain pill_crop_path.")

    exists = df["pill_crop_path"].apply(lambda p: Path(str(p)).exists())

    print(f"{split_name} pill images existing: {int(exists.sum())} / {len(df)}")

    if int((~exists).sum()) > 0:
        print(f"[Warning] {split_name}: bỏ {int((~exists).sum())} dòng vì thiếu pill crop.")
        print(df.loc[~exists, ["pill_crop_path", "pill_label"]].head(10).to_string(index=False))

    df = df[exists].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"{split_name} dataset rỗng sau khi lọc pill image.")

    return df


class PillCropDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["pill_crop_path"]
        label = int(row["mapped_label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout_ps=(0.25, 0.35, 0.45, 0.55)):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_ps])
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.feature(x)

        logits = 0.0
        for d in self.dropouts:
            logits = logits + self.fc(d(h))

        logits = logits / len(self.dropouts)
        return logits


class M23VisualClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name="convnext_tiny.in12k_ft_in1k",
        hidden_dim=512,
        dropout_p=0.4,
        pretrained=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = int(self.encoder.num_features)

        self.neck = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        self.classifier = MultiSampleDropoutHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_ps=(0.25, 0.35, 0.45, 0.55),
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.neck(feat)
        logits = self.classifier(z)

        return {
            "logits": logits,
            "embedding": z,
        }


class FocalCELoss(nn.Module):
    def __init__(
        self,
        class_weights=None,
        gamma=1.5,
        label_smoothing=0.02,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, labels):
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pt = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)

        focal = (1.0 - pt) ** self.gamma
        loss = focal * ce

        return loss.mean()


def build_class_weights(train_df, num_classes, device, exponent=0.35):
    counts = Counter(train_df["mapped_label"].tolist())

    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(1.0 / (float(c) ** exponent))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    print("Class weight exponent:", exponent)
    print("Class weight min:", float(weights.min()))
    print("Class weight max:", float(weights.max()))

    return weights.to(device)


def build_weighted_sampler(train_df, exponent=0.5, max_weight_ratio=20.0):
    labels = train_df["mapped_label"].tolist()
    counts = Counter(labels)

    sample_weights = []
    for y in labels:
        sample_weights.append(1.0 / (float(counts[y]) ** exponent))

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    if max_weight_ratio is not None and max_weight_ratio > 0:
        min_w = float(sample_weights.min())
        sample_weights = torch.clamp(sample_weights, max=min_w * max_weight_ratio)

    print("Using WeightedRandomSampler")
    print("Sampler exponent:", exponent)
    print("Max weight ratio:", max_weight_ratio)
    print("Sample weight min:", float(sample_weights.min()))
    print("Sample weight max:", float(sample_weights.max()))

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def freeze_backbone(model, freeze=True):
    for p in model.encoder.parameters():
        p.requires_grad = not freeze

    print("Visual backbone frozen:", freeze)


def count_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, clip_grad_norm=1.0):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="M23 Stage1 Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if clip_grad_norm and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    n = max(1, len(loader.dataset))

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return running_loss / n, acc, macro_f1


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_conf = []

    for images, labels in tqdm(loader, desc="M23 Stage1 Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

        conf, preds = probs.max(dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())

    n = max(1, len(loader.dataset))

    acc = accuracy_score(all_labels, all_preds)

    p_present, r_present, f1_present, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    labels_all = list(range(num_classes))

    p_all, r_all, f1_all, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=labels_all,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": running_loss / n,
        "accuracy": float(acc),
        "macro_precision_present": float(p_present),
        "macro_recall_present": float(r_present),
        "macro_f1_present": float(f1_present),
        "macro_precision_all": float(p_all),
        "macro_recall_all": float(r_all),
        "macro_f1_all": float(f1_all),
        "weighted_f1": float(weighted_f1),
        "labels": all_labels,
        "preds": all_preds,
        "confidence": all_conf,
    }


def save_per_class_metrics(y_true, y_pred, num_classes, idx_to_label, output_path):
    labels_all = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )

    df = pd.DataFrame({
        "mapped_label": labels_all,
        "original_label": [idx_to_label.get(i, i) for i in labels_all],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    df.to_csv(output_path, index=False)


def prepare_dataframe(csv_path, label_to_idx, split_name):
    df = pd.read_csv(csv_path)
    df = add_mapped_columns(df, label_to_idx)
    df = check_pill_paths(df, split_name)

    return df


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== M23 STAGE 1 VISUAL PRETRAINING ===")
    print("Using device:", device)
    print("Train CSV:", args.train_csv)
    print("Val CSV:", args.val_csv)
    print("Test CSV:", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    graph_dir = Path(args.graph_artifacts_dir)

    label_to_idx = normalize_label_to_idx(load_json(graph_dir / "label_to_idx.json"))
    idx_to_label = normalize_idx_to_label(load_json(graph_dir / "idx_to_label.json"))

    num_classes = len(label_to_idx)

    print("Num classes:", num_classes)

    train_df = prepare_dataframe(args.train_csv, label_to_idx, "Train")
    val_df = prepare_dataframe(args.val_csv, label_to_idx, "Val")

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Train labels:", train_df["mapped_label"].nunique())
    print("Val labels:", val_df["mapped_label"].nunique())

    train_labels = set(train_df["mapped_label"].unique())
    val_labels = set(val_df["mapped_label"].unique())

    print("Labels in train but missing in val:", len(train_labels - val_labels))
    print(sorted(list(train_labels - val_labels)))

    save_json({str(k): int(v) for k, v in label_to_idx.items()}, Path(args.output_dir) / "label_to_idx.json")
    save_json({str(k): int(v) for k, v in idx_to_label.items()}, Path(args.output_dir) / "idx_to_label.json")

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = PillCropDataset(train_df, transform=train_tfms)
    val_dataset = PillCropDataset(val_df, transform=val_tfms)

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(
            train_df,
            exponent=args.sampler_exponent,
            max_weight_ratio=args.max_sample_weight_ratio,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    model = M23VisualClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        pretrained=True,
    ).to(device)

    print("Created M23VisualClassifier")
    print("Backbone:", args.backbone_name)
    print("Hidden dim:", args.hidden_dim)
    print("Dropout:", args.dropout_p)

    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model, freeze=True)

    total_params, trainable_params = count_params(model)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    class_weights = build_class_weights(
        train_df,
        num_classes=num_classes,
        device=device,
        exponent=args.class_weight_exponent,
    )

    criterion = FocalCELoss(
        class_weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_path = Path(args.output_dir) / args.best_name
    last_path = Path(args.output_dir) / args.last_name

    config = {
        "stage": "M23_stage1_visual_pretraining",
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "graph_artifacts_dir": args.graph_artifacts_dir,
        "num_classes": num_classes,
        "backbone_name": args.backbone_name,
        "hidden_dim": args.hidden_dim,
        "dropout_p": args.dropout_p,
        "image_size": args.image_size,
        "class_weight_exponent": args.class_weight_exponent,
        "sampler_exponent": args.sampler_exponent,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    save_json(config, Path(args.output_dir) / "m23_stage1_config.json")

    for epoch in range(1, args.epochs + 1):
        print(f"\nM23 Stage1 Visual Epoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                freeze_backbone(model, freeze=True)
            elif epoch == args.freeze_backbone_epochs + 1:
                freeze_backbone(model, freeze=False)

                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - epoch + 1),
                )

                print("Rebuilt optimizer after unfreezing.")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_result = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss  : {val_result['loss']:.4f} | "
            f"Val Acc  : {val_result['accuracy']:.4f} | "
            f"Val Macro Precision: {val_result['macro_precision_present']:.4f} | "
            f"Val Macro Recall: {val_result['macro_recall_present']:.4f} | "
            f"Val Macro F1 Present: {val_result['macro_f1_present']:.4f} | "
            f"Val Macro F1 All: {val_result['macro_f1_all']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_result["loss"],
            "val_acc": val_result["accuracy"],
            "val_macro_precision_present": val_result["macro_precision_present"],
            "val_macro_recall_present": val_result["macro_recall_present"],
            "val_macro_f1_present": val_result["macro_f1_present"],
            "val_macro_f1_all": val_result["macro_f1_all"],
            "val_weighted_f1": val_result["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            Path(args.output_dir) / "train_history.csv",
            index=False,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "neck_state_dict": model.neck.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "num_classes": num_classes,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "backbone_name": args.backbone_name,
            "hidden_dim": args.hidden_dim,
            "dropout_p": args.dropout_p,
            "image_size": args.image_size,
            "val_macro_f1_present": val_result["macro_f1_present"],
            "val_macro_f1_all": val_result["macro_f1_all"],
            "val_acc": val_result["accuracy"],
            "model_type": "M23VisualClassifier",
            "stage": "M23_stage1_visual_pretraining",
            "config": config,
        }

        torch.save(checkpoint, last_path)

        improvement = val_result["macro_f1_present"] - best_f1

        if improvement > args.min_delta:
            best_f1 = val_result["macro_f1_present"]
            bad_epochs = 0

            torch.save(checkpoint, best_path)

            save_per_class_metrics(
                y_true=val_result["labels"],
                y_pred=val_result["preds"],
                num_classes=num_classes,
                idx_to_label=idx_to_label,
                output_path=Path(args.output_dir) / "best_val_per_class_metrics.csv",
            )

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1 Present: {best_f1:.4f}")
        else:
            bad_epochs += 1
            print(f"No significant improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nM23 Stage1 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1 Present:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M23 Stage 1: visual-only pill crop pretraining."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M23_stage1_visual_pretraining/train_run",
    )

    parser.add_argument("--best_name", type=str, default="M23_stage1_visual_best.pth")
    parser.add_argument("--last_name", type=str, default="M23_stage1_visual_last.pth")

    parser.add_argument("--backbone_name", type=str, default="convnext_tiny.in12k_ft_in1k")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.45)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_exponent", type=float, default=0.5)
    parser.add_argument("--max_sample_weight_ratio", type=float, default=20.0)

    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

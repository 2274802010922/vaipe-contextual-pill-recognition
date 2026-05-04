import os
import json
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    BestPIKAModel,
    BestPIKADataset,
    build_label_mapping,
    add_mapped_columns,
    check_image_paths,
    build_graph_matrix,
    build_transforms,
)


def build_class_weights(train_df, num_classes, device, exponent=0.35):
    counts = Counter(train_df["mapped_label"].tolist())

    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = 1.0 / (float(c) ** exponent)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def build_weighted_sampler(train_df, exponent=0.35, max_weight_ratio=15.0):
    labels = train_df["mapped_label"].tolist()
    counts = Counter(labels)

    sample_weights = []
    for y in labels:
        w = 1.0 / (float(counts[y]) ** exponent)
        sample_weights.append(w)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    if max_weight_ratio is not None and max_weight_ratio > 0:
        min_w = float(sample_weights.min())
        max_allowed = min_w * max_weight_ratio
        sample_weights = torch.clamp(sample_weights, max=max_allowed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("Using WeightedRandomSampler")
    print("Sampler exponent:", exponent)
    print("Max weight ratio:", max_weight_ratio)
    print("Sample weight min:", float(sample_weights.min()))
    print("Sample weight max:", float(sample_weights.max()))

    return sampler


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.2, label_smoothing=0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        return focal.mean()


def set_dropout_p(model, dropout_p):
    changed = 0

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            changed += 1

    print(f"Updated Dropout modules: {changed} | dropout_p={dropout_p}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="M16 Training", leave=False)

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(str(device).startswith("cuda"))):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if args_clip_grad_norm := getattr(train_one_epoch, "clip_grad_norm", None):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args_clip_grad_norm)

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

    pbar = tqdm(loader, desc="M16 Validation", leave=False)

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(str(device).startswith("cuda"))):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("Using device:", device)
    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    print("Test CSV :", args.test_csv)
    print("Data root:", args.data_root)
    print("Output dir:", args.output_dir)

    label_to_idx, idx_to_label = build_label_mapping(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )

    label_to_idx = {int(k): int(v) for k, v in label_to_idx.items()}
    idx_to_label = {int(k): int(v) for k, v in idx_to_label.items()}

    num_classes = len(label_to_idx)

    print("Num classes:", num_classes)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_df = add_mapped_columns(train_df, label_to_idx)
    val_df = add_mapped_columns(val_df, label_to_idx)

    train_df = check_image_paths(train_df, "Train")
    val_df = check_image_paths(val_df, "Val")

    train_labels = set(train_df["mapped_label"].unique())
    val_labels = set(val_df["mapped_label"].unique())

    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))
    print("Train labels:", len(train_labels))
    print("Val labels  :", len(val_labels))
    print("Labels in train but missing in val:", len(train_labels - val_labels))
    print(sorted(list(train_labels - val_labels)))

    graph_labels_json = args.graph_labels_json
    graph_pmi_npy = args.graph_pmi_npy

    if graph_labels_json == "":
        graph_labels_json = os.path.join(args.data_root, "graph_labels.json")

    if graph_pmi_npy == "":
        graph_pmi_npy = os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

    sub_pmi = build_graph_matrix(
        graph_labels_json=graph_labels_json,
        graph_pmi_npy=graph_pmi_npy,
        idx_to_label=idx_to_label,
        device=device,
    )

    print("Graph PMI shape:", tuple(sub_pmi.shape))
    print("Max context length:", args.max_context_len)

    save_json(
        {str(k): int(v) for k, v in label_to_idx.items()},
        os.path.join(args.output_dir, "label_to_idx.json"),
    )

    save_json(
        {str(k): int(v) for k, v in idx_to_label.items()},
        os.path.join(args.output_dir, "idx_to_label.json"),
    )

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = BestPIKADataset(
        train_df,
        max_context_len=args.max_context_len,
        pill_transform=train_tfms,
        pres_transform=train_tfms,
    )

    val_dataset = BestPIKADataset(
        val_df,
        max_context_len=args.max_context_len,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(
            train_df=train_df,
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

    model = BestPIKAModel(
        num_classes=num_classes,
        adj_matrix=sub_pmi,
        pill_model_name=args.pill_model_name,
        pres_model_name=args.pres_model_name,
        hidden_dim=args.hidden_dim,
        pretrained=True,
    ).to(device)

    print("Created BestPIKAModel from pretrained backbones.")
    print("Pill model        :", args.pill_model_name)
    print("Prescription model:", args.pres_model_name)
    print("Hidden dim        :", args.hidden_dim)

    if args.dropout_p >= 0:
        set_dropout_p(model, args.dropout_p)

    if args.freeze_backbone_epochs > 0:
        print(f"Will freeze backbones for first {args.freeze_backbone_epochs} epoch(s).")

    class_weights = build_class_weights(
        train_df=train_df,
        num_classes=num_classes,
        device=device,
        exponent=args.class_weight_exponent,
    )

    if args.loss_type == "focal":
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        print("Using FocalLoss")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
        print("Using CrossEntropyLoss")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(str(device).startswith("cuda")))

    train_one_epoch.clip_grad_norm = args.clip_grad_norm if args.clip_grad_norm > 0 else None

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    for epoch in range(1, args.epochs + 1):
        print(f"\nM16 Clean Best PIKA Epoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        if args.freeze_backbone_epochs > 0:
            freeze_now = epoch <= args.freeze_backbone_epochs

            for p in model.pill_encoder.parameters():
                p.requires_grad = not freeze_now

            for p in model.pres_encoder.parameters():
                p.requires_grad = not freeze_now

            print("Backbone frozen:", freeze_now)

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

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss  : {val_loss:.4f} | "
            f"Val Acc  : {val_acc:.4f} | "
            f"Val Macro Precision: {val_precision:.4f} | "
            f"Val Macro Recall: {val_recall:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
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
            "lr": optimizer.param_groups[0]["lr"],
            "loss_type": args.loss_type,
            "class_weight_exponent": args.class_weight_exponent,
            "sampler_exponent": args.sampler_exponent,
            "dropout_p": args.dropout_p,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
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
            "max_context_len": args.max_context_len,
            "graph_labels_json": graph_labels_json,
            "graph_pmi_npy": graph_pmi_npy,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "BestPIKAModel",
            "stage": "M16_clean_best_pika",
            "clean_split": True,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "class_weight_exponent": args.class_weight_exponent,
            "sampler_exponent": args.sampler_exponent,
            "dropout_p": args.dropout_p,
            "label_smoothing": args.label_smoothing,
        }

        torch.save(checkpoint, last_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1: {best_f1:.4f}")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nM16 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train M16 Clean Best PIKA model from scratch on clean paper-like split."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M16_clean_best_pika",
    )

    parser.add_argument(
        "--best_name",
        type=str,
        default="M16_clean_best_pika_best.pth",
    )

    parser.add_argument(
        "--last_name",
        type=str,
        default="M16_clean_best_pika_last.pth",
    )

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=1.2)

    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_exponent", type=float, default=0.35)
    parser.add_argument("--max_sample_weight_ratio", type=float, default=15.0)

    parser.add_argument("--dropout_p", type=float, default=0.45)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)

    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

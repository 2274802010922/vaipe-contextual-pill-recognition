import os
import json
import argparse
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

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


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_label_to_idx(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def normalize_idx_to_label(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def build_class_weights(train_df: pd.DataFrame, num_classes: int, device: str):
    train_label_counts = Counter(train_df["mapped_label"].tolist())

    class_weights = []
    for class_id in range(num_classes):
        count = train_label_counts.get(class_id, 1)
        class_weights.append(1.0 / count)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights.to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Fine-tuning", leave=False)

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
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

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
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


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    print("Using device:", device)

    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    print("Test CSV :", args.test_csv)
    print("Base checkpoint:", args.base_checkpoint)
    print("Data root:", args.data_root)

    base_ckpt = safe_torch_load(args.base_checkpoint, device)

    if "model_state_dict" not in base_ckpt:
        raise RuntimeError("Base checkpoint thiếu model_state_dict.")

    # Ưu tiên dùng label mapping từ checkpoint M6 để đảm bảo mapping giống M6.
    if "label_to_idx" in base_ckpt and "idx_to_label" in base_ckpt:
        label_to_idx = normalize_label_to_idx(base_ckpt["label_to_idx"])
        idx_to_label = normalize_idx_to_label(base_ckpt["idx_to_label"])
    else:
        label_to_idx, idx_to_label = build_label_mapping(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
        )

    num_classes = int(base_ckpt.get("num_classes", len(label_to_idx)))
    pill_model_name = base_ckpt.get("pill_model_name", args.pill_model_name)
    pres_model_name = base_ckpt.get("pres_model_name", args.pres_model_name)
    hidden_dim = int(base_ckpt.get("hidden_dim", args.hidden_dim))
    max_context_len_from_ckpt = int(base_ckpt.get("max_context_len", args.max_context_len))

    print("Pill model      :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Hidden dim      :", hidden_dim)
    print("Num classes     :", num_classes)

    graph_labels_json = args.graph_labels_json
    graph_pmi_npy = args.graph_pmi_npy

    if graph_labels_json == "":
        graph_labels_json = os.path.join(args.data_root, "graph_labels.json")

    if graph_pmi_npy == "":
        graph_pmi_npy = os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_df = add_mapped_columns(train_df, label_to_idx)
    val_df = add_mapped_columns(val_df, label_to_idx)

    train_df = check_image_paths(train_df, "Train")
    val_df = check_image_paths(val_df, "Val")

    sub_pmi = build_graph_matrix(
        graph_labels_json=graph_labels_json,
        graph_pmi_npy=graph_pmi_npy,
        idx_to_label=idx_to_label,
        device=device,
    )

    max_context_len = max_context_len_from_ckpt
    print("Max context length used:", max_context_len)
    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))

    with open(os.path.join(args.output_dir, "label_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_to_idx.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "idx_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=2)

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = BestPIKADataset(
        train_df,
        max_context_len=max_context_len,
        pill_transform=train_tfms,
        pres_transform=train_tfms,
    )

    val_dataset = BestPIKADataset(
        val_df,
        max_context_len=max_context_len,
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

    model = BestPIKAModel(
        num_classes=num_classes,
        adj_matrix=sub_pmi,
        pill_model_name=pill_model_name,
        pres_model_name=pres_model_name,
        hidden_dim=hidden_dim,
        pretrained=False,
    ).to(device)

    model.load_state_dict(base_ckpt["model_state_dict"], strict=True)
    print("Loaded base checkpoint successfully.")

    if args.freeze_backbone:
        print("Freezing pill_encoder and pres_encoder.")
        for p in model.pill_encoder.parameters():
            p.requires_grad = False
        for p in model.pres_encoder.parameters():
            p.requires_grad = False

    class_weights = build_class_weights(train_df, num_classes, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
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

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1 = float(base_ckpt.get("val_macro_f1", -1.0))
    print("Starting best_f1 from base checkpoint:", best_f1)

    bad_epochs = 0
    history = []

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    # Lưu bản base làm best ban đầu, để nếu fine-tune không cải thiện thì vẫn giữ được checkpoint tốt nhất.
    initial_checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "pill_model_name": pill_model_name,
        "pres_model_name": pres_model_name,
        "hidden_dim": hidden_dim,
        "max_context_len": max_context_len,
        "graph_labels_json": graph_labels_json,
        "graph_pmi_npy": graph_pmi_npy,
        "val_macro_f1": best_f1,
        "val_acc": base_ckpt.get("val_acc", None),
        "model_type": "BestPIKAModel",
        "fine_tune_from": args.base_checkpoint,
    }
    torch.save(initial_checkpoint, best_path)

    for epoch in range(1, args.epochs + 1):
        print(f"\nFine-tune Epoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

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
            "pill_model_name": pill_model_name,
            "pres_model_name": pres_model_name,
            "hidden_dim": hidden_dim,
            "max_context_len": max_context_len,
            "graph_labels_json": graph_labels_json,
            "graph_pmi_npy": graph_pmi_npy,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "BestPIKAModel",
            "fine_tune_from": args.base_checkpoint,
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

    print("\nFine-tuning done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M7 fine-tune BestPIKAModel from M6 checkpoint")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")

    parser.add_argument("--base_checkpoint", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/model/M7_split_finetune_v1")
    parser.add_argument("--best_name", type=str, default="M7_finetune_v1_split_best.pth")
    parser.add_argument("--last_name", type=str, default="M7_finetune_v1_split_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_context_len", type=int, default=5)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

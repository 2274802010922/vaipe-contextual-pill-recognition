import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    add_mapped_columns,
    build_transforms,
)

from train_m23_stage1_visual_classifier import (
    load_json,
    save_json,
    normalize_label_to_idx,
    normalize_idx_to_label,
    check_pill_paths,
    PillCropDataset,
    M23VisualClassifier,
    freeze_backbone,
    count_params,
    save_per_class_metrics,
)


def build_class_weights(train_df, num_classes, device, exponent=0.20):
    """
    M23 v2 dùng class weight nhẹ hơn v1.
    v1 dùng sampler + focal + class weight nên có thể ép imbalance quá mạnh.
    """
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


def prepare_dataframe(csv_path, label_to_idx, split_name):
    df = pd.read_csv(csv_path)
    df = add_mapped_columns(df, label_to_idx)
    df = check_pill_paths(df, split_name)
    return df


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, clip_grad_norm=1.0):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="M23 v2 Training", leave=False):
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

    for images, labels in tqdm(loader, desc="M23 v2 Validation", leave=False):
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


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== M23 STAGE 1 VISUAL PRETRAINING V2 ===")
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

    model = M23VisualClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        pretrained=True,
    ).to(device)

    print("Created M23VisualClassifier v2")
    print("Backbone:", args.backbone_name)
    print("Hidden dim:", args.hidden_dim)
    print("Dropout:", args.dropout_p)
    print("Loss: CrossEntropyLoss")
    print("Weighted sampler: False")

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

    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_path = Path(args.output_dir) / args.best_name
    last_path = Path(args.output_dir) / args.last_name

    config = {
        "stage": "M23_stage1_visual_pretraining_v2",
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
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "loss": "CrossEntropyLoss",
        "use_weighted_sampler": False,
    }

    save_json(config, Path(args.output_dir) / "m23_stage1_v2_config.json")

    for epoch in range(1, args.epochs + 1):
        print(f"\nM23 Stage1 v2 Visual Epoch [{epoch}/{args.epochs}]")
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
            "stage": "M23_stage1_visual_pretraining_v2",
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

    print("\nM23 Stage1 v2 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1 Present:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M23 Stage 1 v2: stable visual-only pill crop pretraining."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M23_stage1_visual_pretraining/train_run_v2_ce",
    )

    parser.add_argument("--best_name", type=str, default="M23_stage1_v2_visual_best.pth")
    parser.add_argument("--last_name", type=str, default="M23_stage1_v2_visual_last.pth")

    parser.add_argument("--backbone_name", type=str, default="convnext_tiny.in12k_ft_in1k")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.55)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.20)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

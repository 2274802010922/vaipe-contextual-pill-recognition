import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    BestPIKADataset,
    build_transforms,
)

from train_m17_faithful_pika import (
    save_json,
    normalize_label_to_idx,
    normalize_idx_to_label,
    build_class_weights,
    build_weighted_sampler,
    M17FaithfulPIKAModel,
    M17PIKALoss,
    train_one_epoch,
    validate_one_epoch,
    prepare_dataframe,
)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def set_dropout_p(model, dropout_p):
    changed = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_p
            changed += 1

    print(f"Updated Dropout modules: {changed} | dropout_p={dropout_p}")


def freeze_visual_backbones(model, freeze=True):
    for p in model.pill_encoder.parameters():
        p.requires_grad = not freeze

    for p in model.pres_encoder.parameters():
        p.requires_grad = not freeze

    print("Visual backbones frozen:", freeze)


def count_trainable_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return total, trainable


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== M18 IMPROVED PIKA TRAINING ===")
    print("Using device:", device)
    print("Base checkpoint:", args.base_checkpoint)
    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    print("Test CSV :", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    ckpt = safe_torch_load(args.base_checkpoint, device)

    if "model_state_dict" not in ckpt:
        raise RuntimeError("Base checkpoint does not contain model_state_dict.")

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt["num_classes"])
    hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
    graph_dim = int(ckpt.get("graph_dim", args.graph_dim))
    max_context_len = int(ckpt.get("max_context_len", args.max_context_len))

    pill_model_name = ckpt.get("pill_model_name", args.pill_model_name)
    pres_model_name = ckpt.get("pres_model_name", args.pres_model_name)

    base_epoch = int(ckpt.get("epoch", -1))
    base_val_f1 = float(ckpt.get("val_macro_f1", -1.0))

    print("\nBase epoch:", base_epoch)
    print("Base Val Macro F1:", base_val_f1)
    print("Num classes:", num_classes)
    print("Graph dim:", graph_dim)
    print("Max context len:", max_context_len)
    print("Pill model:", pill_model_name)
    print("Prescription model:", pres_model_name)

    graph_dir = Path(args.graph_artifacts_dir)
    graph_embeddings_path = graph_dir / "graph_embeddings.npy"

    if not graph_embeddings_path.exists():
        raise FileNotFoundError(f"Missing graph embeddings: {graph_embeddings_path}")

    graph_embeddings = np.load(graph_embeddings_path).astype("float32")

    print("\nGraph embeddings shape:", graph_embeddings.shape)

    if graph_embeddings.shape[0] != num_classes:
        raise ValueError(
            f"Graph embedding rows {graph_embeddings.shape[0]} != num_classes {num_classes}"
        )

    if graph_embeddings.shape[1] != graph_dim:
        raise ValueError(
            f"Graph embedding dim {graph_embeddings.shape[1]} != graph_dim {graph_dim}"
        )

    train_df = prepare_dataframe(args.train_csv, label_to_idx, "Train")
    val_df = prepare_dataframe(args.val_csv, label_to_idx, "Val")

    train_labels = set(train_df["mapped_label"].unique())
    val_labels = set(val_df["mapped_label"].unique())

    print("\nTrain rows:", len(train_df))
    print("Val rows  :", len(val_df))
    print("Train labels:", len(train_labels))
    print("Val labels  :", len(val_labels))
    print("Labels in train but missing in val:", len(train_labels - val_labels))
    print(sorted(list(train_labels - val_labels)))

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

    model = M17FaithfulPIKAModel(
        num_classes=num_classes,
        graph_embeddings=graph_embeddings,
        pill_model_name=pill_model_name,
        pres_model_name=pres_model_name,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
        dropout_p=args.dropout_p,
        pretrained=False,
        train_graph_embeddings=args.train_graph_embeddings,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print("\nLoaded base M17 checkpoint successfully.")
    print("Train graph embeddings:", args.train_graph_embeddings)

    if args.dropout_p >= 0:
        set_dropout_p(model, args.dropout_p)

    if args.freeze_backbone_epochs > 0:
        print(f"Will freeze visual backbones for first {args.freeze_backbone_epochs} epoch(s).")

    total_params, trainable_params = count_trainable_params(model)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    class_weights = build_class_weights(
        train_df=train_df,
        num_classes=num_classes,
        device=device,
        exponent=args.class_weight_exponent,
    )

    criterion = M17PIKALoss(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        pseudo_loss_weight=args.pseudo_loss_weight,
        link_loss_weight=args.link_loss_weight,
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

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    best_f1 = base_val_f1
    bad_epochs = 0

    # Copy base checkpoint as initial best in M18 output folder.
    if not os.path.exists(best_path):
        shutil.copy2(args.base_checkpoint, best_path)
        print("Copied base checkpoint as initial M18 best:", best_path)

    history = []

    if args.previous_history_csv and os.path.exists(args.previous_history_csv):
        old_hist = pd.read_csv(args.previous_history_csv)
        history = old_hist.to_dict("records")
        print("Loaded previous history rows:", len(history))

    config = {
        "stage": "M18_improved_pika",
        "base_checkpoint": args.base_checkpoint,
        "base_epoch": base_epoch,
        "base_val_macro_f1": base_val_f1,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "graph_artifacts_dir": args.graph_artifacts_dir,
        "num_classes": num_classes,
        "graph_dim": graph_dim,
        "pill_model_name": pill_model_name,
        "pres_model_name": pres_model_name,
        "hidden_dim": hidden_dim,
        "max_context_len": max_context_len,
        "train_graph_embeddings": args.train_graph_embeddings,
        "pseudo_loss_weight": args.pseudo_loss_weight,
        "link_loss_weight": args.link_loss_weight,
        "label_smoothing": args.label_smoothing,
        "class_weight_exponent": args.class_weight_exponent,
        "dropout_p": args.dropout_p,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "note": "M18 improves M17 by fine-tuning with lower auxiliary losses and trainable graph embeddings.",
    }

    save_json(config, os.path.join(args.output_dir, "m18_train_config.json"))

    for epoch in range(1, args.epochs + 1):
        print(f"\nM18 Improved PIKA Fine-tune Epoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        if args.freeze_backbone_epochs > 0:
            freeze_now = epoch <= args.freeze_backbone_epochs
            freeze_visual_backbones(model, freeze=freeze_now)

            # Rebuild optimizer when unfreezing starts.
            if epoch == args.freeze_backbone_epochs + 1:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - epoch + 1),
                )
                print("Rebuilt optimizer after unfreezing backbones.")

        train_loss, train_main, train_pseudo, train_link, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            clip_grad_norm=args.clip_grad_norm,
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
            f"Main: {train_main:.4f} | "
            f"Pseudo: {train_pseudo:.4f} | "
            f"Link: {train_link:.4f} | "
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
            "m18_epoch": epoch,
            "base_epoch": base_epoch,
            "train_loss": train_loss,
            "train_main_loss": train_main,
            "train_pseudo_loss": train_pseudo,
            "train_link_loss": train_link,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_precision": val_precision,
            "val_macro_recall": val_recall,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "pseudo_loss_weight": args.pseudo_loss_weight,
            "link_loss_weight": args.link_loss_weight,
            "train_graph_embeddings": args.train_graph_embeddings,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            os.path.join(args.output_dir, "train_history.csv"),
            index=False,
        )

        checkpoint = {
            "epoch": base_epoch + epoch,
            "m18_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "pill_model_name": pill_model_name,
            "pres_model_name": pres_model_name,
            "hidden_dim": hidden_dim,
            "graph_dim": graph_dim,
            "max_context_len": max_context_len,
            "graph_artifacts_dir": args.graph_artifacts_dir,
            "graph_embeddings_path": str(Path(args.graph_artifacts_dir) / "graph_embeddings.npy"),
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "M17FaithfulPIKAModel",
            "stage": "M18_improved_pika",
            "config": config,
        }

        torch.save(checkpoint, last_path)

        improvement = val_f1 - best_f1

        if improvement > args.min_delta:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1: {best_f1:.4f}")
        else:
            bad_epochs += 1
            print(f"No significant improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nM18 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train M18 improved PIKA by fine-tuning M17 faithful baseline."
    )

    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--previous_history_csv", type=str, default="")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M18_improved_pika/train_run",
    )

    parser.add_argument("--best_name", type=str, default="M18_improved_pika_best.pth")
    parser.add_argument("--last_name", type=str, default="M18_improved_pika_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--pseudo_loss_weight", type=float, default=0.15)
    parser.add_argument("--link_loss_weight", type=float, default=0.05)

    parser.add_argument("--dropout_p", type=float, default=0.45)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_exponent", type=float, default=0.35)
    parser.add_argument("--max_sample_weight_ratio", type=float, default=15.0)

    parser.add_argument("--train_graph_embeddings", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)

    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

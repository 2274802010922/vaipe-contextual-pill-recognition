import os
import json
import shutil
import argparse
from pathlib import Path

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
    load_json,
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


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== M17 FAITHFUL PIKA RESUME TRAINING ===")
    print("Using device:", device)
    print("Resume checkpoint:", args.resume_checkpoint)
    print("Train CSV:", args.train_csv)
    print("Val CSV  :", args.val_csv)
    print("Test CSV :", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    ckpt = safe_torch_load(args.resume_checkpoint, device)

    if "model_state_dict" not in ckpt:
        raise RuntimeError("Resume checkpoint does not contain model_state_dict.")

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt["num_classes"])
    hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
    graph_dim = int(ckpt.get("graph_dim", args.graph_dim))
    max_context_len = int(ckpt.get("max_context_len", args.max_context_len))

    pill_model_name = ckpt.get("pill_model_name", args.pill_model_name)
    pres_model_name = ckpt.get("pres_model_name", args.pres_model_name)

    resume_epoch = int(ckpt.get("epoch", 0))
    start_epoch = resume_epoch + 1
    best_f1 = float(ckpt.get("val_macro_f1", -1.0))

    print("\nResume epoch:", resume_epoch)
    print("Start epoch :", start_epoch)
    print("Initial best Val Macro F1:", best_f1)
    print("Num classes:", num_classes)
    print("Graph dim:", graph_dim)
    print("Max context len:", max_context_len)

    graph_dir = Path(args.graph_artifacts_dir)
    graph_embeddings_path = graph_dir / "graph_embeddings.npy"

    if not graph_embeddings_path.exists():
        raise FileNotFoundError(f"Missing graph embeddings: {graph_embeddings_path}")

    import numpy as np
    graph_embeddings = np.load(graph_embeddings_path).astype("float32")

    print("Graph embeddings shape:", graph_embeddings.shape)

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

    print("\nLoaded model weights from resume checkpoint.")
    print("Pill model        :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Hidden dim        :", hidden_dim)
    print("Train graph embeddings:", args.train_graph_embeddings)

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
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    remaining_epochs = max(1, args.total_epochs - resume_epoch)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    # Copy resume checkpoint as initial best/last in the new output folder.
    if not os.path.exists(best_path):
        shutil.copy2(args.resume_checkpoint, best_path)
        print("Copied resume checkpoint as initial best:", best_path)

    if not os.path.exists(last_path):
        shutil.copy2(args.resume_checkpoint, last_path)
        print("Copied resume checkpoint as initial last:", last_path)

    history = []

    if args.previous_history_csv and os.path.exists(args.previous_history_csv):
        old_hist = pd.read_csv(args.previous_history_csv)
        history = old_hist.to_dict("records")
        print("Loaded previous history rows:", len(history))

    config = {
        "stage": "M17_faithful_pika_resume",
        "resume_checkpoint": args.resume_checkpoint,
        "resume_epoch": resume_epoch,
        "start_epoch": start_epoch,
        "total_epochs": args.total_epochs,
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
        "pseudo_loss_weight": args.pseudo_loss_weight,
        "link_loss_weight": args.link_loss_weight,
        "label_smoothing": args.label_smoothing,
        "class_weight_exponent": args.class_weight_exponent,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "note": "This is a safe fine-tune continuation from model weights. Optimizer state is freshly initialized.",
    }

    save_json(config, os.path.join(args.output_dir, "m17_resume_config.json"))

    bad_epochs = 0

    for epoch in range(start_epoch, args.total_epochs + 1):
        print(f"\nM17 Resume Epoch [{epoch}/{args.total_epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

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
            "epoch": epoch,
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
            "resume_from_epoch": resume_epoch,
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
            "graph_dim": graph_dim,
            "max_context_len": max_context_len,
            "graph_artifacts_dir": args.graph_artifacts_dir,
            "graph_embeddings_path": str(graph_embeddings_path),
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "M17FaithfulPIKAModel",
            "stage": "M17_faithful_pika_resume",
            "config": config,
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

    print("\nM17 resume training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume M17 faithful PIKA training safely from an existing checkpoint."
    )

    parser.add_argument("--resume_checkpoint", type=str, required=True)
    parser.add_argument("--previous_history_csv", type=str, default="")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M17_faithful_pika/train_run_resume_ep5",
    )

    parser.add_argument("--best_name", type=str, default="M17_faithful_pika_resume_best.pth")
    parser.add_argument("--last_name", type=str, default="M17_faithful_pika_resume_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--total_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=7.5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--pseudo_loss_weight", type=float, default=0.30)
    parser.add_argument("--link_loss_weight", type=float, default=0.10)

    parser.add_argument("--dropout_p", type=float, default=0.45)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_exponent", type=float, default=0.35)
    parser.add_argument("--max_sample_weight_ratio", type=float, default=15.0)

    parser.add_argument("--train_graph_embeddings", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

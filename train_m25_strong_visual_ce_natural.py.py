import argparse
import copy
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    build_transforms,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


class M25NaturalVisualDataset(Dataset):
    def __init__(self, df, transform=None, hard_weight_max=1.05):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.hard_weight_max = float(hard_weight_max)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["pill_crop_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["mapped_label"])

        hard_weight = float(row.get("hard_class_weight", 1.0))
        hard_weight = max(1.0, min(self.hard_weight_max, hard_weight))

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "hard_weight": torch.tensor(hard_weight, dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class M25StrongVisualCEModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        hidden_dim=512,
        dropout_p=0.45,
        pretrained=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = int(self.encoder.num_features)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images):
        feat = self.encoder(images)
        logits = self.classifier(feat)

        return {
            "logits": logits,
            "features": feat,
        }


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if torch.is_floating_point(ema_v):
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)

    def to(self, device):
        self.ema.to(device)
        return self


def build_class_weights(train_df, num_classes, device, exponent=0.05):
    counts = Counter(train_df["mapped_label"].astype(int).tolist())

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


def natural_ce_loss(logits, labels, class_weights=None, sample_weights=None, label_smoothing=0.03):
    loss = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    if sample_weights is not None:
        loss = loss * sample_weights

    return loss.mean()


def mixup_batch(images, labels, sample_weights, alpha=0.0):
    if alpha is None or alpha <= 0:
        return images, labels, labels, sample_weights, sample_weights, 1.0

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[perm]
    labels_a = labels
    labels_b = labels[perm]

    weights_a = sample_weights
    weights_b = sample_weights[perm]

    return mixed_images, labels_a, labels_b, weights_a, weights_b, float(lam)


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


def load_metadata_csv(path, split_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} metadata CSV: {path}")

    df = pd.read_csv(path)

    required = [
        "pill_crop_path",
        "mapped_label",
        "hard_class_weight",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{split_name} metadata missing columns: {missing}")

    df["mapped_label"] = df["mapped_label"].astype(int)
    df = check_pill_paths(df, split_name)

    return df


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    encoder_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    param_groups = []

    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": backbone_lr,
        })

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    return optimizer


def get_main_lr(optimizer):
    return max(group["lr"] for group in optimizer.param_groups)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    class_weights,
    label_smoothing,
    mixup_alpha,
    clip_grad_norm,
    ema=None,
):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc="M25 Natural CE Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        sample_weights = batch["hard_weight"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            images_m, labels_a, labels_b, weights_a, weights_b, lam = mixup_batch(
                images,
                labels,
                sample_weights,
                alpha=mixup_alpha,
            )

            outputs = model(images_m)
            logits = outputs["logits"]

            if mixup_alpha and mixup_alpha > 0:
                loss_a = natural_ce_loss(
                    logits,
                    labels_a,
                    class_weights=class_weights,
                    sample_weights=weights_a,
                    label_smoothing=label_smoothing,
                )

                loss_b = natural_ce_loss(
                    logits,
                    labels_b,
                    class_weights=class_weights,
                    sample_weights=weights_b,
                    label_smoothing=label_smoothing,
                )

                loss = lam * loss_a + (1.0 - lam) * loss_b
            else:
                loss = natural_ce_loss(
                    logits,
                    labels,
                    class_weights=class_weights,
                    sample_weights=sample_weights,
                    label_smoothing=label_smoothing,
                )

        scaler.scale(loss).backward()

        if clip_grad_norm and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)

        # Train metric dùng original labels, chỉ để theo dõi tương đối.
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    n = max(1, len(all_labels))

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": running_loss / n,
        "acc": float(acc),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def validate_one_epoch(model, loader, device, num_classes, class_weights, label_smoothing):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_conf = []

    for batch in tqdm(loader, desc="M25 Natural CE Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        sample_weights = batch["hard_weight"].to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)
            logits = outputs["logits"]

            loss = natural_ce_loss(
                logits,
                labels,
                class_weights=class_weights,
                sample_weights=sample_weights,
                label_smoothing=label_smoothing,
            )

            probs = torch.softmax(logits, dim=1)

        conf, preds = probs.max(dim=1)

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())

    n = max(1, len(all_labels))

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

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
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
        "weighted_f1": float(f1_w),
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


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== M25 STRONG VISUAL CE NATURAL TRAINING ===")
    print("Using device:", device)
    print("Train metadata:", args.train_metadata)
    print("Val metadata:", args.val_metadata)
    print("Test metadata:", args.test_metadata)
    print("Output dir:", args.output_dir)

    train_df = load_metadata_csv(args.train_metadata, "Train")
    val_df = load_metadata_csv(args.val_metadata, "Val")

    idx_to_label = normalize_idx_to_label(load_json(args.idx_to_label))
    num_classes = len(idx_to_label)

    print("Num classes:", num_classes)
    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Train labels:", train_df["mapped_label"].nunique())
    print("Val labels:", val_df["mapped_label"].nunique())

    train_labels = set(train_df["mapped_label"].unique())
    val_labels = set(val_df["mapped_label"].unique())

    print("Labels in train but missing in val:", len(train_labels - val_labels))
    print(sorted(list(train_labels - val_labels)))

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = M25NaturalVisualDataset(
        train_df,
        transform=train_tfms,
        hard_weight_max=args.hard_weight_max,
    )

    val_dataset = M25NaturalVisualDataset(
        val_df,
        transform=val_tfms,
        hard_weight_max=args.hard_weight_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = M25StrongVisualCEModel(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        pretrained=True,
    ).to(device)

    print("Created M25StrongVisualCEModel")
    print("Backbone:", args.backbone_name)
    print("Hidden dim:", args.hidden_dim)
    print("Dropout:", args.dropout_p)
    print("Natural shuffle: True")
    print("Mixup alpha:", args.mixup_alpha)
    print("EMA enabled:", args.use_ema)
    print("hard_weight_max:", args.hard_weight_max)

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

    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay).to(device)

    best_f1 = -1.0
    best_source = "none"
    bad_epochs = 0
    history = []

    output_dir = Path(args.output_dir)

    best_path = output_dir / args.best_name
    last_path = output_dir / args.last_name

    config = {
        "stage": "M25_strong_visual_ce_natural",
        "train_metadata": args.train_metadata,
        "val_metadata": args.val_metadata,
        "test_metadata": args.test_metadata,
        "idx_to_label": args.idx_to_label,
        "num_classes": num_classes,
        "backbone_name": args.backbone_name,
        "hidden_dim": args.hidden_dim,
        "dropout_p": args.dropout_p,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "class_weight_exponent": args.class_weight_exponent,
        "hard_weight_max": args.hard_weight_max,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "natural_sampling": True,
    }

    save_json(config, output_dir / "m25_strong_visual_ce_natural_config.json")
    save_json({str(k): int(v) for k, v in idx_to_label.items()}, output_dir / "idx_to_label.json")

    for epoch in range(1, args.epochs + 1):
        print(f"\nM25 Strong Visual CE Epoch [{epoch}/{args.epochs}]")
        print(f"Current max LR: {get_main_lr(optimizer):.8f}")

        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                freeze_backbone(model, freeze=True)
            elif epoch == args.freeze_backbone_epochs + 1:
                freeze_backbone(model, freeze=False)

                optimizer = build_optimizer(
                    model,
                    backbone_lr=args.backbone_lr,
                    head_lr=args.head_lr,
                    weight_decay=args.weight_decay,
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - epoch + 1),
                )

                if args.use_ema:
                    ema = ModelEMA(model, decay=args.ema_decay).to(device)

                print("Rebuilt optimizer after unfreezing.")

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            clip_grad_norm=args.clip_grad_norm,
            ema=ema,
        )

        val_result = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )

        ema_val_result = None
        if ema is not None:
            ema_val_result = validate_one_epoch(
                model=ema.ema,
                loader=val_loader,
                device=device,
                num_classes=num_classes,
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
            )

        scheduler.step()

        print(
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Train Acc: {train_result['acc']:.4f} | "
            f"Train Macro F1: {train_result['macro_f1']:.4f}"
        )

        print(
            f"Val Loss  : {val_result['loss']:.4f} | "
            f"Val Acc  : {val_result['accuracy']:.4f} | "
            f"Val Macro Precision: {val_result['macro_precision_present']:.4f} | "
            f"Val Macro Recall: {val_result['macro_recall_present']:.4f} | "
            f"Val Macro F1 Present: {val_result['macro_f1_present']:.4f} | "
            f"Val Macro F1 All: {val_result['macro_f1_all']:.4f}"
        )

        if ema_val_result is not None:
            print(
                f"EMA Val Loss  : {ema_val_result['loss']:.4f} | "
                f"EMA Val Acc  : {ema_val_result['accuracy']:.4f} | "
                f"EMA Val Macro F1 Present: {ema_val_result['macro_f1_present']:.4f} | "
                f"EMA Val Macro F1 All: {ema_val_result['macro_f1_all']:.4f}"
            )

        selected_result = val_result
        selected_source = "model"

        if ema_val_result is not None:
            if ema_val_result["macro_f1_present"] >= val_result["macro_f1_present"]:
                selected_result = ema_val_result
                selected_source = "ema"

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_acc": train_result["acc"],
            "train_macro_f1": train_result["macro_f1"],
            "val_loss": val_result["loss"],
            "val_acc": val_result["accuracy"],
            "val_macro_precision_present": val_result["macro_precision_present"],
            "val_macro_recall_present": val_result["macro_recall_present"],
            "val_macro_f1_present": val_result["macro_f1_present"],
            "val_macro_f1_all": val_result["macro_f1_all"],
            "val_weighted_f1": val_result["weighted_f1"],
            "ema_val_loss": ema_val_result["loss"] if ema_val_result else None,
            "ema_val_acc": ema_val_result["accuracy"] if ema_val_result else None,
            "ema_val_macro_f1_present": ema_val_result["macro_f1_present"] if ema_val_result else None,
            "ema_val_macro_f1_all": ema_val_result["macro_f1_all"] if ema_val_result else None,
            "selected_val_macro_f1_present": selected_result["macro_f1_present"],
            "selected_source": selected_source,
            "lr": get_main_lr(optimizer),
        }

        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)

        checkpoint_model = ema.ema if selected_source == "ema" and ema is not None else model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": checkpoint_model.state_dict(),
            "num_classes": num_classes,
            "idx_to_label": idx_to_label,
            "backbone_name": args.backbone_name,
            "hidden_dim": args.hidden_dim,
            "dropout_p": args.dropout_p,
            "image_size": args.image_size,
            "val_macro_f1_present": selected_result["macro_f1_present"],
            "val_macro_f1_all": selected_result["macro_f1_all"],
            "val_acc": selected_result["accuracy"],
            "selected_source": selected_source,
            "model_type": "M25StrongVisualCEModel",
            "stage": "M25_strong_visual_ce_natural",
            "config": config,
        }

        torch.save(checkpoint, last_path)

        improvement = selected_result["macro_f1_present"] - best_f1

        if improvement > args.min_delta:
            best_f1 = selected_result["macro_f1_present"]
            best_source = selected_source
            bad_epochs = 0

            torch.save(checkpoint, best_path)

            save_per_class_metrics(
                y_true=selected_result["labels"],
                y_pred=selected_result["preds"],
                num_classes=num_classes,
                idx_to_label=idx_to_label,
                output_path=output_dir / "best_val_per_class_metrics.csv",
            )

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1 Present: {best_f1:.4f}")
            print(f"Best source: {best_source}")
        else:
            bad_epochs += 1
            print(f"No significant improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nM25 strong visual CE natural training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1 Present:", best_f1)
    print("Best source:", best_source)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M25 strong visual classifier with natural sampling, CE, light weights, EMA."
    )

    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--val_metadata", type=str, required=True)
    parser.add_argument("--test_metadata", type=str, required=True)
    parser.add_argument("--idx_to_label", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M25_strong_visual_ce_natural/train_run_v1",
    )

    parser.add_argument("--best_name", type=str, default="M25_strong_visual_ce_best.pth")
    parser.add_argument("--last_name", type=str, default="M25_strong_visual_ce_last.pth")

    parser.add_argument("--backbone_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.45)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.05)
    parser.add_argument("--hard_weight_max", type=float, default=1.05)
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    parser.add_argument("--mixup_alpha", type=float, default=0.0)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

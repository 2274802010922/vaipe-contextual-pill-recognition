import argparse
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.utils.data import Dataset, DataLoader, Sampler
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


class M24V2HardMetricDataset(Dataset):
    def __init__(self, df, transform=None, hard_weight_max=1.3):
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

        is_hard = bool(row.get("is_hard_class", False))

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "hard_weight": torch.tensor(hard_weight, dtype=torch.float32),
            "is_hard": torch.tensor(int(is_hard), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class ClassBalancedBatchSampler(Sampler):
    """
    Batch có nhiều class, mỗi class có >=2 samples để SupCon có positive pairs.
    M24 v2 dùng class-balanced nhẹ hơn, không ép quá mạnh.
    """

    def __init__(
        self,
        labels,
        hard_class_weight_by_label=None,
        classes_per_batch=16,
        samples_per_class=2,
        num_batches=None,
        seed=42,
        hard_weight_max=1.3,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.seed = int(seed)
        self.hard_weight_max = float(hard_weight_max)

        self.label_to_indices = defaultdict(list)

        for idx, y in enumerate(self.labels):
            self.label_to_indices[int(y)].append(idx)

        self.classes = sorted(list(self.label_to_indices.keys()))

        if hard_class_weight_by_label is None:
            hard_class_weight_by_label = {}

        counts = Counter(self.labels.tolist())

        class_weights = []

        for c in self.classes:
            freq_weight = 1.0 / math.sqrt(float(counts[c]))
            hard_w = float(hard_class_weight_by_label.get(int(c), 1.0))
            hard_w = max(1.0, min(self.hard_weight_max, hard_w))

            # Hard weight chỉ ảnh hưởng nhẹ.
            class_weights.append(freq_weight * hard_w)

        class_weights = np.asarray(class_weights, dtype=np.float64)
        class_weights = class_weights / class_weights.sum()

        self.class_probs = class_weights

        if num_batches is None:
            num_batches = math.ceil(len(self.labels) / self.batch_size)

        self.num_batches = int(num_batches)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + np.random.randint(0, 1_000_000))

        for _ in range(self.num_batches):
            replace_classes = len(self.classes) < self.classes_per_batch

            chosen_classes = rng.choice(
                self.classes,
                size=self.classes_per_batch,
                replace=replace_classes,
                p=self.class_probs,
            )

            batch = []

            for c in chosen_classes:
                idxs = self.label_to_indices[int(c)]
                replace_samples = len(idxs) < self.samples_per_class

                chosen_idxs = rng.choice(
                    idxs,
                    size=self.samples_per_class,
                    replace=replace_samples,
                )

                batch.extend([int(x) for x in chosen_idxs])

            rng.shuffle(batch)
            yield batch


class M24MetricVisualV2Model(nn.Module):
    """
    M24 v2:
    - Không dùng CosFace.
    - Linear classifier bình thường để ổn định validation.
    - Embedding vẫn được SupCon regularize.
    """

    def __init__(
        self,
        num_classes,
        backbone_name="convnext_tiny.in12k_ft_in1k",
        embedding_dim=512,
        dropout_p=0.45,
        pretrained=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = int(self.encoder.num_features)

        self.neck = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, images):
        feat = self.encoder(images)
        emb = self.neck(feat)
        emb_norm = F.normalize(emb, dim=1)
        logits = self.classifier(emb)

        return {
            "logits": logits,
            "embedding": emb_norm,
        }


class WeightedCELoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, labels, sample_weights=None):
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings, labels):
        device = embeddings.device

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        embeddings = F.normalize(embeddings, dim=1)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        batch_size = labels.shape[0]

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return embeddings.new_tensor(0.0)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()

        return loss


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


def build_class_weights(train_df, num_classes, device, exponent=0.12):
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


def load_metadata_csv(path, split_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} metadata CSV: {path}")

    df = pd.read_csv(path)

    required = [
        "pill_crop_path",
        "mapped_label",
        "hard_class_weight",
        "is_hard_class",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{split_name} metadata missing columns: {missing}")

    df["mapped_label"] = df["mapped_label"].astype(int)
    df = check_pill_paths(df, split_name)

    return df


def build_hard_weight_by_label(train_df, hard_weight_max=1.3):
    tmp = (
        train_df.groupby("mapped_label")["hard_class_weight"]
        .mean()
        .reset_index()
    )

    out = {}

    for _, row in tmp.iterrows():
        label = int(row["mapped_label"])
        w = float(row["hard_class_weight"])
        w = max(1.0, min(float(hard_weight_max), w))
        out[label] = w

    return out


def train_one_epoch(
    model,
    loader,
    ce_criterion,
    supcon_criterion,
    optimizer,
    scaler,
    device,
    ce_weight,
    supcon_weight,
    clip_grad_norm,
):
    model.train()

    running_total = 0.0
    running_ce = 0.0
    running_supcon = 0.0

    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc="M24 v2 Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        hard_weights = batch["hard_weight"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)

            logits = outputs["logits"]
            emb = outputs["embedding"]

            ce_loss = ce_criterion(logits, labels, sample_weights=hard_weights)
            supcon_loss = supcon_criterion(emb, labels)

            loss = ce_weight * ce_loss + supcon_weight * supcon_loss

        scaler.scale(loss).backward()

        if clip_grad_norm and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)

        bs = images.size(0)

        running_total += loss.item() * bs
        running_ce += ce_loss.item() * bs
        running_supcon += supcon_loss.item() * bs

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    n = max(1, len(all_labels))

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": running_total / n,
        "ce_loss": running_ce / n,
        "supcon_loss": running_supcon / n,
        "acc": float(acc),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def validate_one_epoch(model, loader, ce_criterion, device, num_classes):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_conf = []

    for batch in tqdm(loader, desc="M24 v2 Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        hard_weights = batch["hard_weight"].to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)

            logits = outputs["logits"]
            loss = ce_criterion(logits, labels, sample_weights=hard_weights)
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

    print("=== M24 METRIC VISUAL V2 TRAINING ===")
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

    train_dataset = M24V2HardMetricDataset(
        train_df,
        transform=train_tfms,
        hard_weight_max=args.hard_weight_max,
    )

    val_dataset = M24V2HardMetricDataset(
        val_df,
        transform=val_tfms,
        hard_weight_max=args.hard_weight_max,
    )

    hard_weight_by_label = build_hard_weight_by_label(
        train_df,
        hard_weight_max=args.hard_weight_max,
    )

    sampler = ClassBalancedBatchSampler(
        labels=train_df["mapped_label"].astype(int).tolist(),
        hard_class_weight_by_label=hard_weight_by_label,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        hard_weight_max=args.hard_weight_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = M24MetricVisualV2Model(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        embedding_dim=args.embedding_dim,
        dropout_p=args.dropout_p,
        pretrained=True,
    ).to(device)

    print("Created M24MetricVisualV2Model")
    print("Backbone:", args.backbone_name)
    print("Embedding dim:", args.embedding_dim)
    print("Dropout:", args.dropout_p)
    print("Batch design:", args.classes_per_batch, "classes x", args.samples_per_class, "samples")
    print("Loss: CE + SupCon")
    print("No CosFace")
    print("No HardPair loss")
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

    ce_criterion = WeightedCELoss(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    supcon_criterion = SupConLoss(
        temperature=args.supcon_temperature,
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

    output_dir = Path(args.output_dir)

    best_path = output_dir / args.best_name
    last_path = output_dir / args.last_name

    config = {
        "stage": "M24_metric_visual_v2",
        "train_metadata": args.train_metadata,
        "val_metadata": args.val_metadata,
        "test_metadata": args.test_metadata,
        "idx_to_label": args.idx_to_label,
        "num_classes": num_classes,
        "backbone_name": args.backbone_name,
        "embedding_dim": args.embedding_dim,
        "dropout_p": args.dropout_p,
        "image_size": args.image_size,
        "classes_per_batch": args.classes_per_batch,
        "samples_per_class": args.samples_per_class,
        "class_weight_exponent": args.class_weight_exponent,
        "hard_weight_max": args.hard_weight_max,
        "ce_weight": args.ce_weight,
        "supcon_weight": args.supcon_weight,
        "supcon_temperature": args.supcon_temperature,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_effective_size": args.classes_per_batch * args.samples_per_class,
        "no_cosface": True,
        "no_hardpair_loss": True,
    }

    save_json(config, output_dir / "m24_metric_visual_v2_config.json")
    save_json({str(k): int(v) for k, v in idx_to_label.items()}, output_dir / "idx_to_label.json")

    for epoch in range(1, args.epochs + 1):
        print(f"\nM24 Metric Visual v2 Epoch [{epoch}/{args.epochs}]")
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

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            ce_criterion=ce_criterion,
            supcon_criterion=supcon_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ce_weight=args.ce_weight,
            supcon_weight=args.supcon_weight,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_result = validate_one_epoch(
            model=model,
            loader=val_loader,
            ce_criterion=ce_criterion,
            device=device,
            num_classes=num_classes,
        )

        scheduler.step()

        print(
            f"Train Loss: {train_result['loss']:.4f} | "
            f"CE: {train_result['ce_loss']:.4f} | "
            f"SupCon: {train_result['supcon_loss']:.4f} | "
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

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_ce_loss": train_result["ce_loss"],
            "train_supcon_loss": train_result["supcon_loss"],
            "train_acc": train_result["acc"],
            "train_macro_f1": train_result["macro_f1"],
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
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "neck_state_dict": model.neck.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "num_classes": num_classes,
            "idx_to_label": idx_to_label,
            "backbone_name": args.backbone_name,
            "embedding_dim": args.embedding_dim,
            "dropout_p": args.dropout_p,
            "image_size": args.image_size,
            "val_macro_f1_present": val_result["macro_f1_present"],
            "val_macro_f1_all": val_result["macro_f1_all"],
            "val_acc": val_result["accuracy"],
            "model_type": "M24MetricVisualV2Model",
            "stage": "M24_metric_visual_v2",
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
                output_path=output_dir / "best_val_per_class_metrics.csv",
            )

            print(f"Saved best checkpoint: {best_path}")
            print(f"Best Val Macro F1 Present: {best_f1:.4f}")
        else:
            bad_epochs += 1
            print(f"No significant improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    print("\nM24 metric visual v2 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1 Present:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M24 v2 metric visual model: stable CE + SupCon without CosFace/HardPair."
    )

    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--val_metadata", type=str, required=True)
    parser.add_argument("--test_metadata", type=str, required=True)
    parser.add_argument("--idx_to_label", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M24_metric_learning/train_run_v2_stable",
    )

    parser.add_argument("--best_name", type=str, default="M24_metric_visual_v2_best.pth")
    parser.add_argument("--last_name", type=str, default="M24_metric_visual_v2_last.pth")

    parser.add_argument("--backbone_name", type=str, default="convnext_tiny.in12k_ft_in1k")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.45)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--classes_per_batch", type=int, default=16)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.12)
    parser.add_argument("--hard_weight_max", type=float, default=1.3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.05)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)

    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

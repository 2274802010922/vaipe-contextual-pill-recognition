import os
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    BestPIKADataset,
    add_mapped_columns,
    check_image_paths,
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


def build_class_weights(train_df, num_classes, device, exponent=0.35):
    counts = Counter(train_df["mapped_label"].tolist())

    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(1.0 / (float(c) ** exponent))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def build_weighted_sampler(train_df, exponent=0.20, max_weight_ratio=8.0):
    labels = train_df["mapped_label"].tolist()
    counts = Counter(labels)

    sample_weights = []
    for y in labels:
        sample_weights.append(1.0 / (float(counts[y]) ** exponent))

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    if max_weight_ratio is not None and max_weight_ratio > 0:
        min_w = float(sample_weights.min())
        sample_weights = torch.clamp(sample_weights, max=min_w * max_weight_ratio)

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


class MultiSampleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_ps=(0.35, 0.45, 0.55, 0.65)):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_ps])
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.feature(x)
        logits = 0.0

        for dropout in self.dropouts:
            logits = logits + self.fc(dropout(h))

        logits = logits / len(self.dropouts)
        return logits


class M21StrongVisualPIKA(nn.Module):
    """
    Stronger architecture than M19.

    Key changes:
    - stronger pill backbone by default: convnext_tiny.in12k_ft_in1k
    - common_dim 128 instead of 64
    - graph embeddings are projected into common_dim
    - richer fusion with 9 interaction blocks
    - multi-sample dropout classifier
    - context attention with context dropout
    """

    def __init__(
        self,
        num_classes,
        graph_embeddings,
        pill_model_name="convnext_tiny.in12k_ft_in1k",
        pres_model_name="resnet18.a1_in1k",
        common_dim=128,
        hidden_dim=384,
        dropout_p=0.50,
        context_dropout_p=0.20,
        pretrained=True,
        train_graph_embeddings=False,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.common_dim = int(common_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dropout_p = float(context_dropout_p)

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

        pill_dim = int(self.pill_encoder.num_features)
        pres_dim = int(self.pres_encoder.num_features)

        graph_embeddings = torch.tensor(graph_embeddings, dtype=torch.float32)
        graph_dim = int(graph_embeddings.shape[1])

        self.graph_embedding = nn.Embedding.from_pretrained(
            graph_embeddings,
            freeze=not train_graph_embeddings,
        )

        self.pill_projection = nn.Sequential(
            nn.Linear(pill_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, common_dim),
        )

        self.pres_projection = nn.Sequential(
            nn.Linear(pres_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, common_dim),
        )

        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, common_dim),
        )

        self.context_query = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        self.context_refiner = nn.Sequential(
            nn.Linear(common_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, common_dim),
        )

        self.context_gate = nn.Sequential(
            nn.Linear(common_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, common_dim),
            nn.Sigmoid(),
        )

        # Fusion:
        # pill, prescription, context,
        # pill*context, pill*pres, pres*context,
        # abs(pill-context), abs(pill-pres), abs(pres-context)
        fusion_dim = common_dim * 9

        self.classifier = MultiSampleClassifier(
            input_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_ps=(0.35, 0.45, 0.55, 0.65),
        )

        self.temperature = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))

    def apply_context_dropout(self, context_mask):
        if not self.training:
            return context_mask

        if self.context_dropout_p <= 0:
            return context_mask

        original = context_mask.bool()

        keep = torch.rand_like(context_mask.float()) > self.context_dropout_p
        dropped = original & keep

        original_has = original.any(dim=1)
        dropped_has = dropped.any(dim=1)

        restore = original_has & (~dropped_has)
        if restore.any():
            dropped[restore] = original[restore]

        return dropped

    def get_projected_graph_weight(self):
        graph_weight = self.graph_embedding.weight
        graph_z = self.graph_projection(graph_weight)
        graph_z = nn.functional.normalize(graph_z, dim=1)
        return graph_z

    def attend_context(self, pill_z, pres_z, context_indices, context_mask):
        context_indices = context_indices.clamp(min=0, max=self.num_classes - 1)

        context_mask = self.apply_context_dropout(context_mask)
        mask = context_mask.bool()

        raw_context = self.graph_embedding(context_indices)
        context_z = self.graph_projection(raw_context)
        context_z = nn.functional.normalize(context_z, dim=-1)

        query = self.context_query(torch.cat([pill_z, pres_z], dim=1))
        query = nn.functional.normalize(query, dim=1)

        scores = torch.sum(context_z * query.unsqueeze(1), dim=-1)
        scores = scores / np.sqrt(float(self.common_dim))
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=1)
        attn = attn * mask.float()

        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn = attn / denom

        context_vec = torch.sum(attn.unsqueeze(-1) * context_z, dim=1)

        has_context = mask.any(dim=1).float().unsqueeze(1)
        context_vec = context_vec * has_context

        return context_vec, attn

    def forward(self, pill_imgs, pres_imgs, context_indices, context_mask):
        pill_feat = self.pill_encoder(pill_imgs)
        pres_feat = self.pres_encoder(pres_imgs)

        pill_z = self.pill_projection(pill_feat)
        pres_z = self.pres_projection(pres_feat)

        pill_z = nn.functional.normalize(pill_z, dim=1)
        pres_z = nn.functional.normalize(pres_z, dim=1)

        raw_context_z, attn = self.attend_context(
            pill_z=pill_z,
            pres_z=pres_z,
            context_indices=context_indices,
            context_mask=context_mask,
        )

        refined_context = self.context_refiner(
            torch.cat([pill_z, pres_z, raw_context_z], dim=1)
        )
        refined_context = nn.functional.normalize(refined_context, dim=1)

        gate = self.context_gate(torch.cat([pill_z, pres_z, refined_context], dim=1))
        context_z = gate * refined_context + (1.0 - gate) * pres_z
        context_z = nn.functional.normalize(context_z, dim=1)

        fusion = torch.cat(
            [
                pill_z,
                pres_z,
                context_z,
                pill_z * context_z,
                pill_z * pres_z,
                pres_z * context_z,
                torch.abs(pill_z - context_z),
                torch.abs(pill_z - pres_z),
                torch.abs(pres_z - context_z),
            ],
            dim=1,
        )

        main_logits = self.classifier(fusion)

        graph_z = self.get_projected_graph_weight()
        pseudo_logits = torch.matmul(pill_z, graph_z.t()) * self.temperature.clamp(min=1.0, max=30.0)

        return {
            "main_logits": main_logits,
            "pseudo_logits": pseudo_logits,
            "pill_z": pill_z,
            "pres_z": pres_z,
            "context_z": context_z,
            "attn": attn,
        }


class M21Loss(nn.Module):
    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.02,
        pseudo_loss_weight=0.20,
        link_loss_weight=0.05,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.pseudo_loss_weight = float(pseudo_loss_weight)
        self.link_loss_weight = float(link_loss_weight)

    def forward(self, outputs, labels, model):
        main_logits = outputs["main_logits"]
        pseudo_logits = outputs["pseudo_logits"]
        pill_z = outputs["pill_z"]

        main_loss = nn.functional.cross_entropy(
            main_logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        pseudo_loss = nn.functional.cross_entropy(
            pseudo_logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        graph_z = model.get_projected_graph_weight()
        target_graph = graph_z[labels]
        target_graph = nn.functional.normalize(target_graph, dim=1)

        link_loss = nn.functional.mse_loss(pill_z, target_graph)

        total = (
            main_loss
            + self.pseudo_loss_weight * pseudo_loss
            + self.link_loss_weight * link_loss
        )

        return total, {
            "main_loss": float(main_loss.detach().cpu()),
            "pseudo_loss": float(pseudo_loss.detach().cpu()),
            "link_loss": float(link_loss.detach().cpu()),
        }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, clip_grad_norm=1.0):
    model.train()

    running_loss = 0.0
    running_main = 0.0
    running_pseudo = 0.0
    running_link = 0.0

    all_preds = []
    all_labels = []

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(
        loader, desc="M21 Training", leave=False
    ):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
            loss, parts = criterion(outputs, labels, model)

        scaler.scale(loss).backward()

        if clip_grad_norm and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        b = pill_imgs.size(0)

        running_loss += loss.item() * b
        running_main += parts["main_loss"] * b
        running_pseudo += parts["pseudo_loss"] * b
        running_link += parts["link_loss"] * b

        preds = outputs["main_logits"].argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    n = max(1, len(loader.dataset))

    return (
        running_loss / n,
        running_main / n,
        running_pseudo / n,
        running_link / n,
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    )


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(
        loader, desc="M21 Validation", leave=False
    ):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
            loss, _ = criterion(outputs, labels, model)

        running_loss += loss.item() * pill_imgs.size(0)

        preds = outputs["main_logits"].argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    n = max(1, len(loader.dataset))

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return (
        running_loss / n,
        accuracy_score(all_labels, all_preds),
        precision,
        recall,
        macro_f1,
    )


def prepare_dataframe(csv_path, label_to_idx, split_name):
    df = pd.read_csv(csv_path)
    df = add_mapped_columns(df, label_to_idx)
    df = check_image_paths(df, split_name)
    return df


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

    print("=== M21 STRONG VISUAL PIKA TRAINING ===")
    print("Using device:", device)
    print("Train CSV:", args.train_csv)
    print("Val CSV:", args.val_csv)
    print("Test CSV:", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    graph_dir = Path(args.graph_artifacts_dir)

    label_to_idx = normalize_label_to_idx(load_json(graph_dir / "label_to_idx.json"))
    idx_to_label = normalize_idx_to_label(load_json(graph_dir / "idx_to_label.json"))
    graph_embeddings = np.load(graph_dir / "graph_embeddings.npy").astype(np.float32)

    num_classes = len(label_to_idx)

    print("Num classes:", num_classes)
    print("Graph embeddings shape:", graph_embeddings.shape)

    train_df = prepare_dataframe(args.train_csv, label_to_idx, "Train")
    val_df = prepare_dataframe(args.val_csv, label_to_idx, "Val")

    train_labels = set(train_df["mapped_label"].unique())
    val_labels = set(val_df["mapped_label"].unique())

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Train labels:", len(train_labels))
    print("Val labels:", len(val_labels))
    print("Labels in train but missing in val:", len(train_labels - val_labels))
    print(sorted(list(train_labels - val_labels)))

    save_json({str(k): int(v) for k, v in label_to_idx.items()}, os.path.join(args.output_dir, "label_to_idx.json"))
    save_json({str(k): int(v) for k, v in idx_to_label.items()}, os.path.join(args.output_dir, "idx_to_label.json"))

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

    model = M21StrongVisualPIKA(
        num_classes=num_classes,
        graph_embeddings=graph_embeddings,
        pill_model_name=args.pill_model_name,
        pres_model_name=args.pres_model_name,
        common_dim=args.common_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        context_dropout_p=args.context_dropout_p,
        pretrained=True,
        train_graph_embeddings=args.train_graph_embeddings,
    ).to(device)

    print("Created M21StrongVisualPIKA.")
    print("Pill model:", args.pill_model_name)
    print("Prescription model:", args.pres_model_name)
    print("Common dim:", args.common_dim)
    print("Hidden dim:", args.hidden_dim)
    print("Dropout:", args.dropout_p)
    print("Context dropout:", args.context_dropout_p)
    print("Max context len:", args.max_context_len)
    print("Train graph embeddings:", args.train_graph_embeddings)

    if args.freeze_backbone_epochs > 0:
        freeze_visual_backbones(model, freeze=True)

    total_params, trainable_params = count_trainable_params(model)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    class_weights = build_class_weights(
        train_df,
        num_classes=num_classes,
        device=device,
        exponent=args.class_weight_exponent,
    )

    criterion = M21Loss(
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

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_path = os.path.join(args.output_dir, args.best_name)
    last_path = os.path.join(args.output_dir, args.last_name)

    config = {
        "stage": "M21_strong_visual_pika",
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "graph_artifacts_dir": args.graph_artifacts_dir,
        "num_classes": num_classes,
        "pill_model_name": args.pill_model_name,
        "pres_model_name": args.pres_model_name,
        "common_dim": args.common_dim,
        "hidden_dim": args.hidden_dim,
        "max_context_len": args.max_context_len,
        "dropout_p": args.dropout_p,
        "context_dropout_p": args.context_dropout_p,
        "train_graph_embeddings": args.train_graph_embeddings,
        "pseudo_loss_weight": args.pseudo_loss_weight,
        "link_loss_weight": args.link_loss_weight,
        "class_weight_exponent": args.class_weight_exponent,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    save_json(config, os.path.join(args.output_dir, "m21_train_config.json"))

    for epoch in range(1, args.epochs + 1):
        print(f"\nM21 Strong Visual PIKA Epoch [{epoch}/{args.epochs}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                freeze_visual_backbones(model, freeze=True)
            elif epoch == args.freeze_backbone_epochs + 1:
                freeze_visual_backbones(model, freeze=False)
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

        train_loss, train_main, train_pseudo, train_link, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
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
            "common_dim": args.common_dim,
            "hidden_dim": args.hidden_dim,
            "max_context_len": args.max_context_len,
            "graph_artifacts_dir": args.graph_artifacts_dir,
            "graph_embeddings_path": str(graph_dir / "graph_embeddings.npy"),
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "M21StrongVisualPIKA",
            "stage": "M21_strong_visual_pika",
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

    print("\nM21 training done.")
    print("Best checkpoint:", best_path)
    print("Best Val Macro F1:", best_f1)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train M21 strong visual PIKA."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M21_strong_visual_pika/train_run",
    )

    parser.add_argument("--best_name", type=str, default="M21_strong_visual_pika_best.pth")
    parser.add_argument("--last_name", type=str, default="M21_strong_visual_pika_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="convnext_tiny.in12k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")

    parser.add_argument("--common_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--max_context_len", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--pseudo_loss_weight", type=float, default=0.20)
    parser.add_argument("--link_loss_weight", type=float, default=0.05)

    parser.add_argument("--dropout_p", type=float, default=0.50)
    parser.add_argument("--context_dropout_p", type=float, default=0.25)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--sampler_exponent", type=float, default=0.20)
    parser.add_argument("--max_sample_weight_ratio", type=float, default=8.0)

    parser.add_argument("--train_graph_embeddings", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)

    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

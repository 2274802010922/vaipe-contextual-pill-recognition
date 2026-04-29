import os
import json
import math
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_adjacency(adj: np.ndarray):
    adj = adj.astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)

    degree = adj.sum(axis=1)
    degree = np.where(degree == 0, 1.0, degree)

    d_inv_sqrt = np.power(degree, -0.5)
    d_mat = np.diag(d_inv_sqrt)

    return d_mat @ adj @ d_mat


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


class GraphPIKADataset(Dataset):
    def __init__(self, df, max_context_len, pill_transform=None, pres_transform=None):
        self.df = df.reset_index(drop=True)
        self.max_context_len = max_context_len
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
        context_labels = context_labels[:self.max_context_len]

        padded = [-1] * self.max_context_len
        mask = [0] * self.max_context_len

        for i, val in enumerate(context_labels):
            padded[i] = int(val)
            mask[i] = 1

        context_indices = torch.tensor(padded, dtype=torch.long)
        context_mask = torch.tensor(mask, dtype=torch.bool)

        return pill_img, pres_img, context_indices, context_mask, label


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)

        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, adj):
        x = self.node_embeddings

        x = self.conv1(x, adj)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, adj)
        x = self.norm2(x)
        x = torch.relu(x)

        return x


class GraphContextPIKA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        adj_matrix,
        pill_model_name: str,
        pres_model_name: str,
        hidden_dim: int = 256,
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

        self.pill_proj = nn.Linear(pill_dim, hidden_dim)
        self.pres_proj = nn.Linear(pres_dim, hidden_dim)

        self.graph_encoder = GraphEncoder(
            num_nodes=num_classes,
            hidden_dim=hidden_dim,
        )

        self.register_buffer("adj_matrix", adj_matrix)

        self.context_gate = nn.Sequential(
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

    def forward(self, pill_img, pres_img, context_indices, context_mask):
        pill_feat = self.pill_proj(self.pill_encoder(pill_img))
        pres_feat = self.pres_proj(self.pres_encoder(pres_img))

        graph_nodes = self.graph_encoder(self.adj_matrix)

        safe_indices = context_indices.clone()
        safe_indices[safe_indices < 0] = 0

        context_emb = graph_nodes[safe_indices]

        scores = (context_emb * pill_feat.unsqueeze(1)).sum(dim=-1) / math.sqrt(context_emb.size(-1))
        scores = scores.masked_fill(~context_mask, -1e9)

        attn_weights = torch.zeros_like(scores)
        has_context = context_mask.any(dim=1)

        if has_context.any():
            attn_weights[has_context] = torch.softmax(scores[has_context], dim=1)

        graph_context = (attn_weights.unsqueeze(-1) * context_emb).sum(dim=1)

        gate_input = torch.cat([pill_feat, pres_feat, graph_context], dim=1)
        gates = self.context_gate(gate_input)

        pill_feat = pill_feat * gates[:, 0].unsqueeze(1)
        pres_feat = pres_feat * gates[:, 1].unsqueeze(1)
        graph_context = graph_context * gates[:, 2].unsqueeze(1)

        fused = torch.cat([pill_feat, pres_feat, graph_context], dim=1)
        fused = self.fusion(fused)

        out = self.classifier(fused)
        return out


def build_graph_matrix(graph_labels_json, graph_pmi_npy, idx_to_label, device):
    graph_labels = load_json(graph_labels_json)
    graph_labels = [int(x) for x in graph_labels]

    pmi = np.load(graph_pmi_npy)

    print("Graph PMI shape:", pmi.shape)

    graph_label_to_idx = {int(label): idx for idx, label in enumerate(graph_labels)}
    ordered_old_labels = [int(idx_to_label[i]) for i in range(len(idx_to_label))]

    missing = [label for label in ordered_old_labels if label not in graph_label_to_idx]

    if len(missing) > 0:
        raise RuntimeError(f"Graph thiếu các label này: {missing[:20]}")

    graph_indices = [graph_label_to_idx[label] for label in ordered_old_labels]

    sub_pmi = pmi[np.ix_(graph_indices, graph_indices)]
    sub_pmi = normalize_adjacency(sub_pmi)

    return torch.tensor(sub_pmi, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

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
    print("Data root:", args.data_root)

    graph_labels_json = args.graph_labels_json
    graph_pmi_npy = args.graph_pmi_npy

    if graph_labels_json == "":
        graph_labels_json = os.path.join(args.data_root, "graph_labels.json")

    if graph_pmi_npy == "":
        graph_pmi_npy = os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

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

    sub_pmi = build_graph_matrix(
        graph_labels_json=graph_labels_json,
        graph_pmi_npy=graph_pmi_npy,
        idx_to_label=idx_to_label,
        device=device,
    )

    context_lengths = train_df["context_labels_mapped"].apply(lambda x: len(json.loads(x)))
    max_context_len = max(1, min(args.max_context_len, int(context_lengths.max())))

    print("Max context length used:", max_context_len)
    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))

    with open(os.path.join(args.output_dir, "label_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_to_idx.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "idx_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=2)

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_dataset = GraphPIKADataset(
        train_df,
        max_context_len=max_context_len,
        pill_transform=train_tfms,
        pres_transform=train_tfms,
    )

    val_dataset = GraphPIKADataset(
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

    model = GraphContextPIKA(
        num_classes=num_classes,
        adj_matrix=sub_pmi,
        pill_model_name=args.pill_model_name,
        pres_model_name=args.pres_model_name,
        hidden_dim=args.hidden_dim,
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
            "max_context_len": max_context_len,
            "graph_labels_json": graph_labels_json,
            "graph_pmi_npy": graph_pmi_npy,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "model_type": "GraphContextPIKA",
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
    parser = argparse.ArgumentParser(description="Train M5 PIKA graph-context model using predefined train/val split CSV")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/model/M5_split_pika_graph")
    parser.add_argument("--best_name", type=str, default="M5_pika_graph_split_best.pth")
    parser.add_argument("--last_name", type=str, default="M5_pika_graph_split_last.pth")

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_context_len", type=int, default=8)

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

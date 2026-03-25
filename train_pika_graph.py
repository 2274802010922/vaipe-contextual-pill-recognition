import os
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = os.environ.get("PIKA_GRAPH_OUTPUT_ROOT", "/content/processed_pika_graph")
CSV_PATH = os.path.join(DATA_ROOT, "pika_graph_metadata.csv")
GRAPH_LABELS_JSON = os.path.join(DATA_ROOT, "graph_labels.json")
GRAPH_COOCCUR_NPY = os.path.join(DATA_ROOT, "graph_cooccur.npy")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42
PILL_MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
OUTPUT_DIR = "./outputs_pika_graph"
GRAPH_HIDDEN_DIM = 256


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_adjacency(adj):
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = adj.sum(axis=1)
    degree = np.where(degree == 0, 1.0, degree)
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


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


class PIKAGraphDataset(Dataset):
    def __init__(self, df, max_context_len, transform=None):
        self.df = df.reset_index(drop=True)
        self.max_context_len = max_context_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pill_img = Image.open(row["pill_crop_path"]).convert("RGB")
        if self.transform:
            pill_img = self.transform(pill_img)

        label = int(row["mapped_label"])

        context_labels = json.loads(row["context_labels_mapped"])
        context_indices = context_labels[:self.max_context_len]

        padded = [-1] * self.max_context_len
        mask = [0] * self.max_context_len

        for i, val in enumerate(context_indices):
            padded[i] = int(val)
            mask[i] = 1

        context_indices = torch.tensor(padded, dtype=torch.long)
        context_mask = torch.tensor(mask, dtype=torch.bool)

        return pill_img, context_indices, context_mask, label


def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tfms, val_tfms


class PIKAGraphModel(nn.Module):
    def __init__(self, num_classes, adj_matrix, pill_model_name, hidden_dim=256):
        super().__init__()

        self.pill_encoder = timm.create_model(
            pill_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        pill_dim = self.pill_encoder.num_features
        self.visual_proj = nn.Linear(pill_dim, hidden_dim)

        self.graph_encoder = GraphEncoder(
            num_nodes=num_classes,
            hidden_dim=hidden_dim
        )

        self.register_buffer("adj_matrix", adj_matrix)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, pill_img, context_indices, context_mask):
        visual_feat = self.pill_encoder(pill_img)
        visual_feat = self.visual_proj(visual_feat)  # [B, H]

        graph_nodes = self.graph_encoder(self.adj_matrix)  # [C, H]

        safe_indices = context_indices.clone()
        safe_indices[safe_indices < 0] = 0
        context_emb = graph_nodes[safe_indices]  # [B, L, H]

        scores = (context_emb * visual_feat.unsqueeze(1)).sum(dim=-1) / math.sqrt(context_emb.size(-1))
        scores = scores.masked_fill(~context_mask, -1e9)

        attn_weights = torch.zeros_like(scores)
        has_context = context_mask.any(dim=1)

        if has_context.any():
            attn_weights[has_context] = torch.softmax(scores[has_context], dim=1)

        context_vec = (attn_weights.unsqueeze(-1) * context_emb).sum(dim=1)  # [B, H]

        fused = torch.cat([
            visual_feat,
            context_vec,
            visual_feat - context_vec,
            visual_feat * context_vec
        ], dim=1)

        fused = self.fusion(fused)
        out = self.classifier(fused)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for pill_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pill_imgs, context_indices, context_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pill_imgs.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Validation", leave=False)
    for pill_imgs, context_indices, context_mask, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        outputs = model(pill_imgs, context_indices, context_mask)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * pill_imgs.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def main():
    seed_everything(SEED)
    ensure_dir(OUTPUT_DIR)
    device = get_device()
    print("Using device:", device)

    df = pd.read_csv(CSV_PATH)
    print("Total metadata rows:", len(df))

    with open(GRAPH_LABELS_JSON, "r", encoding="utf-8") as f:
        graph_labels = json.load(f)

    cooccur = np.load(GRAPH_COOCCUR_NPY)
    print("Graph matrix shape:", cooccur.shape)

    label_counts = Counter(df["pill_label"].tolist())
    valid_labels = sorted([label for label, count in label_counts.items() if count >= 2])

    df = df[df["pill_label"].isin(valid_labels)].copy()
    old_to_new = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    df["mapped_label"] = df["pill_label"].map(old_to_new)

    def remap_context(s):
        vals = json.loads(s)
        vals = [old_to_new[v] for v in vals if v in old_to_new]
        vals = sorted(list(set(vals)))
        return json.dumps(vals)

    df["context_labels_mapped"] = df["context_labels"].apply(remap_context)

    print("Rows after filtering rare classes:", len(df))
    print("Usable classes:", df["mapped_label"].nunique())
    print("Min mapped label:", int(df["mapped_label"].min()))
    print("Max mapped label:", int(df["mapped_label"].max()))

    graph_label_to_idx = {label: idx for idx, label in enumerate(graph_labels)}
    graph_indices = [graph_label_to_idx[label] for label in valid_labels]
    sub_cooccur = cooccur[np.ix_(graph_indices, graph_indices)]
    sub_cooccur = normalize_adjacency(sub_cooccur)
    sub_cooccur = torch.tensor(sub_cooccur, dtype=torch.float32).to(device)

    context_lengths = df["context_labels_mapped"].apply(lambda x: len(json.loads(x)))
    max_context_len = max(1, int(context_lengths.max()))
    print("Max context length:", max_context_len)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["mapped_label"]
    )

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))

    train_tfms, val_tfms = build_transforms(IMAGE_SIZE)

    train_dataset = PIKAGraphDataset(
        train_df,
        max_context_len=max_context_len,
        transform=train_tfms
    )
    val_dataset = PIKAGraphDataset(
        val_df,
        max_context_len=max_context_len,
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = df["mapped_label"].nunique()
    model = PIKAGraphModel(
        num_classes=num_classes,
        adj_matrix=sub_cooccur,
        pill_model_name=PILL_MODEL_NAME,
        hidden_dim=GRAPH_HIDDEN_DIM
    ).to(device)

    train_label_counts = Counter(train_df["mapped_label"].tolist())
    class_weights = []
    for class_id in range(num_classes):
        count = train_label_counts.get(class_id, 1)
        class_weights.append(1.0 / count)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0
    best_labels, best_preds = None, None

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_labels, val_preds = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_labels, best_preds = val_labels, val_preds
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_pika_graph_model.pth"))
            print(f"Saved best model with Val Macro F1 = {best_f1:.4f}")

    print("\nBest Val Macro F1:", round(best_f1, 4))

    if best_labels is not None and best_preds is not None:
        print("\nClassification report on best validation set:")
        print(classification_report(best_labels, best_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = os.environ.get("PIKA_BEST_OUTPUT_ROOT", "/content/processed_pika_best")
CSV_PATH = os.path.join(DATA_ROOT, "best_pika_metadata.csv")
GRAPH_LABELS_JSON = os.path.join(DATA_ROOT, "graph_labels.json")
GRAPH_PMI_NPY = os.path.join(DATA_ROOT, "graph_pmi.npy")

BASE_CHECKPOINT = os.environ.get(
    "PIKA_BASE_CHECKPOINT",
    "/content/vaipe-contextual-pill-recognition/outputs_best_pika/best_pika_model.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 20
EPOCHS = 6
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

PILL_MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
PRES_MODEL_NAME = "resnet18.a1_in1k"

HIDDEN_DIM = 256
MAX_CONTEXT_LEN = 8
OUTPUT_DIR = "./outputs_best_pika_finetune"


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


def effective_num_weights(counts, beta=0.999):
    weights = []
    for c in counts:
        c = max(int(c), 1)
        eff_num = 1.0 - (beta ** c)
        w = (1.0 - beta) / eff_num
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return weights


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


class BestPIKADataset(Dataset):
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


def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
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


class BestPIKAModel(nn.Module):
    def __init__(self, num_classes, adj_matrix, pill_model_name, pres_model_name, hidden_dim=256):
        super().__init__()

        self.pill_encoder = timm.create_model(
            pill_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.pres_encoder = timm.create_model(
            pres_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        pill_dim = self.pill_encoder.num_features
        pres_dim = self.pres_encoder.num_features

        self.pill_proj = nn.Linear(pill_dim, hidden_dim)
        self.pres_proj = nn.Linear(pres_dim, hidden_dim)

        self.graph_encoder = GraphEncoder(num_nodes=num_classes, hidden_dim=hidden_dim)
        self.register_buffer("adj_matrix", adj_matrix)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, 512),
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

        graph_ctx = (attn_weights.unsqueeze(-1) * context_emb).sum(dim=1)

        gate_input = torch.cat([pill_feat, pres_feat, graph_ctx], dim=1)
        gates = self.gate(gate_input)

        pill_feat = pill_feat * gates[:, 0].unsqueeze(1)
        pres_feat = pres_feat * gates[:, 1].unsqueeze(1)
        graph_ctx = graph_ctx * gates[:, 2].unsqueeze(1)

        fused = torch.cat([
            pill_feat,
            pres_feat,
            graph_ctx,
            pill_feat * pres_feat,
            pill_feat * graph_ctx,
            pres_feat * graph_ctx
        ], dim=1)

        fused = self.fusion(fused)
        out = self.classifier(fused)
        return out


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

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

    pmi = np.load(GRAPH_PMI_NPY)
    print("Graph PMI shape:", pmi.shape)

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

    graph_label_to_idx = {label: idx for idx, label in enumerate(graph_labels)}
    graph_indices = [graph_label_to_idx[label] for label in valid_labels]
    sub_pmi = pmi[np.ix_(graph_indices, graph_indices)]
    sub_pmi = normalize_adjacency(sub_pmi)
    sub_pmi = torch.tensor(sub_pmi, dtype=torch.float32).to(device)

    context_lengths = df["context_labels_mapped"].apply(lambda x: len(json.loads(x)))
    max_context_len = max(1, min(MAX_CONTEXT_LEN, int(context_lengths.max())))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["mapped_label"]
    )

    print("Rows after filtering rare classes:", len(df))
    print("Usable classes:", df["mapped_label"].nunique())
    print("Min mapped label:", int(df["mapped_label"].min()))
    print("Max mapped label:", int(df["mapped_label"].max()))
    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Max context length used:", max_context_len)

    train_tfms, val_tfms = build_transforms(IMAGE_SIZE)

    train_dataset = BestPIKADataset(train_df, max_context_len, train_tfms, train_tfms)
    val_dataset = BestPIKADataset(val_df, max_context_len, val_tfms, val_tfms)

    train_counts = Counter(train_df["mapped_label"].tolist())

    sample_weights = []
    for y in train_df["mapped_label"].tolist():
        sample_weights.append(1.0 / math.sqrt(train_counts[y]))
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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
    model = BestPIKAModel(
        num_classes=num_classes,
        adj_matrix=sub_pmi,
        pill_model_name=PILL_MODEL_NAME,
        pres_model_name=PRES_MODEL_NAME,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    if os.path.exists(BASE_CHECKPOINT):
        state = torch.load(BASE_CHECKPOINT, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {BASE_CHECKPOINT}")
    else:
        print(f"Checkpoint not found: {BASE_CHECKPOINT}")
        print("Stop here and set PIKA_BASE_CHECKPOINT correctly.")
        return

    counts_arr = [train_counts[i] for i in range(num_classes)]
    class_weights = effective_num_weights(counts_arr, beta=0.999)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1 = 0.0
    best_labels, best_preds = None, None

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_f1, val_labels, val_preds = validate_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_labels, best_preds = val_labels, val_preds
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_pika_model_finetuned.pth"))
            print(f"Saved best model with Val Macro F1 = {best_f1:.4f}")

    print("\nBest Val Macro F1:", round(best_f1, 4))
    if best_labels is not None and best_preds is not None:
        print("\nClassification report on best validation set:")
        print(classification_report(best_labels, best_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()


import os
import json
import math
import copy
import random
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import timm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_graph_artifacts(graph_dir: str, prune_ratio: float = 0.20):
    """Load graph artifacts và áp dụng edge pruning."""
    graph_dir = Path(graph_dir)

    with open(graph_dir / "label_to_idx.json") as f:
        label_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(graph_dir / "idx_to_label.json") as f:
        idx_to_label = {int(k): int(v) for k, v in json.load(f).items()}

    embeddings = np.load(graph_dir / "graph_embeddings.npy").astype(np.float32)

    # Load PMI matrix và prune edges yếu nhất
    pmi_path = graph_dir / "graph_pmi.npy"
    pmi = np.load(pmi_path).astype(np.float32)

    if prune_ratio > 0:
        nonzero_vals = pmi[pmi > 0]
        if len(nonzero_vals) > 0:
            threshold = np.percentile(nonzero_vals, prune_ratio * 100)
            pmi_pruned = pmi.copy()
            pmi_pruned[pmi_pruned < threshold] = 0
            n_before = (pmi > 0).sum()
            n_after  = (pmi_pruned > 0).sum()
            logger.info(
                f"Graph pruning {prune_ratio*100:.0f}%: edges {n_before} → {n_after}"
            )
            pmi = pmi_pruned

    logger.info(
        f"Graph embeddings: {embeddings.shape}, "
        f"labels: {len(label_to_idx)}"
    )
    return embeddings, pmi, label_to_idx, idx_to_label


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────
from torchvision import transforms as T

def get_pill_train_transforms(img_size: int = 288):
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

def get_pill_val_transforms(img_size: int = 288):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def get_rx_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class M31PIKADataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_to_idx: dict,
        graph_embeddings: np.ndarray,
        pill_transform=None,
        rx_transform=None,
        max_context_len: int = 10,
        context_dropout_p: float = 0.0,   # dùng khi training
        mode: str = "train",
    ):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx
        self.graph_embeddings = torch.tensor(graph_embeddings, dtype=torch.float32)
        self.pill_transform = pill_transform
        self.rx_transform = rx_transform
        self.max_context_len = max_context_len
        self.context_dropout_p = context_dropout_p
        self.mode = mode
        self.num_classes = len(label_to_idx)

        # Parse context_labels
        self.df["_context_list"] = self.df["context_labels"].apply(self._parse_context)

        # Map labels
        self.df["_mapped_label"] = self.df["pill_label"].apply(
            lambda x: label_to_idx.get(int(x), 0)
        )

        logger.info(
            f"[{mode}] rows={len(self.df)}, "
            f"labels={self.df['_mapped_label'].nunique()}"
        )

    def _parse_context(self, ctx):
        if pd.isna(ctx):
            return []
        try:
            if isinstance(ctx, str):
                ctx = ctx.strip()
                if ctx.startswith("["):
                    return json.loads(ctx.replace("'", '"'))
                return [int(x.strip()) for x in ctx.split(",") if x.strip()]
            return list(ctx)
        except Exception:
            return []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Pill image ──────────────────────────────────────────────
        pill_path = str(row["pill_crop_path"])
        try:
            pill_img = Image.open(pill_path).convert("RGB")
        except Exception:
            pill_img = Image.new("RGB", (288, 288), color=128)
        if self.pill_transform:
            pill_img = self.pill_transform(pill_img)

        # ── Prescription image ─────────────────────────────────────
        rx_path = str(row["prescription_image_path"])
        try:
            rx_img = Image.open(rx_path).convert("RGB")
        except Exception:
            rx_img = Image.new("RGB", (224, 224), color=200)
        if self.rx_transform:
            rx_img = self.rx_transform(rx_img)

        # ── Context graph embeddings ──────────────────────────────
        context_labels = row["_context_list"]

        # Context dropout khi training (giúp model không quá phụ thuộc context)
        if self.mode == "train" and self.context_dropout_p > 0:
            context_labels = [
                c for c in context_labels
                if random.random() > self.context_dropout_p
            ]

        # Lấy graph embeddings của các class trong context
        valid_ctx = []
        for lbl in context_labels:
            lbl_idx = self.label_to_idx.get(int(lbl), -1)
            if lbl_idx >= 0:
                valid_ctx.append(self.graph_embeddings[lbl_idx])

        if len(valid_ctx) > self.max_context_len:
            valid_ctx = valid_ctx[:self.max_context_len]

        emb_dim = self.graph_embeddings.shape[1]
        ctx_len = len(valid_ctx)

        context_embs = torch.zeros(self.max_context_len, emb_dim, dtype=torch.float32)
        context_mask = torch.zeros(self.max_context_len, dtype=torch.bool)  # True = valid

        for i, emb in enumerate(valid_ctx):
            context_embs[i] = emb
            context_mask[i] = True

        label = torch.tensor(row["_mapped_label"], dtype=torch.long)

        return {
            "pill_img": pill_img,
            "rx_img":   rx_img,
            "context_embs": context_embs,   # (max_ctx, emb_dim)
            "context_mask": context_mask,   # (max_ctx,)
            "label": label,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadContextAttention(nn.Module):
    """
    Multi-head attention: query = graph embeddings, key/value = projected visual.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        query:            (B, Nq, D)  – graph embeddings
        key_value:        (B, Nkv, D) – projected visual feature (repeated or single)
        key_padding_mask: (B, Nkv)    – True = invalid (padding)
        Returns context:  (B, D)
        """
        # Invert mask: MultiheadAttention expects True=ignore
        attn_mask = (~key_padding_mask) if key_padding_mask is not None else None
        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=attn_mask,
        )
        out = self.norm(out + query)
        # Mean pool over query positions
        return out.mean(dim=1)  # (B, D)


class M31PIKAModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 108,
        graph_emb_dim: int = 64,
        common_dim: int = 256,
        pill_backbone: str = "convnext_small.in12k_ft_in1k",
        rx_backbone: str = "resnet18.a1_in1k",
        dropout: float = 0.4,
        num_attn_heads: int = 4,
        context_dropout_p: float = 0.1,
        img_size: int = 288,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.common_dim = common_dim

        # ── Pill visual encoder ────────────────────────────────────
        self.pill_encoder = timm.create_model(
            pill_backbone,
            pretrained=True,
            num_classes=0,  # remove head
            global_pool="avg",
        )
        pill_feat_dim = self.pill_encoder.num_features

        # ── Prescription image encoder (lightweight) ──────────────
        self.rx_encoder = timm.create_model(
            rx_backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        rx_feat_dim = self.rx_encoder.num_features

        # ── Feature projections → common_dim ──────────────────────
        self.pill_proj = nn.Sequential(
            nn.Linear(pill_feat_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rx_proj = nn.Sequential(
            nn.Linear(rx_feat_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Pseudo Classifier (Section 3.3 paper) ─────────────────
        # Rough classifier dùng để filter graph info (không dùng prescription)
        self.pseudo_classifier = nn.Linear(common_dim, num_classes)

        # ── V2G Projection (Section 3.3 paper) ────────────────────
        # Map visual features → graph embedding space
        self.v2g_proj = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.Tanh(),
            nn.Linear(common_dim, common_dim // 2),
            nn.Tanh(),
            nn.Linear(common_dim // 2, graph_emb_dim),
        )

        # ── Context Attention (Section 3.4 paper) ─────────────────
        # graph_emb_dim → common_dim trước khi attention
        self.graph_emb_proj = nn.Linear(graph_emb_dim, common_dim)
        self.context_attn = MultiHeadContextAttention(
            embed_dim=common_dim,
            num_heads=num_attn_heads,
            dropout=dropout * 0.5,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # ── Fusion → Classifier ────────────────────────────────────
        # pill_z, rx_z, context_z, pill*ctx, pill*rx, |pill-ctx|, |pill-rx|
        fusion_in = common_dim * 7
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, common_dim * 2),
            nn.LayerNorm(common_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(common_dim, num_classes),
        )

        self.context_dropout_p = context_dropout_p
        logger.info(
            f"M31PIKAModel: pill_feat={pill_feat_dim}, rx_feat={rx_feat_dim}, "
            f"common_dim={common_dim}, graph_emb={graph_emb_dim}"
        )

    def encode_pill(self, pill_img):
        feat = self.pill_encoder(pill_img)          # (B, pill_feat_dim)
        return self.pill_proj(feat)                 # (B, common_dim)

    def forward(
        self,
        pill_img,
        rx_img,
        context_embs,   # (B, max_ctx, graph_emb_dim)
        context_mask,   # (B, max_ctx)  True=valid
        graph_embeddings_all=None,  # (num_classes, graph_emb_dim) – for condensed R
    ):
        B = pill_img.shape[0]

        # ── Visual features ────────────────────────────────────────
        pill_feat = self.pill_encoder(pill_img)     # (B, pill_feat_dim)
        rx_feat   = self.rx_encoder(rx_img)         # (B, rx_feat_dim)

        pill_z = self.pill_proj(pill_feat)          # (B, common_dim)
        rx_z   = self.rx_proj(rx_feat)              # (B, common_dim)

        # ── Pseudo classification ──────────────────────────────────
        pseudo_logits = self.pseudo_classifier(pill_z)  # (B, num_classes)

        # ── Condensed Relational Feature R (Eq. 4 paper) ──────────
        # R = softmax(pseudo_logits) @ graph_embeddings_all
        if graph_embeddings_all is not None:
            # (B, num_classes) x (num_classes, graph_emb_dim) → (B, graph_emb_dim)
            pseudo_probs = F.softmax(pseudo_logits, dim=-1)
            condensed_R = pseudo_probs @ graph_embeddings_all  # (B, graph_emb_dim)
            condensed_R_proj = self.graph_emb_proj(condensed_R)  # (B, common_dim)
        else:
            condensed_R_proj = torch.zeros(B, self.common_dim, device=pill_img.device)

        # ── V2G Projection ─────────────────────────────────────────
        pill_v2g = self.v2g_proj(pill_z)            # (B, graph_emb_dim)

        # ── Context Attention ──────────────────────────────────────
        # context_embs: (B, max_ctx, graph_emb_dim) → project to common_dim
        ctx_proj = self.graph_emb_proj(context_embs)  # (B, max_ctx, common_dim)

        # Dùng pill_z làm query, context_embs làm key/value
        # key_padding_mask: True = valid → invert cho attn (True = ignore)
        key_padding_mask = ~context_mask  # (B, max_ctx) True=padding

        # Query: pill_v2g projected to common_dim (broadcast over max_ctx as 1 query)
        pill_query = self.graph_emb_proj(pill_v2g).unsqueeze(1)  # (B, 1, common_dim)

        has_context = context_mask.any(dim=1)  # (B,)

        context_z = self.context_attn(
            query=pill_query,
            key_value=ctx_proj,
            key_padding_mask=key_padding_mask,
        )  # (B, common_dim)
        context_z = self.context_proj(context_z)

        # Fallback: nếu không có context, dùng condensed_R
        context_z = torch.where(
            has_context.unsqueeze(-1),
            context_z,
            condensed_R_proj,
        )

        # ── Rich Fusion ────────────────────────────────────────────
        fusion = torch.cat([
            pill_z,
            rx_z,
            context_z,
            pill_z * context_z,
            pill_z * rx_z,
            (pill_z - context_z).abs(),
            (pill_z - rx_z).abs(),
        ], dim=-1)  # (B, common_dim * 7)

        logits = self.classifier(fusion)  # (B, num_classes)

        return {
            "logits":        logits,
            "pseudo_logits": pseudo_logits,
            "pill_v2g":      pill_v2g,       # (B, graph_emb_dim) for linkage loss
        }


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────
class LinkageLoss(nn.Module):
    """
    JS Divergence giữa distribution của pill_v2g và graph embeddings.
    Mục tiêu: kéo projected visual features gần graph space.
    (Section 3.5, Eq. 8 paper)
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def _pairwise_conditional(self, X: torch.Tensor) -> torch.Tensor:
        """Tính conditional probability matrix từ cosine similarity."""
        X_norm = F.normalize(X, dim=-1)
        sim = X_norm @ X_norm.T                        # (N, N)
        sim = sim / (2 * self.sigma ** 2)
        # Mask diagonal
        mask = torch.eye(X.shape[0], device=X.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        return F.softmax(sim, dim=-1)                  # (N, N)

    def forward(self, pill_v2g: torch.Tensor, graph_embs: torch.Tensor) -> torch.Tensor:
        """
        pill_v2g:  (B, graph_emb_dim)
        graph_embs: (B, graph_emb_dim) – graph embedding của đúng class
        """
        if pill_v2g.shape[0] < 4:
            return torch.tensor(0.0, device=pill_v2g.device)

        P = self._pairwise_conditional(graph_embs)     # (B, B)
        Q = self._pairwise_conditional(pill_v2g)        # (B, B)

        eps = 1e-8
        P = P.clamp(min=eps)
        Q = Q.clamp(min=eps)

        M = 0.5 * (P + Q)
        js = 0.5 * (P * (P / M).log()).sum() + \
             0.5 * (Q * (Q / M).log()).sum()
        return js / (pill_v2g.shape[0] ** 2)


class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(log_prob, self.smoothing / self.num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_prob).sum(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.original:
                param.data = self.original[name]
        self.original = {}


# ──────────────────────────────────────────────────────────────────────────────
# Load M25 backbone weights (optional)
# ──────────────────────────────────────────────────────────────────────────────
def load_m25_visual_weights(model: M31PIKAModel, m25_checkpoint_path: str, device):
    """
    Load pill_encoder weights từ M25 checkpoint vào M31.
    M25 là visual-only model, nên chỉ có pill_encoder (backbone) tương thích.
    """
    if not Path(m25_checkpoint_path).exists():
        logger.warning(f"M25 checkpoint not found: {m25_checkpoint_path}. Skipping.")
        return model

    ckpt = torch.load(m25_checkpoint_path, map_location=device, weights_only=False)

    # M25 checkpoint có thể lưu dưới key "model_state_dict" hoặc "state_dict"
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Lọc keys thuộc pill_encoder (prefix "backbone." hoặc "pill_encoder.")
    m31_state = model.state_dict()
    matched = 0
    for m25_key, val in state_dict.items():
        # Thử map từ M25 backbone key → M31 pill_encoder key
        candidates = [
            f"pill_encoder.{m25_key}",
            m25_key.replace("backbone.", "pill_encoder."),
            m25_key.replace("model.", "pill_encoder."),
        ]
        for cand in candidates:
            if cand in m31_state and m31_state[cand].shape == val.shape:
                m31_state[cand] = val
                matched += 1
                break

    model.load_state_dict(m31_state, strict=False)
    logger.info(f"Loaded {matched} parameters from M25 checkpoint → pill_encoder.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, present_labels=None):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1_present": f1_macro,
        "weighted_f1": f1_weighted,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training / Eval loops
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, optimizer, scaler,
    graph_embeddings_tensor,
    criterion_cls, criterion_pseudo, linkage_loss_fn,
    device, alpha=0.9, beta_pseudo=0.1, gamma_link=0.05,
    grad_clip=1.0,
):
    model.train()
    total_loss = total_cls = total_pseudo = total_link = 0
    correct = total = 0

    for batch in tqdm(loader, desc="  train", leave=False, ncols=80):
        pill_img    = batch["pill_img"].to(device, non_blocking=True)
        rx_img      = batch["rx_img"].to(device, non_blocking=True)
        ctx_embs    = batch["context_embs"].to(device, non_blocking=True)
        ctx_mask    = batch["context_mask"].to(device, non_blocking=True)
        labels      = batch["label"].to(device, non_blocking=True)

        graph_embs  = graph_embeddings_tensor.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out = model(
                pill_img=pill_img,
                rx_img=rx_img,
                context_embs=ctx_embs,
                context_mask=ctx_mask,
                graph_embeddings_all=graph_embs,
            )
            logits        = out["logits"]
            pseudo_logits = out["pseudo_logits"]
            pill_v2g      = out["pill_v2g"]

            # Classification loss (main + pseudo)
            loss_cls    = criterion_cls(logits, labels)
            loss_pseudo = criterion_pseudo(pseudo_logits, labels)

            # Linkage loss: pull pill_v2g gần graph emb của đúng class
            target_graph_embs = graph_embs[labels]   # (B, graph_emb_dim)
            loss_link = linkage_loss_fn(pill_v2g, target_graph_embs)

            loss = alpha * loss_cls + beta_pseudo * loss_pseudo + gamma_link * loss_link

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        total_loss   += loss.item() * bs
        total_cls    += loss_cls.item() * bs
        total_pseudo += loss_pseudo.item() * bs
        total_link   += loss_link.item() * bs
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += bs

    n = total
    return {
        "loss":        total_loss / n,
        "loss_cls":    total_cls / n,
        "loss_pseudo": total_pseudo / n,
        "loss_link":   total_link / n,
        "accuracy":    correct / n,
    }


@torch.no_grad()
def evaluate(model, loader, graph_embeddings_tensor, device, present_labels=None):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  eval", leave=False, ncols=80):
        pill_img = batch["pill_img"].to(device, non_blocking=True)
        rx_img   = batch["rx_img"].to(device, non_blocking=True)
        ctx_embs = batch["context_embs"].to(device, non_blocking=True)
        ctx_mask = batch["context_mask"].to(device, non_blocking=True)
        labels   = batch["label"].to(device, non_blocking=True)
        graph_embs = graph_embeddings_tensor.to(device)

        with autocast():
            out = model(
                pill_img=pill_img,
                rx_img=rx_img,
                context_embs=ctx_embs,
                context_mask=ctx_mask,
                graph_embeddings_all=graph_embs,
            )
        preds = out["logits"].argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds, present_labels=present_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="M31 PIKA Strong Backbone")

    # Data
    parser.add_argument("--train_csv",       required=True)
    parser.add_argument("--val_csv",         required=True)
    parser.add_argument("--test_csv",        required=True)
    parser.add_argument("--graph_artifacts_dir", required=True)
    parser.add_argument("--output_dir",      required=True)

    # Optional: load M25 visual weights
    parser.add_argument("--m25_checkpoint",  default=None,
                        help="Path to M25 checkpoint để init pill encoder")

    # Model
    parser.add_argument("--pill_backbone",   default="convnext_small.in12k_ft_in1k")
    parser.add_argument("--rx_backbone",     default="resnet18.a1_in1k")
    parser.add_argument("--common_dim",      type=int, default=256)
    parser.add_argument("--graph_emb_dim",   type=int, default=64)
    parser.add_argument("--num_attn_heads",  type=int, default=4)
    parser.add_argument("--dropout",         type=float, default=0.4)
    parser.add_argument("--img_size",        type=int, default=288)
    parser.add_argument("--max_context_len", type=int, default=10)
    parser.add_argument("--graph_prune_ratio", type=float, default=0.20,
                        help="Prune ratio cho graph edges (0 = no pruning)")

    # Loss weights
    parser.add_argument("--alpha",           type=float, default=0.88,
                        help="Weight cho main classification loss")
    parser.add_argument("--beta_pseudo",     type=float, default=0.08,
                        help="Weight cho pseudo classification loss")
    parser.add_argument("--gamma_link",      type=float, default=0.04,
                        help="Weight cho linkage loss")
    parser.add_argument("--label_smoothing", type=float, default=0.10)

    # Training
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--lr_backbone",     type=float, default=3e-5,
                        help="LR riêng cho backbone (thường nhỏ hơn head)")
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--warmup_epochs",   type=int,   default=3)
    parser.add_argument("--context_dropout", type=float, default=0.15)
    parser.add_argument("--use_ema",         action="store_true", default=True)
    parser.add_argument("--ema_decay",       type=float, default=0.9995)
    parser.add_argument("--use_sampler",     action="store_true", default=True,
                        help="Class-balanced weighted sampler")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--num_workers",     type=int, default=4)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== M31 PIKA STRONG BACKBONE TRAINING ===")
    logger.info(f"Device: {device}")
    logger.info(f"Pill backbone: {args.pill_backbone}")
    logger.info(f"M25 checkpoint: {args.m25_checkpoint}")
    logger.info(f"Output dir: {output_dir}")

    # ── Load graph artifacts ─────────────────────────────────────
    graph_embs_np, pmi, label_to_idx, idx_to_label = load_graph_artifacts(
        args.graph_artifacts_dir, prune_ratio=args.graph_prune_ratio
    )
    num_classes = len(label_to_idx)
    graph_emb_dim = graph_embs_np.shape[1]
    graph_embs_tensor = torch.tensor(graph_embs_np, dtype=torch.float32)
    logger.info(f"Num classes: {num_classes}, graph_emb_dim: {graph_emb_dim}")

    # ── Transforms ──────────────────────────────────────────────
    pill_train_tf = get_pill_train_transforms(args.img_size)
    pill_val_tf   = get_pill_val_transforms(args.img_size)
    rx_tf         = get_rx_transforms(224)

    # ── Datasets ─────────────────────────────────────────────────
    train_ds = M31PIKADataset(
        args.train_csv, label_to_idx, graph_embs_np,
        pill_transform=pill_train_tf, rx_transform=rx_tf,
        max_context_len=args.max_context_len,
        context_dropout_p=args.context_dropout,
        mode="train",
    )
    val_ds = M31PIKADataset(
        args.val_csv, label_to_idx, graph_embs_np,
        pill_transform=pill_val_tf, rx_transform=rx_tf,
        max_context_len=args.max_context_len,
        context_dropout_p=0.0,
        mode="val",
    )
    test_ds = M31PIKADataset(
        args.test_csv, label_to_idx, graph_embs_np,
        pill_transform=pill_val_tf, rx_transform=rx_tf,
        max_context_len=args.max_context_len,
        context_dropout_p=0.0,
        mode="test",
    )

    # ── Sampler (class-balanced) ─────────────────────────────────
    if args.use_sampler:
        label_counts = Counter(train_ds.df["_mapped_label"].tolist())
        weights = [1.0 / label_counts[row] for row in train_ds.df["_mapped_label"]]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Tính present labels cho val/test (chỉ evaluate classes có trong split đó)
    val_present  = sorted(val_ds.df["_mapped_label"].unique().tolist())
    test_present = sorted(test_ds.df["_mapped_label"].unique().tolist())

    # ── Model ────────────────────────────────────────────────────
    model = M31PIKAModel(
        num_classes=num_classes,
        graph_emb_dim=graph_emb_dim,
        common_dim=args.common_dim,
        pill_backbone=args.pill_backbone,
        rx_backbone=args.rx_backbone,
        dropout=args.dropout,
        num_attn_heads=args.num_attn_heads,
        context_dropout_p=args.context_dropout,
        img_size=args.img_size,
    ).to(device)

    # Load M25 weights nếu có
    if args.m25_checkpoint:
        model = load_m25_visual_weights(model, args.m25_checkpoint, device)

    # ── Optimizer với differential LR ────────────────────────────
    backbone_params = list(model.pill_encoder.parameters()) + \
                      list(model.rx_encoder.parameters())
    head_params = [
        p for n, p in model.named_parameters()
        if not any(n.startswith(x) for x in ["pill_encoder", "rx_encoder"])
    ]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # ── LR Scheduler: warmup + cosine decay ──────────────────────
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Losses ───────────────────────────────────────────────────
    criterion_cls    = LabelSmoothingCE(num_classes, smoothing=args.label_smoothing)
    criterion_pseudo = LabelSmoothingCE(num_classes, smoothing=args.label_smoothing * 2)
    linkage_loss_fn  = LinkageLoss(sigma=1.0)

    # ── EMA ──────────────────────────────────────────────────────
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────
    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            graph_embeddings_tensor=graph_embs_tensor,
            criterion_cls=criterion_cls,
            criterion_pseudo=criterion_pseudo,
            linkage_loss_fn=linkage_loss_fn,
            device=device,
            alpha=args.alpha,
            beta_pseudo=args.beta_pseudo,
            gamma_link=args.gamma_link,
            grad_clip=args.grad_clip,
        )

        if ema is not None:
            ema.update(model)

        # Evaluate với EMA weights
        if ema is not None:
            ema.apply_shadow(model)

        val_metrics = evaluate(
            model, val_loader, graph_embs_tensor, device,
            present_labels=val_present,
        )

        if ema is not None:
            ema.restore(model)

        current_lr = optimizer.param_groups[1]["lr"]
        val_f1 = val_metrics["macro_f1_present"]

        logger.info(
            f"  Train loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, "
            f"pseudo={train_metrics['loss_pseudo']:.4f}, "
            f"link={train_metrics['loss_link']:.4f})"
        )
        logger.info(
            f"  Val   acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_f1:.4f} "
            f"weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"lr={current_lr:.2e}"
        )

        row = {
            "epoch":            epoch,
            "train_loss":       train_metrics["loss"],
            "train_loss_cls":   train_metrics["loss_cls"],
            "train_loss_pseudo":train_metrics["loss_pseudo"],
            "train_loss_link":  train_metrics["loss_link"],
            "train_acc":        train_metrics["accuracy"],
            "val_acc":          val_metrics["accuracy"],
            "val_macro_f1":     val_f1,
            "val_weighted_f1":  val_metrics["weighted_f1"],
            "lr":               current_lr,
        }
        history.append(row)

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = output_dir / "M31_pika_strong_backbone_best.pth"
            torch.save({
                "epoch":          epoch,
                "model_state_dict": model.state_dict(),
                "ema_shadow":     ema.shadow if ema else None,
                "val_macro_f1":   val_f1,
                "args":           vars(args),
            }, save_path)
            logger.info(f"  ✓ Best model saved (val_macro_f1={val_f1:.4f})")

        # Update scheduler sau mỗi step (đã gọi trong train_one_epoch qua per-step)
        # Nếu muốn per-epoch scheduler:
        scheduler.step(epoch * len(train_loader))

        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)

    # ── Load best → evaluate test ─────────────────────────────────
    logger.info("\n=== FINAL TEST EVALUATION ===")
    best_ckpt = torch.load(
        output_dir / "M31_pika_strong_backbone_best.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Apply EMA shadow nếu có
    if ema is not None and best_ckpt.get("ema_shadow"):
        ema.shadow = best_ckpt["ema_shadow"]
        ema.apply_shadow(model)

    test_metrics = evaluate(
        model, test_loader, graph_embs_tensor, device,
        present_labels=test_present,
    )

    logger.info(f"Test Accuracy      : {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Macro F1 Present: {test_metrics['macro_f1_present']:.4f}")
    logger.info(f"Test Weighted F1   : {test_metrics['weighted_f1']:.4f}")
    logger.info(f"Best Val Macro F1  : {best_val_f1:.4f}")

    # Lưu test metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({
            **test_metrics,
            "best_val_macro_f1": best_val_f1,
            "best_epoch": best_ckpt["epoch"],
        }, f, indent=2)

    logger.info(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

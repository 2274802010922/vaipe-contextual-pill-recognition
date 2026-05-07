#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M30 - Hard-error and class-balanced finetuning for VAIPE contextual pill recognition.

Why M30 exists:
- M26 is the current best post-processing/ensemble result.
- M28/M29-style post-processing can improve validation but may not improve test.
- M30 trains a real neural model again, focusing on classes where M26 has low F1/recall.

Important:
- This script uses the TEST split only for final reporting.
- Hard classes and class weights are computed from TRAIN distribution + M26 VALIDATION diagnostics.
- It does not tune on test.

Example:
python train_m30_hard_error_finetune.py \
  --train_metadata /content/drive/MyDrive/model/M24_metric_learning/hard_pair_metadata/m24_train_hard_metadata.csv \
  --val_metadata /content/drive/MyDrive/model/M24_metric_learning/hard_pair_metadata/m24_val_hard_metadata.csv \
  --test_metadata /content/drive/MyDrive/model/M24_metric_learning/hard_pair_metadata/m24_test_hard_metadata.csv \
  --m26_dir /content/drive/MyDrive/model/M26_calibrated_context_ensemble/run_v1 \
  --output_dir /content/drive/MyDrive/model/M30_hard_error_finetune/run_v1 \
  --num_classes 108 \
  --true_col true_mapped_label \
  --epochs 5 \
  --batch_size 32
"""

import argparse
import ast
import glob
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    import timm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script requires timm. In Colab run: !pip -q install timm\n"
        f"Original import error: {e}"
    )

EPS = 1e-12

LABEL_COL_CANDIDATES = [
    "true_mapped_label", "mapped_label", "label_idx", "class_idx", "target", "target_idx",
    "y_true", "label", "class_id", "medicine_id", "pill_id",
]
PRED_COL_CANDIDATES = [
    "m26_pred_mapped_label", "pred_mapped_label", "ensemble_pred_mapped_label",
    "pred_idx", "pred_label", "prediction", "y_pred",
]
PILL_PATH_CANDIDATES = [
    "pill_crop_path", "crop_path", "pill_image_path", "image_path", "path", "filename",
]
PRESCRIPTION_PATH_CANDIDATES = [
    "prescription_image_path", "prescription_path", "pres_image_path", "rx_image_path",
]
CONTEXT_COL_CANDIDATES = [
    "context_labels_mapped", "context_labels", "prescription_labels_mapped", "rx_context_labels",
]


@dataclass
class Metrics:
    accuracy: float
    macro_f1_present: float
    macro_f1_all: float
    weighted_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_one_file(directory: str, patterns: Sequence[str], required: bool = True) -> Optional[str]:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(directory, pat)))
    matches = sorted(set(matches), key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find file in {directory} with patterns: {patterns}")
    return None


def find_m26_csv(directory: str, split: str) -> str:
    patterns = [
        f"{split}_m26_predictions.csv", f"{split}_M26_predictions.csv",
        f"*{split}*m26*pred*.csv", f"*{split}*M26*pred*.csv",
        f"{split}_predictions.csv", f"*{split}*pred*.csv",
    ]
    return find_one_file(directory, patterns, required=True)  # type: ignore[return-value]


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str] = None, required: bool = True) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            if required:
                raise ValueError(f"Explicit column '{explicit}' not found. Available columns: {list(df.columns)}")
            return None
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Cannot find required column. Tried {list(candidates)}. Available columns: {list(df.columns)}")
    return None


def to_int_array(series: pd.Series, name: str) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().any():
        bad = series[vals.isna()].head(10).tolist()
        raise ValueError(f"Column '{name}' must contain numeric class indices. Bad examples: {bad}")
    return vals.astype(int).to_numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Metrics:
    labels_all = list(range(num_classes))
    present = sorted([int(x) for x in np.unique(y_true) if 0 <= int(x) < num_classes])
    if len(present) == 0:
        present = labels_all
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1_present=float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        macro_f1_all=float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0)),
    )


def metrics_row(split: str, model: str, m: Metrics) -> Dict[str, float]:
    d = asdict(m)
    d.update({"split": split, "model": model})
    return d


def print_metrics_table(rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    cols = ["split", "model", "accuracy", "macro_f1_present", "macro_f1_all", "weighted_f1"]
    print(df[cols].to_string(index=False))


def parse_context_labels(value: object, num_classes: int) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                raw = list(parsed)
            else:
                raw = [parsed]
        except Exception:
            # fallback: split common separators
            for ch in ["[", "]", "(", ")", "{", "}", "'", '"']:
                s = s.replace(ch, " ")
            for sep in [",", ";", "|"]:
                s = s.replace(sep, " ")
            raw = s.split()
    out: List[int] = []
    for x in raw:
        try:
            ix = int(float(x))
            if 0 <= ix < num_classes:
                out.append(ix)
        except Exception:
            continue
    return sorted(set(out))


def resolve_path(path_value: object, data_root: Optional[str]) -> str:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return ""
    p = str(path_value).strip()
    if not p:
        return ""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if data_root:
        p2 = os.path.join(data_root, p)
        if os.path.exists(p2):
            return p2
        # common case: metadata begins with processed folder name but root is its parent
        p3 = os.path.join(data_root, os.path.basename(p))
        if os.path.exists(p3):
            return p3
    return p


def load_rgb(path: str, image_size: int) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    # Keep training/eval running even if a few paths are bad.
    return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class PillContextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        pill_col: str,
        prescription_col: Optional[str],
        context_col: Optional[str],
        num_classes: int,
        image_size: int,
        data_root: Optional[str],
        train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.label_col = label_col
        self.pill_col = pill_col
        self.prescription_col = prescription_col
        self.context_col = context_col
        self.num_classes = num_classes
        self.image_size = image_size
        self.data_root = data_root
        self.tf = build_transforms(image_size=image_size, train=train)
        self.labels = to_int_array(self.df[label_col], label_col)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        pill_path = resolve_path(row.get(self.pill_col, ""), self.data_root)
        pres_path = resolve_path(row.get(self.prescription_col, "") if self.prescription_col else "", self.data_root)
        pill = self.tf(load_rgb(pill_path, self.image_size))
        pres = self.tf(load_rgb(pres_path, self.image_size)) if pres_path else torch.zeros_like(pill)
        ctx = torch.zeros(self.num_classes, dtype=torch.float32)
        if self.context_col:
            for c in parse_context_labels(row.get(self.context_col), self.num_classes):
                ctx[c] = 1.0
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return {"pill": pill, "prescription": pres, "context": ctx, "label": y, "index": torch.tensor(idx, dtype=torch.long)}


class M30HardErrorModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pill_backbone_name: str,
        prescription_backbone_name: str,
        hidden_dim: int = 512,
        context_dim: int = 128,
        dropout: float = 0.25,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pill_backbone = timm.create_model(pill_backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.prescription_backbone = timm.create_model(prescription_backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        pill_dim = int(getattr(self.pill_backbone, "num_features"))
        pres_dim = int(getattr(self.prescription_backbone, "num_features"))
        self.context_mlp = nn.Sequential(
            nn.Linear(num_classes, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(pill_dim + pres_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.pill_backbone.parameters():
            p.requires_grad = trainable
        for p in self.prescription_backbone.parameters():
            p.requires_grad = trainable

    def forward(self, pill: torch.Tensor, prescription: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        pf = self.pill_backbone(pill)
        rf = self.prescription_backbone(prescription)
        cf = self.context_mlp(context)
        x = torch.cat([pf, rf, cf], dim=1)
        return self.classifier(x)


def clean_checkpoint_state(raw: object) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        return {}
    state: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue
        kk = str(k)
        for prefix in ["module.", "model."]:
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        state[kk] = v
    return state


def load_partial_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"init_checkpoint not found: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    state = clean_checkpoint_state(raw)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = len(state) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded partial checkpoint: {checkpoint_path}")
    print(f"Compatible tensors loaded: {len(compatible)} | skipped shape/name mismatch: {skipped}")
    print(f"Missing after partial load: {len(missing)} | unexpected: {len(unexpected)}")


def read_m26_metrics(m26_dir: str, num_classes: int, true_col: Optional[str]) -> Tuple[List[Dict[str, object]], Optional[pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    val_diag: Optional[pd.DataFrame] = None
    for split in ["val", "test"]:
        try:
            csv_path = find_m26_csv(m26_dir, split)
            df = pd.read_csv(csv_path)
            tcol = first_existing_col(df, LABEL_COL_CANDIDATES, explicit=true_col, required=True)
            pcol = first_existing_col(df, PRED_COL_CANDIDATES, explicit=None, required=True)
            y = to_int_array(df[tcol], tcol)  # type: ignore[arg-type]
            pred = to_int_array(df[pcol], pcol)  # type: ignore[arg-type]
            m = compute_metrics(y, pred, num_classes)
            rows.append(metrics_row(split, "M26", m))
            if split == "val":
                rep = class_report(y, pred, num_classes)
                val_diag = rep
        except Exception as e:
            print(f"Warning: could not read M26 {split} metrics from {m26_dir}: {e}")
    return rows, val_diag


def class_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> pd.DataFrame:
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    pred_support = np.bincount(np.clip(y_pred, 0, num_classes - 1), minlength=num_classes)
    return pd.DataFrame({
        "class_id": labels,
        "support": support.astype(int),
        "pred_support": pred_support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def compute_class_weights(
    train_labels: np.ndarray,
    num_classes: int,
    m26_val_report: Optional[pd.DataFrame],
    hard_f1_threshold: float,
    hard_boost: float,
    rare_boost: float,
    max_weight: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    counts = np.bincount(np.clip(train_labels, 0, num_classes - 1), minlength=num_classes).astype(np.float64)
    nonzero = counts[counts > 0]
    median_count = float(np.median(nonzero)) if len(nonzero) else 1.0
    # sqrt inverse-frequency, gentle by default
    freq_weight = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            freq_weight[c] = math.sqrt(median_count / max(counts[c], 1.0))
        else:
            freq_weight[c] = 1.0
    freq_weight = 1.0 + rare_boost * (freq_weight - 1.0)

    hard_weight = np.ones(num_classes, dtype=np.float64)
    f1 = np.full(num_classes, np.nan, dtype=np.float64)
    recall = np.full(num_classes, np.nan, dtype=np.float64)
    val_support = np.zeros(num_classes, dtype=np.int64)
    if m26_val_report is not None:
        tmp = m26_val_report.set_index("class_id")
        for c in range(num_classes):
            if c in tmp.index:
                f1[c] = float(tmp.loc[c, "f1"])
                recall[c] = float(tmp.loc[c, "recall"])
                val_support[c] = int(tmp.loc[c, "support"])
                if val_support[c] > 0 and f1[c] < hard_f1_threshold:
                    gap = (hard_f1_threshold - f1[c]) / max(hard_f1_threshold, EPS)
                    hard_weight[c] += hard_boost * gap
                    if recall[c] == 0:
                        hard_weight[c] += 0.50 * hard_boost

    weight = freq_weight * hard_weight
    # no need to weight classes absent from train
    weight[counts <= 0] = 0.0
    active = weight[counts > 0]
    if len(active):
        weight[counts > 0] = weight[counts > 0] / max(float(active.mean()), EPS)
    weight = np.clip(weight, 0.0, max_weight)

    detail = pd.DataFrame({
        "class_id": np.arange(num_classes),
        "train_support": counts.astype(int),
        "m26_val_support": val_support.astype(int),
        "m26_val_recall": recall,
        "m26_val_f1": f1,
        "freq_weight": freq_weight,
        "hard_weight": hard_weight,
        "final_weight": weight,
    }).sort_values(["final_weight", "train_support"], ascending=[False, True])
    return weight.astype(np.float32), detail


def preflight_paths(df: pd.DataFrame, col: Optional[str], data_root: Optional[str], name: str, limit: int = 5) -> None:
    if not col:
        print(f"{name}: no column, will use blank images")
        return
    existing = 0
    missing_examples: List[str] = []
    for v in df[col].head(min(len(df), 2000)):
        p = resolve_path(v, data_root)
        if p and os.path.exists(p):
            existing += 1
        elif len(missing_examples) < limit:
            missing_examples.append(str(v))
    checked = min(len(df), 2000)
    print(f"{name}: existing in first {checked} rows = {existing}/{checked}")
    if missing_examples:
        print(f"{name}: missing examples: {missing_examples}")


def make_optimizer(model: nn.Module, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("pill_backbone") or name.startswith("prescription_backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp: bool,
    log_every: int,
) -> Tuple[float, Metrics]:
    model.train()
    total_loss = 0.0
    y_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        pill = batch["pill"].to(device, non_blocking=True)
        pres = batch["prescription"].to(device, non_blocking=True)
        ctx = batch["context"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(pill, pres, ctx)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        pred = torch.argmax(logits.detach(), dim=1)
        y_all.append(y.detach().cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        if log_every > 0 and step % log_every == 0:
            print(f"  step {step:05d}/{len(loader):05d} | loss={total_loss/max(n,1):.4f} | elapsed={(time.time()-t0)/60:.1f} min")

    y_np = np.concatenate(y_all) if y_all else np.array([], dtype=int)
    p_np = np.concatenate(pred_all) if pred_all else np.array([], dtype=int)
    return total_loss / max(n, 1), compute_metrics(y_np, p_np, getattr(model, "num_classes", int(y_np.max()) + 1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> Tuple[float, Metrics, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    y_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    prob_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []
    for batch in loader:
        pill = batch["pill"].to(device, non_blocking=True)
        pres = batch["prescription"].to(device, non_blocking=True)
        ctx = batch["context"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(pill, pres, ctx)
            loss = criterion(logits, y) if criterion is not None else torch.tensor(0.0, device=device)
        probs = F.softmax(logits.float(), dim=1)
        pred = torch.argmax(probs, dim=1)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        y_all.append(y.cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        prob_all.append(probs.cpu().numpy())
        idx_all.append(batch["index"].cpu().numpy())
    y_np = np.concatenate(y_all) if y_all else np.array([], dtype=int)
    p_np = np.concatenate(pred_all) if pred_all else np.array([], dtype=int)
    prob_np = np.concatenate(prob_all) if prob_all else np.zeros((0, num_classes), dtype=np.float32)
    idx_np = np.concatenate(idx_all) if idx_all else np.array([], dtype=int)
    return total_loss / max(n, 1), compute_metrics(y_np, p_np, num_classes), y_np, p_np, prob_np, idx_np


def save_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    indices: np.ndarray,
    out_csv: str,
    out_probs: str,
    pred_col: str = "m30_pred_mapped_label",
) -> None:
    out = df.iloc[indices].reset_index(drop=True).copy()
    out["m30_true_mapped_label"] = y_true.astype(int)
    out[pred_col] = y_pred.astype(int)
    out["m30_confidence"] = probs.max(axis=1) if len(probs) else []
    out["m30_is_correct"] = (y_true == y_pred).astype(int)
    topk = min(5, probs.shape[1]) if probs.ndim == 2 else 0
    if topk > 0:
        top_idx = np.argsort(-probs, axis=1)[:, :topk]
        for k in range(topk):
            out[f"m30_top{k+1}"] = top_idx[:, k].astype(int)
            out[f"m30_top{k+1}_prob"] = probs[np.arange(len(probs)), top_idx[:, k]]
    out.to_csv(out_csv, index=False)
    np.save(out_probs, probs.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--val_metadata", type=str, required=True)
    parser.add_argument("--test_metadata", type=str, required=True)
    parser.add_argument("--m26_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=108)
    parser.add_argument("--true_col", type=str, default=None)
    parser.add_argument("--pill_col", type=str, default=None)
    parser.add_argument("--prescription_col", type=str, default=None)
    parser.add_argument("--context_col", type=str, default=None)
    parser.add_argument("--pill_backbone", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--prescription_backbone", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--hard_f1_threshold", type=float, default=0.35)
    parser.add_argument("--hard_boost", type=float, default=2.0)
    parser.add_argument("--rare_boost", type=float, default=0.8)
    parser.add_argument("--max_class_weight", type=float, default=6.0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--max_train_rows", type=int, default=0, help="0 means use all train rows; positive value is for quick debug only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, "m30_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=== M30 HARD-ERROR CLASS-BALANCED FINETUNE ===")
    print(f"Train metadata: {args.train_metadata}")
    print(f"Val metadata  : {args.val_metadata}")
    print(f"Test metadata : {args.test_metadata}")
    print(f"M26 dir       : {args.m26_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Num classes   : {args.num_classes}")
    print(f"Pill backbone : {args.pill_backbone}")
    print(f"Rx backbone   : {args.prescription_backbone}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    print(f"Using device  : {device} | AMP: {amp}")

    train_df = pd.read_csv(args.train_metadata)
    val_df = pd.read_csv(args.val_metadata)
    test_df = pd.read_csv(args.test_metadata)
    if args.max_train_rows and args.max_train_rows > 0 and args.max_train_rows < len(train_df):
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Debug mode: sampled train rows to {len(train_df)}")

    label_col = first_existing_col(train_df, LABEL_COL_CANDIDATES, explicit=args.true_col, required=True)
    # val/test may use true_mapped_label; if explicit missing, fall back independently
    val_label_col = first_existing_col(val_df, LABEL_COL_CANDIDATES, explicit=args.true_col if args.true_col in val_df.columns else None, required=True)
    test_label_col = first_existing_col(test_df, LABEL_COL_CANDIDATES, explicit=args.true_col if args.true_col in test_df.columns else None, required=True)
    pill_col = first_existing_col(train_df, PILL_PATH_CANDIDATES, explicit=args.pill_col, required=True)
    prescription_col = first_existing_col(train_df, PRESCRIPTION_PATH_CANDIDATES, explicit=args.prescription_col, required=False)
    context_col = first_existing_col(train_df, CONTEXT_COL_CANDIDATES, explicit=args.context_col, required=False)

    print("\n=== DETECTED COLUMNS ===")
    print(json.dumps({
        "train_label_col": label_col,
        "val_label_col": val_label_col,
        "test_label_col": test_label_col,
        "pill_col": pill_col,
        "prescription_col": prescription_col,
        "context_col": context_col,
    }, indent=2))

    # Standardize label column name for dataset usage across splits
    if val_label_col != label_col:
        val_df[label_col] = val_df[val_label_col]
    if test_label_col != label_col:
        test_df[label_col] = test_df[test_label_col]

    print("\n=== ROW COUNTS ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows  : {len(val_df)}")
    print(f"Test rows : {len(test_df)}")
    preflight_paths(train_df, pill_col, args.data_root, "Train pill images")
    preflight_paths(train_df, prescription_col, args.data_root, "Train prescription images")

    print("\n=== BASE M26 METRICS ===")
    base_rows, m26_val_report = read_m26_metrics(args.m26_dir, args.num_classes, true_col=args.true_col)
    if base_rows:
        print_metrics_table(base_rows)
    else:
        print("Could not load M26 metrics; continuing M30 training anyway.")

    train_labels = to_int_array(train_df[label_col], label_col)  # type: ignore[arg-type]
    class_weights_np, class_weight_detail = compute_class_weights(
        train_labels=train_labels,
        num_classes=args.num_classes,
        m26_val_report=m26_val_report,
        hard_f1_threshold=args.hard_f1_threshold,
        hard_boost=args.hard_boost,
        rare_boost=args.rare_boost,
        max_weight=args.max_class_weight,
    )
    class_weight_detail.to_csv(os.path.join(args.output_dir, "m30_class_weights.csv"), index=False)
    print("\n=== TOP 30 M30 HARD/RARE CLASS WEIGHTS ===")
    print(class_weight_detail.head(30).to_string(index=False))

    train_ds = PillContextDataset(train_df, label_col, pill_col, prescription_col, context_col, args.num_classes, args.image_size, args.data_root, train=True)
    val_ds = PillContextDataset(val_df, label_col, pill_col, prescription_col, context_col, args.num_classes, args.image_size, args.data_root, train=False)
    test_ds = PillContextDataset(test_df, label_col, pill_col, prescription_col, context_col, args.num_classes, args.image_size, args.data_root, train=False)

    sample_weights = class_weights_np[np.clip(train_ds.labels, 0, args.num_classes - 1)]
    sample_weights = np.where(sample_weights > 0, sample_weights, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = M30HardErrorModel(
        num_classes=args.num_classes,
        pill_backbone_name=args.pill_backbone,
        prescription_backbone_name=args.prescription_backbone,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        dropout=args.dropout,
        pretrained=True,
    )
    if args.init_checkpoint:
        load_partial_checkpoint(model, args.init_checkpoint)
    model.to(device)

    class_weights_t = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
    # Avoid zero loss weight for absent classes causing NaN in rare edge cases.
    class_weights_t = torch.where(class_weights_t > 0, class_weights_t, torch.ones_like(class_weights_t))
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)

    best_score = -1e9
    best_epoch = -1
    history: List[Dict[str, object]] = []
    best_path = os.path.join(args.output_dir, "m30_best.pth")
    last_path = os.path.join(args.output_dir, "m30_last.pth")

    print("\n=== TRAINING M30 ===")
    print(f"Epochs: {args.epochs} | batch_size: {args.batch_size} | lr={args.lr} | head_lr={args.head_lr}")
    for epoch in range(1, args.epochs + 1):
        trainable_backbone = epoch > args.freeze_backbone_epochs
        model.set_backbone_trainable(trainable_backbone)
        optimizer = make_optimizer(model, lr=args.lr, head_lr=args.head_lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        print(f"\nEpoch [{epoch}/{args.epochs}] | backbone_trainable={trainable_backbone}")
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp, args.log_every)
        val_loss, val_m, _, _, _, _ = evaluate(model, val_loader, criterion, device, amp, args.num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_m.accuracy,
            "train_macro_f1_present": train_m.macro_f1_present,
            "train_macro_f1_all": train_m.macro_f1_all,
            "train_weighted_f1": train_m.weighted_f1,
            "val_accuracy": val_m.accuracy,
            "val_macro_f1_present": val_m.macro_f1_present,
            "val_macro_f1_all": val_m.macro_f1_all,
            "val_weighted_f1": val_m.weighted_f1,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "m30_train_log.csv"), index=False)
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_m.accuracy:.4f} | Train Macro F1 Present: {train_m.macro_f1_present:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_m.accuracy:.4f} | Val   Macro F1 Present: {val_m.macro_f1_present:.4f} | Val Macro F1 All: {val_m.macro_f1_all:.4f}"
        )

        score = val_m.macro_f1_present
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "val_metrics": asdict(val_m),
            "class_weights": class_weights_np,
            "label_col": label_col,
            "pill_col": pill_col,
            "prescription_col": prescription_col,
            "context_col": context_col,
        }, last_path)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "val_metrics": asdict(val_m),
                "class_weights": class_weights_np,
                "label_col": label_col,
                "pill_col": pill_col,
                "prescription_col": prescription_col,
                "context_col": context_col,
            }, best_path)
            print(f"Saved new best: {best_path}")

    print(f"\nBest epoch by val macro_f1_present: {best_epoch} | score={best_score:.6f}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)

    print("\n=== FINAL EVALUATION WITH BEST M30 CHECKPOINT ===")
    val_loss, val_m, val_y, val_pred, val_probs, val_indices = evaluate(model, val_loader, criterion, device, amp, args.num_classes)
    test_loss, test_m, test_y, test_pred, test_probs, test_indices = evaluate(model, test_loader, criterion, device, amp, args.num_classes)
    final_rows = [metrics_row("val", "M30", val_m), metrics_row("test", "M30", test_m)]
    print_metrics_table(final_rows)

    with open(os.path.join(args.output_dir, "val_m30_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(val_m), f, indent=2)
    with open(os.path.join(args.output_dir, "test_m30_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(test_m), f, indent=2)
    pd.DataFrame(final_rows).to_csv(os.path.join(args.output_dir, "m30_summary.csv"), index=False)

    save_predictions(val_df, val_y, val_pred, val_probs, val_indices, os.path.join(args.output_dir, "val_m30_predictions.csv"), os.path.join(args.output_dir, "val_m30_probs.npy"))
    save_predictions(test_df, test_y, test_pred, test_probs, test_indices, os.path.join(args.output_dir, "test_m30_predictions.csv"), os.path.join(args.output_dir, "test_m30_probs.npy"))

    print("\nSaved files:")
    for fn in [
        "m30_config.json", "m30_class_weights.csv", "m30_train_log.csv", "m30_best.pth", "m30_last.pth",
        "val_m30_predictions.csv", "val_m30_probs.npy", "val_m30_metrics.json",
        "test_m30_predictions.csv", "test_m30_probs.npy", "test_m30_metrics.json", "m30_summary.csv",
    ]:
        print(os.path.join(args.output_dir, fn))
    print("\nDone.")


if __name__ == "__main__":
    main()

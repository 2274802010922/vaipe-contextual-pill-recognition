#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M29 - Class-wise logit bias calibration for VAIPE contextual pill recognition.

Goal:
- Start from the current best M26 probabilities.
- Tune only post-hoc calibration parameters on validation.
- Apply the selected parameters once to test.

This script does NOT retrain the neural network and does NOT tune on test.
It is designed to squeeze macro-F1 by correcting class imbalance and low-recall classes.

Example:
python run_m29_classwise_logit_bias_calibration.py \
  --m26_dir /content/drive/MyDrive/model/M26_calibrated_context_ensemble/run_v1 \
  --output_dir /content/drive/MyDrive/model/M29_classwise_logit_bias_calibration/run_v1 \
  --num_classes 108 \
  --true_col true_mapped_label
"""

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

EPS = 1e-12

TRUE_COL_CANDIDATES = [
    "true_mapped_label", "true_idx", "y_true", "target", "target_idx", "label_idx", "class_idx", "gt_idx",
    "true_label", "label", "class_id", "medicine_id", "pill_id", "mapped_label",
]

KEY_COL_CANDIDATES = [
    "sample_id", "row_id", "id", "image_name", "crop_name", "crop_path", "pill_crop_path", "pill_image_path",
]


@dataclass
class Metrics:
    accuracy: float
    macro_f1_present: float
    macro_f1_all: float
    weighted_f1: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_one_file(directory: str, patterns: List[str], required: bool = True) -> Optional[str]:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(directory, pat)))
    matches = sorted(set(matches), key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find file in {directory} with patterns: {patterns}")
    return None


def find_prediction_csv(directory: str, split: str) -> str:
    patterns = [
        f"{split}_m26_predictions.csv",
        f"{split}_M26_predictions.csv",
        f"{split}_m26*pred*.csv",
        f"{split}_M26*pred*.csv",
        f"*{split}*m26*pred*.csv",
        f"*{split}*M26*pred*.csv",
        f"{split}_predictions.csv",
        f"*{split}*pred*.csv",
    ]
    return find_one_file(directory, patterns, required=True)  # type: ignore[return-value]


def find_prob_npy(directory: str, split: str) -> str:
    patterns = [
        f"{split}_m26_probs.npy",
        f"{split}_M26_probs.npy",
        f"{split}_m26*prob*.npy",
        f"{split}_M26*prob*.npy",
        f"*{split}*m26*prob*.npy",
        f"*{split}*M26*prob*.npy",
        f"{split}_probs.npy",
        f"*{split}*prob*.npy",
    ]
    return find_one_file(directory, patterns, required=True)  # type: ignore[return-value]


def first_existing_col(df: pd.DataFrame, candidates: List[str], explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit column '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find true-label column. Use --true_col. Available columns: {list(df.columns)}")


def to_int_array(series: pd.Series, name: str) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().any():
        bad = series[vals.isna()].head(10).tolist()
        raise ValueError(f"Column '{name}' must be numeric class indices. Bad examples: {bad}")
    return vals.astype(int).to_numpy()


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"Probability array must be 2D, got shape={p.shape}")
    p = np.clip(p, EPS, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), EPS, None)


def softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), EPS, None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Metrics:
    labels_present = sorted(np.unique(y_true).tolist())
    labels_all = list(range(num_classes))
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1_present=float(f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)),
        macro_f1_all=float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels_present, average="weighted", zero_division=0)),
    )


def load_split(split: str, m26_dir: str, num_classes: int, true_col: Optional[str]) -> Dict[str, object]:
    csv_path = find_prediction_csv(m26_dir, split)
    prob_path = find_prob_npy(m26_dir, split)
    df = pd.read_csv(csv_path)
    probs = normalize_probs(np.load(prob_path))
    if len(df) != probs.shape[0]:
        raise ValueError(f"Length mismatch for {split}: len(df)={len(df)}, probs={probs.shape}")
    if probs.shape[1] != num_classes:
        raise ValueError(f"num_classes mismatch for {split}: expected {num_classes}, probs={probs.shape}")
    y_col = first_existing_col(df, TRUE_COL_CANDIDATES, true_col)
    y_true = to_int_array(df[y_col], y_col)
    return {"df": df, "probs": probs, "y_true": y_true, "csv_path": csv_path, "prob_path": prob_path, "y_col": y_col}


def class_stats(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, num_classes: int) -> pd.DataFrame:
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    pred_support = np.bincount(y_pred, minlength=num_classes)
    mean_prob_true = np.zeros(num_classes, dtype=np.float64)
    mean_rank_true = np.zeros(num_classes, dtype=np.float64)
    for c in labels:
        mask = y_true == c
        if mask.sum() > 0:
            mean_prob_true[c] = float(probs[mask, c].mean())
            # rank 1 means the true class has highest probability
            order = np.argsort(-probs[mask], axis=1)
            ranks = np.where(order == c)[1] + 1
            mean_rank_true[c] = float(ranks.mean())
        else:
            mean_prob_true[c] = 0.0
            mean_rank_true[c] = 0.0
    return pd.DataFrame({
        "class_id": labels,
        "support": support.astype(int),
        "pred_support": pred_support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_true_prob": mean_prob_true,
        "mean_true_rank": mean_rank_true,
    })


def make_bias_vector(
    stats: pd.DataFrame,
    num_classes: int,
    alpha_prior: float,
    beta_recall: float,
    gamma_precision_penalty: float,
    delta_false_positive_penalty: float,
    support_smoothing: float,
    bias_clip: float,
) -> np.ndarray:
    support = stats["support"].to_numpy(dtype=np.float64)
    pred_support = stats["pred_support"].to_numpy(dtype=np.float64)
    precision = stats["precision"].to_numpy(dtype=np.float64)
    recall = stats["recall"].to_numpy(dtype=np.float64)

    present = support > 0
    prior = (support + support_smoothing) / (support.sum() + support_smoothing * num_classes)
    prior_mean = 1.0 / num_classes

    # Rare-class boost. Rare classes get positive bias, frequent classes get negative bias.
    prior_bias = -np.log(np.clip(prior / prior_mean, EPS, None))

    # Low-recall boost. Classes that exist but are missed often get positive bias.
    recall_gap = np.where(present, 1.0 - recall, 0.0)

    # Low-precision penalty. Prevent classes that already attract many false positives from exploding.
    precision_gap = np.where(pred_support > 0, 1.0 - precision, 0.0)

    # Extra penalty for over-predicted classes relative to true support.
    fp_pressure = np.log1p(np.maximum(pred_support - support, 0.0))
    fp_pressure = fp_pressure / np.clip(fp_pressure.max(), 1.0, None)

    bias = (
        alpha_prior * prior_bias
        + beta_recall * recall_gap
        - gamma_precision_penalty * precision_gap
        - delta_false_positive_penalty * fp_pressure
    )

    # Never aggressively boost absent classes from validation.
    bias = np.where(present, bias, np.minimum(bias, 0.0))
    bias = np.clip(bias, -bias_clip, bias_clip)
    bias = bias - bias.mean()
    return bias.astype(np.float64)


def apply_calibration(probs: np.ndarray, temperature: float, bias: np.ndarray, topk: int = 0) -> np.ndarray:
    p = normalize_probs(probs)
    logits = np.log(np.clip(p, EPS, None)) / float(temperature)
    if topk and topk > 0 and topk < p.shape[1]:
        # Only allow bias to alter classes that were already plausible for the sample.
        idx = np.argpartition(p, -topk, axis=1)[:, -topk:]
        mask = np.zeros_like(p, dtype=bool)
        rows = np.arange(p.shape[0])[:, None]
        mask[rows, idx] = True
        adjusted = logits.copy()
        adjusted[mask] += bias.reshape(1, -1).repeat(p.shape[0], axis=0)[mask]
        return softmax_from_logits(adjusted)
    return softmax_from_logits(logits + bias.reshape(1, -1))


def search_configs(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    num_classes: int,
    max_accuracy_drop: float,
    selection_metric: str,
) -> Tuple[pd.DataFrame, Dict[str, object], np.ndarray]:
    base_pred = probs_val.argmax(axis=1)
    base_metrics = compute_metrics(y_val, base_pred, num_classes)
    stats = class_stats(y_val, base_pred, probs_val, num_classes)

    temperatures = [0.65, 0.75, 0.85, 1.00, 1.15, 1.30]
    alpha_priors = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]
    beta_recalls = [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    gamma_precisions = [0.00, 0.10, 0.20, 0.35, 0.50]
    delta_fps = [0.00, 0.10, 0.25, 0.40]
    support_smoothings = [1.0, 3.0, 5.0]
    bias_clips = [0.50, 0.75, 1.00, 1.50, 2.00]
    topks = [0, 20, 50]

    rows: List[Dict[str, object]] = []
    best_bias: Optional[np.ndarray] = None
    best_score = -1e18
    best_row: Optional[Dict[str, object]] = None

    total = 0
    for temp in temperatures:
        for alpha in alpha_priors:
            for beta in beta_recalls:
                for gamma in gamma_precisions:
                    for delta in delta_fps:
                        # Avoid too many redundant all-zero configs.
                        if alpha == 0 and beta == 0 and gamma == 0 and delta == 0:
                            ss_list = [3.0]
                            bc_list = [1.0]
                        else:
                            ss_list = support_smoothings
                            bc_list = bias_clips
                        for ss in ss_list:
                            for bc in bc_list:
                                bias = make_bias_vector(
                                    stats, num_classes,
                                    alpha_prior=alpha,
                                    beta_recall=beta,
                                    gamma_precision_penalty=gamma,
                                    delta_false_positive_penalty=delta,
                                    support_smoothing=ss,
                                    bias_clip=bc,
                                )
                                for topk in topks:
                                    total += 1
                                    calibrated = apply_calibration(probs_val, temp, bias, topk=topk)
                                    pred = calibrated.argmax(axis=1)
                                    m = compute_metrics(y_val, pred, num_classes)
                                    changed = int((pred != base_pred).sum())
                                    acc_drop = base_metrics.accuracy - m.accuracy
                                    # Conservative selection score: optimize macro-F1 but penalize unsafe accuracy loss.
                                    metric_value = getattr(m, selection_metric)
                                    penalty = max(0.0, acc_drop - max_accuracy_drop) * 5.0
                                    score = metric_value - penalty
                                    row = {
                                        "temperature": temp,
                                        "alpha_prior": alpha,
                                        "beta_recall": beta,
                                        "gamma_precision_penalty": gamma,
                                        "delta_false_positive_penalty": delta,
                                        "support_smoothing": ss,
                                        "bias_clip": bc,
                                        "topk": topk,
                                        "changed_from_m26": changed,
                                        "selection_score": score,
                                        **asdict(m),
                                    }
                                    rows.append(row)
                                    if score > best_score:
                                        best_score = score
                                        best_row = row
                                        best_bias = bias.copy()

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        by=["selection_score", selection_metric, "macro_f1_all", "accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)
    if best_row is None or best_bias is None:
        raise RuntimeError("No config was evaluated.")
    selected = dict(grid.iloc[0].to_dict())
    selected["selection_metric"] = selection_metric
    selected["max_accuracy_drop"] = max_accuracy_drop
    selected["num_evaluated_configs"] = int(total)
    selected["base_val_accuracy"] = base_metrics.accuracy
    selected["base_val_macro_f1_present"] = base_metrics.macro_f1_present
    selected["base_val_macro_f1_all"] = base_metrics.macro_f1_all
    return grid, selected, best_bias


def change_diagnostics(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> Dict[str, object]:
    changed = new_pred != base_pred
    base_ok = base_pred == y_true
    new_ok = new_pred == y_true
    return {
        "changed_from_m26": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "changed_helped": int((changed & ~base_ok & new_ok).sum()),
        "changed_hurt": int((changed & base_ok & ~new_ok).sum()),
        "changed_neutral": int((changed & (base_ok == new_ok)).sum()),
    }


def save_split_outputs(
    split: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    num_classes: int,
) -> Metrics:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out_df = df.copy()
    out_df["m29_pred_mapped_label"] = pred.astype(int)
    out_df["m29_confidence"] = conf.astype(float)
    out_df["m29_is_correct"] = (pred == y_true).astype(int)
    out_csv = os.path.join(output_dir, f"{split}_m29_predictions.csv")
    out_npy = os.path.join(output_dir, f"{split}_m29_probs.npy")
    out_df.to_csv(out_csv, index=False)
    np.save(out_npy, probs.astype(np.float32))
    m = compute_metrics(y_true, pred, num_classes)
    with open(os.path.join(output_dir, f"{split}_m29_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, indent=2, ensure_ascii=False)
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m26_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=108)
    parser.add_argument("--true_col", type=str, default=None)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.005)
    parser.add_argument("--selection_metric", type=str, default="macro_f1_present", choices=["macro_f1_present", "macro_f1_all", "weighted_f1", "accuracy"])
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("=== M29 CLASS-WISE LOGIT BIAS CALIBRATION ===")
    print(f"M26 dir    : {args.m26_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Num classes: {args.num_classes}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Max accuracy drop allowed before penalty: {args.max_accuracy_drop}")

    val = load_split("val", args.m26_dir, args.num_classes, args.true_col)
    test = load_split("test", args.m26_dir, args.num_classes, args.true_col)

    print("\n=== INPUT FILES ===")
    print(json.dumps({
        "val": {"csv": val["csv_path"], "probs": val["prob_path"], "true_col": val["y_col"]},
        "test": {"csv": test["csv_path"], "probs": test["prob_path"], "true_col": test["y_col"]},
    }, indent=2, ensure_ascii=False))

    y_val = val["y_true"]  # type: ignore[assignment]
    p_val = val["probs"]  # type: ignore[assignment]
    y_test = test["y_true"]  # type: ignore[assignment]
    p_test = test["probs"]  # type: ignore[assignment]

    base_val_pred = p_val.argmax(axis=1)
    base_test_pred = p_test.argmax(axis=1)
    base_val_metrics = compute_metrics(y_val, base_val_pred, args.num_classes)
    base_test_metrics = compute_metrics(y_test, base_test_pred, args.num_classes)

    base_metrics = pd.DataFrame([
        {"split": "val", "model": "M26", **asdict(base_val_metrics)},
        {"split": "test", "model": "M26", **asdict(base_test_metrics)},
    ])
    print("\n=== BASE M26 METRICS ===")
    print(base_metrics.to_string(index=False))
    base_metrics.to_csv(os.path.join(args.output_dir, "m29_base_m26_metrics.csv"), index=False)

    val_stats = class_stats(y_val, base_val_pred, p_val, args.num_classes)
    val_stats = val_stats.sort_values(["f1", "support"], ascending=[True, False]).reset_index(drop=True)
    val_stats.to_csv(os.path.join(args.output_dir, "m29_val_classwise_m26_stats.csv"), index=False)
    print("\n=== LOWEST 25 M26 VAL CLASS F1 ===")
    print(val_stats.head(25).to_string(index=False))

    print("\nSearching M29 calibration configs on validation only...")
    grid, selected, bias = search_configs(
        y_val=y_val,
        probs_val=p_val,
        num_classes=args.num_classes,
        max_accuracy_drop=args.max_accuracy_drop,
        selection_metric=args.selection_metric,
    )
    grid.to_csv(os.path.join(args.output_dir, "m29_val_grid_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "m29_best_config.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.output_dir, "m29_class_bias.npy"), bias.astype(np.float32))

    print("\n=== TOP 20 M29 VAL CONFIGS ===")
    show_cols = [
        "temperature", "alpha_prior", "beta_recall", "gamma_precision_penalty",
        "delta_false_positive_penalty", "support_smoothing", "bias_clip", "topk",
        "changed_from_m26", "selection_score", "accuracy", "macro_f1_present", "macro_f1_all", "weighted_f1",
    ]
    print(grid[show_cols].head(20).to_string(index=False))

    print("\n=== SELECTED M29 CONFIG BY VALIDATION ===")
    print(json.dumps(selected, indent=2, ensure_ascii=False))

    temp = float(selected["temperature"])
    topk = int(selected["topk"])
    p29_val = apply_calibration(p_val, temp, bias, topk=topk)
    p29_test = apply_calibration(p_test, temp, bias, topk=topk)

    val_m29_metrics = save_split_outputs("val", val["df"], y_val, p29_val, args.output_dir, args.num_classes)  # type: ignore[arg-type]
    test_m29_metrics = save_split_outputs("test", test["df"], y_test, p29_test, args.output_dir, args.num_classes)  # type: ignore[arg-type]

    summary = pd.DataFrame([
        {"split": "val", "model": "M26", **asdict(base_val_metrics)},
        {"split": "val", "model": "M29", **asdict(val_m29_metrics)},
        {"split": "test", "model": "M26", **asdict(base_test_metrics)},
        {"split": "test", "model": "M29", **asdict(test_m29_metrics)},
    ])
    summary.to_csv(os.path.join(args.output_dir, "m29_summary.csv"), index=False)

    print("\n=== M29 FINAL RESULTS ===")
    print(summary.to_string(index=False))

    print("\n=== VAL M29 CHANGE DIAGNOSTICS VS M26 ===")
    print(json.dumps(change_diagnostics(y_val, base_val_pred, p29_val.argmax(axis=1)), indent=2))

    print("\n=== TEST M29 CHANGE DIAGNOSTICS VS M26 ===")
    print(json.dumps(change_diagnostics(y_test, base_test_pred, p29_test.argmax(axis=1)), indent=2))

    print("\nSaved files:")
    for name in [
        "m29_base_m26_metrics.csv",
        "m29_val_classwise_m26_stats.csv",
        "m29_val_grid_results.csv",
        "m29_best_config.json",
        "m29_class_bias.npy",
        "val_m29_predictions.csv",
        "val_m29_probs.npy",
        "val_m29_metrics.json",
        "test_m29_predictions.csv",
        "test_m29_probs.npy",
        "test_m29_metrics.json",
        "m29_summary.csv",
    ]:
        print(os.path.join(args.output_dir, name))
    print("\nDone.")


if __name__ == "__main__":
    main()

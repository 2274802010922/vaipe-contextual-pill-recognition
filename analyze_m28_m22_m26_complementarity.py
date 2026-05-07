
import argparse
import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

EPS = 1e-12


TRUE_COL_CANDIDATES = [
    "true_idx", "y_true", "target", "target_idx", "label_idx", "class_idx", "gt_idx",
    "true_label", "label", "class_id", "medicine_id", "pill_id",
]

PRED_COL_CANDIDATES = [
    "pred_idx_final", "pred_idx", "pred_label_final", "pred_label", "prediction",
    "pred", "y_pred", "m22_pred", "m26_pred",
]

CONF_COL_CANDIDATES = [
    "conf_final", "confidence", "score", "prob", "max_prob", "pred_conf",
]

KEY_COL_CANDIDATES = [
    "sample_id", "row_id", "id", "image_name", "crop_name", "crop_path", "pill_image_path",
]


@dataclass
class Metrics:
    accuracy: float
    macro_f1_present: float
    macro_f1_all: float
    weighted_f1: float


@dataclass
class BestConfig:
    mode: str
    score_key: str
    weight_m26: float
    temp_m22: float
    temp_m26: float
    rel_power: float
    smoothing: float
    min_support: int
    rel_floor: float
    conf_gate: float = -1.0
    margin_gate: float = -1.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_one_file(directory: str, patterns: List[str], required: bool = True) -> Optional[str]:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(directory, pat)))
    # unique, stable order, prefer shorter names
    matches = sorted(set(matches), key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find file in {directory} with patterns: {patterns}")
    return None


def find_prediction_csv(directory: str, split: str, model_key: str) -> str:
    mk = model_key.lower()
    MK = model_key.upper()
    patterns = [
        f"{split}_{mk}_predictions.csv",
        f"{split}_{MK}_predictions.csv",
        f"{split}_{mk}_prediction.csv",
        f"{split}_{MK}_prediction.csv",
        f"{split}_{mk}*pred*.csv",
        f"{split}_{MK}*pred*.csv",
        f"*{split}*{mk}*pred*.csv",
        f"*{split}*{MK}*pred*.csv",
        f"{split}_predictions.csv",
        f"*{split}*pred*.csv",
        f"*{split}*prediction*.csv",
    ]
    return find_one_file(directory, patterns, required=True)


def find_prob_npy(directory: str, split: str, model_key: str) -> str:
    mk = model_key.lower()
    MK = model_key.upper()
    patterns = [
        f"{split}_{mk}_probs.npy",
        f"{split}_{MK}_probs.npy",
        f"{split}_{mk}_prob.npy",
        f"{split}_{MK}_prob.npy",
        f"{split}_{mk}*prob*.npy",
        f"{split}_{MK}*prob*.npy",
        f"*{split}*{mk}*prob*.npy",
        f"*{split}*{MK}*prob*.npy",
        f"{split}_probs.npy",
        f"*{split}*prob*.npy",
    ]
    return find_one_file(directory, patterns, required=True)


def first_existing_col(df: pd.DataFrame, candidates: List[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit column '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_int_array(series: pd.Series, name: str) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().any():
        bad = series[vals.isna()].head(10).tolist()
        raise ValueError(
            f"Column '{name}' must be numeric class indices for this script. "
            f"Bad examples: {bad}"
        )
    return vals.astype(int).to_numpy()


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"Probability array must be 2D, got shape={p.shape}")
    p = np.clip(p, EPS, None)
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum = np.clip(row_sum, EPS, None)
    return p / row_sum


def temp_scale_probs(p: np.ndarray, temp: float) -> np.ndarray:
    p = normalize_probs(p)
    temp = float(temp)
    if abs(temp - 1.0) < 1e-9:
        return p
    logp = np.log(np.clip(p, EPS, None)) / temp
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return normalize_probs(out)


def confidence_and_margin(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    part = np.partition(p, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2] if p.shape[1] >= 2 else np.zeros_like(top1)
    return top1, top1 - top2


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Metrics:
    labels_present = sorted(np.unique(y_true).tolist())
    labels_all = list(range(num_classes))
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1_present=float(f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)),
        macro_f1_all=float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels_present, average="weighted", zero_division=0)),
    )


def metrics_dict(prefix: str, m: Metrics) -> Dict[str, float]:
    return {
        f"{prefix}_accuracy": m.accuracy,
        f"{prefix}_macro_f1_present": m.macro_f1_present,
        f"{prefix}_macro_f1_all": m.macro_f1_all,
        f"{prefix}_weighted_f1": m.weighted_f1,
    }


def model_reliability_by_predicted_class(
    y_true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    smoothing: float = 3.0,
    min_support: int = 5,
    rel_floor: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.zeros(num_classes, dtype=np.float64)
    correct = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = pred == c
        support[c] = mask.sum()
        if support[c] > 0:
            correct[c] = (y_true[mask] == pred[mask]).sum()

    global_acc = float((y_true == pred).mean())
    raw_precision = np.divide(correct, np.maximum(support, 1.0))
    smoothed = (correct + smoothing * global_acc) / np.maximum(support + smoothing, EPS)

    rel = smoothed.copy()
    rel[support < min_support] = global_acc
    rel = np.clip(rel, rel_floor, 1.0)
    return rel, support, raw_precision


def load_split(
    split: str,
    m22_dir: str,
    m26_dir: str,
    num_classes: int,
    true_col: Optional[str] = None,
) -> Dict[str, object]:
    m22_csv = find_prediction_csv(m22_dir, split, "m22")
    m26_csv = find_prediction_csv(m26_dir, split, "m26")
    m22_prob_path = find_prob_npy(m22_dir, split, "m22")
    m26_prob_path = find_prob_npy(m26_dir, split, "m26")

    df22 = pd.read_csv(m22_csv)
    df26 = pd.read_csv(m26_csv)
    p22 = normalize_probs(np.load(m22_prob_path))
    p26 = normalize_probs(np.load(m26_prob_path))

    if len(df22) != len(df26) or len(df22) != p22.shape[0] or len(df26) != p26.shape[0]:
        raise ValueError(
            f"Length mismatch for {split}: len(df22)={len(df22)}, len(df26)={len(df26)}, "
            f"p22={p22.shape}, p26={p26.shape}"
        )
    if p22.shape[1] != num_classes or p26.shape[1] != num_classes:
        raise ValueError(
            f"num_classes mismatch for {split}: expected {num_classes}, p22={p22.shape}, p26={p26.shape}"
        )

    # Basic key check. We keep row-order alignment because M22/M26 outputs are usually generated from the same split.
    common_keys = [c for c in KEY_COL_CANDIDATES if c in df22.columns and c in df26.columns]
    for c in common_keys:
        if not (df22[c].astype(str).to_numpy() == df26[c].astype(str).to_numpy()).all():
            raise ValueError(
                f"Row alignment mismatch on key column '{c}' for split={split}. "
                "Regenerate M22 and M26 predictions on the same split/order, or add a stable row_id."
            )

    y_col22 = first_existing_col(df22, TRUE_COL_CANDIDATES, true_col)
    y_col26 = first_existing_col(df26, TRUE_COL_CANDIDATES, true_col if true_col and true_col in df26.columns else None)
    if y_col22 is None and y_col26 is None:
        raise ValueError(
            f"Cannot find true-label column in either CSV for split={split}. "
            f"Use --true_col. df22 columns={list(df22.columns)}, df26 columns={list(df26.columns)}"
        )
    if y_col22 is not None:
        y_true = to_int_array(df22[y_col22], y_col22)
    else:
        y_true = to_int_array(df26[y_col26], y_col26)  # type: ignore[index]

    if y_col26 is not None:
        y_true26 = to_int_array(df26[y_col26], y_col26)
        if not np.array_equal(y_true, y_true26):
            raise ValueError(f"True labels differ between M22 and M26 CSV for split={split}.")

    pred22 = p22.argmax(axis=1)
    pred26 = p26.argmax(axis=1)

    return {
        "df22": df22,
        "df26": df26,
        "p22": p22,
        "p26": p26,
        "y_true": y_true,
        "pred22": pred22,
        "pred26": pred26,
        "files": {
            "m22_csv": m22_csv,
            "m26_csv": m26_csv,
            "m22_probs": m22_prob_path,
            "m26_probs": m26_prob_path,
        },
    }


def complementarity_table(y: np.ndarray, pred22: np.ndarray, pred26: np.ndarray, num_classes: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    m22_ok = pred22 == y
    m26_ok = pred26 == y
    both_ok = m22_ok & m26_ok
    m22_only = m22_ok & ~m26_ok
    m26_only = ~m22_ok & m26_ok
    both_wrong = ~m22_ok & ~m26_ok

    # Diagnostic oracle: not used for selecting final config.
    oracle_pred = pred26.copy()
    oracle_pred[m22_only] = pred22[m22_only]
    oracle_metrics = compute_metrics(y, oracle_pred, num_classes)

    rows = []
    for name, mask in [
        ("both_correct", both_ok),
        ("m22_correct_m26_wrong", m22_only),
        ("m26_correct_m22_wrong", m26_only),
        ("both_wrong", both_wrong),
    ]:
        rows.append({"case": name, "count": int(mask.sum()), "ratio": float(mask.mean())})
    df = pd.DataFrame(rows)
    extra = {
        "oracle_accuracy": oracle_metrics.accuracy,
        "oracle_macro_f1_present": oracle_metrics.macro_f1_present,
        "oracle_macro_f1_all": oracle_metrics.macro_f1_all,
        "oracle_weighted_f1": oracle_metrics.weighted_f1,
    }
    return df, extra


def apply_reliability_blend(
    p22: np.ndarray,
    p26: np.ndarray,
    rel22: np.ndarray,
    rel26: np.ndarray,
    weight_m26: float,
    temp_m22: float,
    temp_m26: float,
    rel_power: float,
) -> np.ndarray:
    q22 = temp_scale_probs(p22, temp_m22)
    q26 = temp_scale_probs(p26, temp_m26)
    if rel_power > 0:
        r22 = np.power(np.clip(rel22, EPS, 1.0), rel_power)
        r26 = np.power(np.clip(rel26, EPS, 1.0), rel_power)
        q22 = q22 * r22.reshape(1, -1)
        q26 = q26 * r26.reshape(1, -1)
    out = weight_m26 * q26 + (1.0 - weight_m26) * q22
    return normalize_probs(out)


def apply_confidence_gate(
    p22: np.ndarray,
    p26: np.ndarray,
    conf_gate: float,
    margin_gate: float,
) -> np.ndarray:
    # Start from M26. Switch to M22 when M26 is uncertain and M22 is more confident.
    q22 = normalize_probs(p22)
    q26 = normalize_probs(p26)
    c22, m22 = confidence_and_margin(q22)
    c26, m26 = confidence_and_margin(q26)
    switch = (c26 <= conf_gate) & ((c22 - c26) >= margin_gate) & (m22 >= m26)
    out = q26.copy()
    out[switch] = q22[switch]
    return normalize_probs(out)


def search_best_config(
    y_val: np.ndarray,
    p22_val: np.ndarray,
    p26_val: np.ndarray,
    num_classes: int,
    output_dir: str,
) -> Tuple[BestConfig, pd.DataFrame, Dict[str, np.ndarray]]:
    pred22_val = p22_val.argmax(axis=1)
    pred26_val = p26_val.argmax(axis=1)

    rows: List[Dict[str, object]] = []
    rel_cache: Dict[Tuple[float, int, float], Tuple[np.ndarray, np.ndarray]] = {}

    # Stage A: plain temperature + global blend.
    temps = [0.70, 0.85, 1.00, 1.20, 1.50]
    weights = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]
    for t22 in temps:
        q22 = temp_scale_probs(p22_val, t22)
        for t26 in temps:
            q26 = temp_scale_probs(p26_val, t26)
            for w in weights:
                probs = normalize_probs(w * q26 + (1.0 - w) * q22)
                pred = probs.argmax(axis=1)
                m = compute_metrics(y_val, pred, num_classes)
                rows.append({
                    "mode": "global_blend",
                    "weight_m26": w,
                    "temp_m22": t22,
                    "temp_m26": t26,
                    "rel_power": 0.0,
                    "smoothing": 0.0,
                    "min_support": 0,
                    "rel_floor": 0.0,
                    "conf_gate": -1.0,
                    "margin_gate": -1.0,
                    **asdict(m),
                })

    # Stage B: reliability-calibrated blend by predicted-class reliability from validation.
    rel_powers = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    smoothings = [1.0, 3.0, 5.0, 10.0]
    min_supports = [1, 3, 5, 10]
    rel_floors = [0.03, 0.05, 0.10]
    reliab_temps = [0.85, 1.00, 1.20]
    reliab_weights = [round(float(x), 2) for x in np.linspace(0.20, 0.95, 16)]

    for smoothing in smoothings:
        for min_support in min_supports:
            for rel_floor in rel_floors:
                key = (smoothing, min_support, rel_floor)
                rel22, _, _ = model_reliability_by_predicted_class(
                    y_val, pred22_val, num_classes, smoothing=smoothing, min_support=min_support, rel_floor=rel_floor
                )
                rel26, _, _ = model_reliability_by_predicted_class(
                    y_val, pred26_val, num_classes, smoothing=smoothing, min_support=min_support, rel_floor=rel_floor
                )
                rel_cache[key] = (rel22, rel26)
                for rp in rel_powers:
                    for t22 in reliab_temps:
                        for t26 in reliab_temps:
                            for w in reliab_weights:
                                probs = apply_reliability_blend(
                                    p22_val, p26_val, rel22, rel26,
                                    weight_m26=w, temp_m22=t22, temp_m26=t26, rel_power=rp
                                )
                                pred = probs.argmax(axis=1)
                                m = compute_metrics(y_val, pred, num_classes)
                                rows.append({
                                    "mode": "reliability_blend",
                                    "weight_m26": w,
                                    "temp_m22": t22,
                                    "temp_m26": t26,
                                    "rel_power": rp,
                                    "smoothing": smoothing,
                                    "min_support": min_support,
                                    "rel_floor": rel_floor,
                                    "conf_gate": -1.0,
                                    "margin_gate": -1.0,
                                    **asdict(m),
                                })

    # Stage C: confidence gate. This is kept as a baseline, but selected only if val proves it.
    conf_gates = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    margin_gates = [-0.20, -0.10, 0.00, 0.05, 0.10, 0.15, 0.20]
    for cg in conf_gates:
        for mg in margin_gates:
            probs = apply_confidence_gate(p22_val, p26_val, conf_gate=cg, margin_gate=mg)
            pred = probs.argmax(axis=1)
            m = compute_metrics(y_val, pred, num_classes)
            rows.append({
                "mode": "confidence_gate",
                "weight_m26": 1.0,
                "temp_m22": 1.0,
                "temp_m26": 1.0,
                "rel_power": 0.0,
                "smoothing": 0.0,
                "min_support": 0,
                "rel_floor": 0.0,
                "conf_gate": cg,
                "margin_gate": mg,
                **asdict(m),
            })

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        by=["macro_f1_present", "macro_f1_all", "accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)
    grid.to_csv(os.path.join(output_dir, "m28_val_grid_results.csv"), index=False)

    best_row = grid.iloc[0].to_dict()
    best = BestConfig(
        mode=str(best_row["mode"]),
        score_key="macro_f1_present",
        weight_m26=float(best_row["weight_m26"]),
        temp_m22=float(best_row["temp_m22"]),
        temp_m26=float(best_row["temp_m26"]),
        rel_power=float(best_row["rel_power"]),
        smoothing=float(best_row["smoothing"]),
        min_support=int(best_row["min_support"]),
        rel_floor=float(best_row["rel_floor"]),
        conf_gate=float(best_row["conf_gate"]),
        margin_gate=float(best_row["margin_gate"]),
    )

    # Store reliability from the selected config. For non-reliability modes, still store default reliability for diagnostics.
    if best.mode == "reliability_blend":
        key = (best.smoothing, best.min_support, best.rel_floor)
        rel22, rel26 = rel_cache[key]
    else:
        rel22, _, _ = model_reliability_by_predicted_class(y_val, pred22_val, num_classes, smoothing=3.0, min_support=5, rel_floor=0.05)
        rel26, _, _ = model_reliability_by_predicted_class(y_val, pred26_val, num_classes, smoothing=3.0, min_support=5, rel_floor=0.05)

    return best, grid, {"rel22": rel22, "rel26": rel26}


def apply_best_config(p22: np.ndarray, p26: np.ndarray, best: BestConfig, rel22: np.ndarray, rel26: np.ndarray) -> np.ndarray:
    if best.mode in ["global_blend", "reliability_blend"]:
        return apply_reliability_blend(
            p22, p26, rel22, rel26,
            weight_m26=best.weight_m26,
            temp_m22=best.temp_m22,
            temp_m26=best.temp_m26,
            rel_power=best.rel_power if best.mode == "reliability_blend" else 0.0,
        )
    if best.mode == "confidence_gate":
        return apply_confidence_gate(p22, p26, conf_gate=best.conf_gate, margin_gate=best.margin_gate)
    raise ValueError(f"Unknown mode: {best.mode}")


def save_predictions(
    split: str,
    output_dir: str,
    base_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred22: np.ndarray,
    pred26: np.ndarray,
    num_classes: int,
) -> Metrics:
    pred = probs.argmax(axis=1)
    conf, margin = confidence_and_margin(probs)
    m22_ok = pred22 == y_true
    m26_ok = pred26 == y_true
    m28_ok = pred == y_true

    out = base_df.copy()
    out["true_idx_m28_eval"] = y_true
    out["m22_pred_idx"] = pred22
    out["m26_pred_idx"] = pred26
    out["m28_pred_idx"] = pred
    out["m28_conf"] = conf
    out["m28_margin"] = margin
    out["m22_correct"] = m22_ok.astype(int)
    out["m26_correct"] = m26_ok.astype(int)
    out["m28_correct"] = m28_ok.astype(int)
    out["m28_changed_from_m26"] = (pred != pred26).astype(int)
    out["m28_changed_from_m22"] = (pred != pred22).astype(int)

    metrics = compute_metrics(y_true, pred, num_classes)
    out_csv = os.path.join(output_dir, f"{split}_m28_predictions.csv")
    out_npy = os.path.join(output_dir, f"{split}_m28_probs.npy")
    out_json = os.path.join(output_dir, f"{split}_m28_metrics.json")
    out.to_csv(out_csv, index=False)
    np.save(out_npy, probs.astype(np.float32))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
    return metrics


def print_metrics_table(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n=== {title} ===")
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m22_dir", required=True)
    parser.add_argument("--m26_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=108)
    parser.add_argument("--true_col", default=None, help="Optional explicit true-label column name.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("=== M28 ERROR-DRIVEN CALIBRATED SOFT ENSEMBLE ===")
    print(f"M22 dir    : {args.m22_dir}")
    print(f"M26 dir    : {args.m26_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Num classes: {args.num_classes}")

    val = load_split("val", args.m22_dir, args.m26_dir, args.num_classes, true_col=args.true_col)
    test = load_split("test", args.m22_dir, args.m26_dir, args.num_classes, true_col=args.true_col)

    print("\n=== INPUT FILES ===")
    print(json.dumps({"val": val["files"], "test": test["files"]}, indent=2, ensure_ascii=False))

    # Base metrics
    base_rows = []
    for split_name, data in [("val", val), ("test", test)]:
        y = data["y_true"]
        pred22 = data["pred22"]
        pred26 = data["pred26"]
        m22 = compute_metrics(y, pred22, args.num_classes)
        m26 = compute_metrics(y, pred26, args.num_classes)
        base_rows.append({"split": split_name, "model": "M22", **asdict(m22)})
        base_rows.append({"split": split_name, "model": "M26", **asdict(m26)})
    print_metrics_table("BASE METRICS", base_rows)
    pd.DataFrame(base_rows).to_csv(os.path.join(args.output_dir, "m28_base_metrics.csv"), index=False)

    # Complementarity diagnostics
    for split_name, data in [("val", val), ("test", test)]:
        comp_df, oracle = complementarity_table(
            data["y_true"], data["pred22"], data["pred26"], args.num_classes
        )
        comp_df.to_csv(os.path.join(args.output_dir, f"{split_name}_m22_m26_complementarity.csv"), index=False)
        with open(os.path.join(args.output_dir, f"{split_name}_m22_m26_oracle_upper_bound.json"), "w", encoding="utf-8") as f:
            json.dump(oracle, f, indent=2, ensure_ascii=False)
        print(f"\n=== {split_name.upper()} M22/M26 COMPLEMENTARITY ===")
        print(comp_df.to_string(index=False))
        print("Oracle upper bound if we could choose the correct model per row, diagnostic only:")
        print(json.dumps(oracle, indent=2))

    print("\nSearching M28 configs on validation only...")
    best, grid, rels = search_best_config(
        val["y_true"], val["p22"], val["p26"], args.num_classes, args.output_dir
    )
    print("\n=== TOP 20 M28 VAL CONFIGS ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(grid.head(20).to_string(index=False))

    with open(os.path.join(args.output_dir, "m28_best_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(best), f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.output_dir, "m28_selected_rel22.npy"), rels["rel22"].astype(np.float32))
    np.save(os.path.join(args.output_dir, "m28_selected_rel26.npy"), rels["rel26"].astype(np.float32))

    print("\n=== SELECTED M28 CONFIG BY VALIDATION ===")
    print(json.dumps(asdict(best), indent=2, ensure_ascii=False))

    val_probs = apply_best_config(val["p22"], val["p26"], best, rels["rel22"], rels["rel26"])
    test_probs = apply_best_config(test["p22"], test["p26"], best, rels["rel22"], rels["rel26"])

    val_metrics = save_predictions(
        "val", args.output_dir, val["df26"], val["y_true"], val_probs, val["pred22"], val["pred26"], args.num_classes
    )
    test_metrics = save_predictions(
        "test", args.output_dir, test["df26"], test["y_true"], test_probs, test["pred22"], test["pred26"], args.num_classes
    )

    m28_rows = [
        {"split": "val", "model": "M28", **asdict(val_metrics)},
        {"split": "test", "model": "M28", **asdict(test_metrics)},
    ]
    print_metrics_table("M28 FINAL RESULTS", m28_rows)

    # Combined summary: base + M28
    summary = pd.concat([pd.DataFrame(base_rows), pd.DataFrame(m28_rows)], ignore_index=True)
    summary.to_csv(os.path.join(args.output_dir, "m28_summary.csv"), index=False)

    # Extra diagnostics: how many final predictions changed from M26.
    for split_name, data, probs in [("val", val, val_probs), ("test", test, test_probs)]:
        pred = probs.argmax(axis=1)
        changed = pred != data["pred26"]
        better = changed & (pred == data["y_true"]) & (data["pred26"] != data["y_true"])
        worse = changed & (pred != data["y_true"]) & (data["pred26"] == data["y_true"])
        neutral = changed & ~(better | worse)
        diag = {
            "changed_from_m26": int(changed.sum()),
            "changed_ratio": float(changed.mean()),
            "changed_helped": int(better.sum()),
            "changed_hurt": int(worse.sum()),
            "changed_neutral": int(neutral.sum()),
        }
        with open(os.path.join(args.output_dir, f"{split_name}_m28_change_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        print(f"\n=== {split_name.upper()} M28 CHANGE DIAGNOSTICS VS M26 ===")
        print(json.dumps(diag, indent=2))

    print("\nSaved files:")
    for fn in [
        "m28_base_metrics.csv",
        "m28_val_grid_results.csv",
        "m28_best_config.json",
        "val_m28_predictions.csv",
        "val_m28_probs.npy",
        "val_m28_metrics.json",
        "test_m28_predictions.csv",
        "test_m28_probs.npy",
        "test_m28_metrics.json",
        "m28_summary.csv",
    ]:
        print(os.path.join(args.output_dir, fn))
    print("\nDone.")


if __name__ == "__main__":
    main()

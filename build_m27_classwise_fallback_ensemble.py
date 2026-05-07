import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


EPS = 1e-12


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def compute_metrics(y_true, y_pred, num_classes):
    labels_all = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)

    p_present, r_present, f1_present, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    p_all, r_all, f1_all, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        average="macro",
        zero_division=0,
    )

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_precision_present": float(p_present),
        "macro_recall_present": float(r_present),
        "macro_f1_present": float(f1_present),
        "macro_precision_all": float(p_all),
        "macro_recall_all": float(r_all),
        "macro_f1_all": float(f1_all),
        "weighted_f1": float(f1_w),
    }


def load_bundle(m22_dir, m26_dir, prefix):
    m22_dir = Path(m22_dir)
    m26_dir = Path(m26_dir)

    m22_probs = normalize_probs(np.load(m22_dir / f"{prefix}_ensemble_probs.npy"))
    m26_probs = normalize_probs(np.load(m26_dir / f"{prefix}_m26_probs.npy"))

    m22_csv = pd.read_csv(m22_dir / f"{prefix}_ensemble_predictions.csv")
    m26_csv = pd.read_csv(m26_dir / f"{prefix}_m26_predictions.csv")

    if m22_probs.shape != m26_probs.shape:
        raise RuntimeError(f"Prob shape mismatch: {m22_probs.shape} vs {m26_probs.shape}")

    true_22 = m22_csv["true_mapped_label"].astype(int).values
    true_26 = m26_csv["true_mapped_label"].astype(int).values

    if not np.array_equal(true_22, true_26):
        raise RuntimeError(f"y_true mismatch for {prefix}")

    if "ensemble_pred_mapped_label" not in m22_csv.columns:
        raise RuntimeError("M22 csv missing ensemble_pred_mapped_label")

    if "m26_pred_mapped_label" not in m26_csv.columns:
        raise RuntimeError("M26 csv missing m26_pred_mapped_label")

    m22_pred = m22_csv["ensemble_pred_mapped_label"].astype(int).values
    m26_pred = m26_csv["m26_pred_mapped_label"].astype(int).values

    # Prefer saved confidence columns, but recompute from probabilities if needed.
    if "ensemble_confidence" in m22_csv.columns:
        m22_conf = m22_csv["ensemble_confidence"].astype(float).values
    else:
        m22_conf = m22_probs.max(axis=1)

    if "m26_confidence" in m26_csv.columns:
        m26_conf = m26_csv["m26_confidence"].astype(float).values
    else:
        m26_conf = m26_probs.max(axis=1)

    y_true = true_22

    return {
        "m22_probs": m22_probs,
        "m26_probs": m26_probs,
        "m22_csv": m22_csv,
        "m26_csv": m26_csv,
        "y_true": y_true,
        "m22_pred": m22_pred,
        "m26_pred": m26_pred,
        "m22_conf": m22_conf,
        "m26_conf": m26_conf,
    }


def build_pred_reliability(bundle, model_name, num_classes):
    if model_name == "m22":
        pred = bundle["m22_pred"]
    elif model_name == "m26":
        pred = bundle["m26_pred"]
    else:
        raise ValueError(model_name)

    y_true = bundle["y_true"]
    correct = pred == y_true

    rows = []

    for c in range(num_classes):
        mask = pred == c
        support = int(mask.sum())

        if support > 0:
            precision = float(correct[mask].mean())
        else:
            precision = 0.0

        rows.append({
            "pred_class": c,
            f"{model_name}_pred_support": support,
            f"{model_name}_pred_precision": precision,
        })

    return pd.DataFrame(rows)


def build_idx_to_original(pred_df, num_classes):
    idx_to_original = {i: str(i) for i in range(num_classes)}

    candidates = [
        ("true_mapped_label", "true_original_label"),
        ("ensemble_pred_mapped_label", "ensemble_pred_original_label"),
        ("m26_pred_mapped_label", "m26_pred_original_label"),
    ]

    for mapped_col, original_col in candidates:
        if mapped_col in pred_df.columns and original_col in pred_df.columns:
            for _, row in pred_df[[mapped_col, original_col]].drop_duplicates().iterrows():
                try:
                    idx_to_original[int(row[mapped_col])] = str(row[original_col])
                except Exception:
                    pass

    return idx_to_original


def get_rel_arrays(rel_df, num_classes):
    m22_support = np.zeros(num_classes, dtype=np.float64)
    m22_precision = np.zeros(num_classes, dtype=np.float64)
    m26_support = np.zeros(num_classes, dtype=np.float64)
    m26_precision = np.zeros(num_classes, dtype=np.float64)

    for _, row in rel_df.iterrows():
        c = int(row["pred_class"])
        m22_support[c] = float(row["m22_pred_support"])
        m22_precision[c] = float(row["m22_pred_precision"])
        m26_support[c] = float(row["m26_pred_support"])
        m26_precision[c] = float(row["m26_pred_precision"])

    return m22_support, m22_precision, m26_support, m26_precision


def apply_rule(bundle, rel_df, config):
    num_classes = bundle["m26_probs"].shape[1]

    m22_support, m22_precision, m26_support, m26_precision = get_rel_arrays(
        rel_df,
        num_classes,
    )

    m22_pred = bundle["m22_pred"]
    m26_pred = bundle["m26_pred"]

    m22_conf = bundle["m22_conf"]
    m26_conf = bundle["m26_conf"]

    m22_p = m22_precision[m22_pred]
    m26_p = m26_precision[m26_pred]

    m22_s = m22_support[m22_pred]
    m26_s = m26_support[m26_pred]

    min_support = float(config["min_pred_support"])
    m26_bad_thr = float(config["m26_bad_precision_threshold"])
    m22_good_thr = float(config["m22_good_precision_threshold"])
    precision_margin = float(config["precision_margin"])
    m22_conf_thr = float(config["m22_conf_threshold"])
    conf_margin = float(config["conf_margin"])

    switch_mask = (
        (m22_pred != m26_pred)
        & (m26_s >= min_support)
        & (m22_s >= min_support)
        & (m26_p <= m26_bad_thr)
        & (m22_p >= m22_good_thr)
        & ((m22_p - m26_p) >= precision_margin)
        & (m22_conf >= m22_conf_thr)
        & ((m22_conf - m26_conf) >= conf_margin)
    )

    final_probs = bundle["m26_probs"].copy()
    final_probs[switch_mask] = bundle["m22_probs"][switch_mask]

    final_pred = final_probs.argmax(axis=1)

    return final_probs, final_pred, switch_mask


def tune_rules(val_bundle, rel_df, num_classes, args):
    rows = []

    min_support_values = [int(x) for x in args.min_pred_support_values.split(",")]
    m26_bad_values = [float(x) for x in args.m26_bad_precision_thresholds.split(",")]
    m22_good_values = [float(x) for x in args.m22_good_precision_thresholds.split(",")]
    precision_margin_values = [float(x) for x in args.precision_margin_values.split(",")]
    m22_conf_values = [float(x) for x in args.m22_conf_thresholds.split(",")]
    conf_margin_values = [float(x) for x in args.conf_margin_values.split(",")]

    y_true = val_bundle["y_true"]

    for min_support in min_support_values:
        for m26_bad in m26_bad_values:
            for m22_good in m22_good_values:
                for precision_margin in precision_margin_values:
                    for m22_conf_thr in m22_conf_values:
                        for conf_margin in conf_margin_values:
                            config = {
                                "min_pred_support": min_support,
                                "m26_bad_precision_threshold": m26_bad,
                                "m22_good_precision_threshold": m22_good,
                                "precision_margin": precision_margin,
                                "m22_conf_threshold": m22_conf_thr,
                                "conf_margin": conf_margin,
                            }

                            _, pred, switch_mask = apply_rule(
                                val_bundle,
                                rel_df,
                                config,
                            )

                            metrics = compute_metrics(
                                y_true=y_true,
                                y_pred=pred,
                                num_classes=num_classes,
                            )

                            rows.append({
                                **config,
                                "num_switched": int(switch_mask.sum()),
                                **metrics,
                            })

    df = pd.DataFrame(rows)

    # Avoid selecting a rule that switches almost nothing unless it is truly best.
    df = df.sort_values(
        ["macro_f1_present", "accuracy", "weighted_f1", "num_switched"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return df


def save_outputs(output_dir, prefix, bundle, probs, pred, switch_mask, config):
    output_dir = Path(output_dir)
    num_classes = probs.shape[1]
    y_true = bundle["y_true"]

    metrics = compute_metrics(y_true, pred, num_classes)

    idx_to_original = build_idx_to_original(bundle["m26_csv"], num_classes)
    idx_to_original.update(build_idx_to_original(bundle["m22_csv"], num_classes))

    out_df = bundle["m26_csv"].copy()

    out_df["m27_pred_mapped_label"] = pred
    out_df["m27_pred_original_label"] = [
        idx_to_original.get(int(x), str(int(x))) for x in pred
    ]
    out_df["m27_confidence"] = probs.max(axis=1)
    out_df["m27_is_correct"] = out_df["true_mapped_label"].astype(int).values == pred
    out_df["m27_switched_to_m22"] = switch_mask

    out_df["m22_pred_mapped_label"] = bundle["m22_pred"]
    out_df["m26_pred_mapped_label_recheck"] = bundle["m26_pred"]
    out_df["m22_confidence_recheck"] = bundle["m22_conf"]
    out_df["m26_confidence_recheck"] = bundle["m26_conf"]

    pred_csv = output_dir / f"{prefix}_m27_predictions.csv"
    probs_npy = output_dir / f"{prefix}_m27_probs.npy"
    metrics_json = output_dir / f"{prefix}_m27_metrics.json"

    out_df.to_csv(pred_csv, index=False)
    np.save(probs_npy, probs)

    result = {
        "split": prefix,
        "num_rows": int(len(y_true)),
        "num_classes": int(num_classes),
        "num_switched_to_m22": int(switch_mask.sum()),
        "selected_config": config,
        **metrics,
    }

    save_json(result, metrics_json)

    print(f"\n=== {prefix.upper()} M27 RESULTS ===")
    print("Switched to M22   :", int(switch_mask.sum()))
    print(f"Accuracy          : {metrics['accuracy']:.6f}")
    print(f"Macro F1 Present  : {metrics['macro_f1_present']:.6f}")
    print(f"Macro F1 All      : {metrics['macro_f1_all']:.6f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.6f}")

    print("Saved:")
    print(pred_csv)
    print(probs_npy)
    print(metrics_json)

    return result


def compare_base(bundle, num_classes, prefix):
    rows = []

    for model_name in ["m22", "m26"]:
        if model_name == "m22":
            pred = bundle["m22_pred"]
        else:
            pred = bundle["m26_pred"]

        metrics = compute_metrics(
            y_true=bundle["y_true"],
            y_pred=pred,
            num_classes=num_classes,
        )

        rows.append({
            "split": prefix,
            "model": model_name.upper(),
            **metrics,
        })

    return pd.DataFrame(rows)


def main(args):
    ensure_dir(args.output_dir)
    output_dir = Path(args.output_dir)

    print("=== M27 CLASS-WISE FALLBACK ENSEMBLE ===")
    print("M22 dir:", args.m22_dir)
    print("M26 dir:", args.m26_dir)
    print("Output dir:", output_dir)

    val_bundle = load_bundle(args.m22_dir, args.m26_dir, prefix="val")
    test_bundle = load_bundle(args.m22_dir, args.m26_dir, prefix="test")

    num_classes = val_bundle["m26_probs"].shape[1]

    print("Num classes:", num_classes)
    print("Val rows:", len(val_bundle["y_true"]))
    print("Test rows:", len(test_bundle["y_true"]))

    val_base = compare_base(val_bundle, num_classes, "val")
    test_base = compare_base(test_bundle, num_classes, "test")

    base_df = pd.concat([val_base, test_base], ignore_index=True)
    base_df.to_csv(output_dir / "m27_base_metrics.csv", index=False)

    print("\n=== BASE METRICS ===")
    print(
        base_df[
            [
                "split",
                "model",
                "accuracy",
                "macro_f1_present",
                "macro_f1_all",
                "weighted_f1",
            ]
        ].to_string(index=False)
    )

    rel_m22 = build_pred_reliability(val_bundle, "m22", num_classes)
    rel_m26 = build_pred_reliability(val_bundle, "m26", num_classes)

    rel_df = rel_m22.merge(rel_m26, on="pred_class", how="outer").fillna(0)
    rel_df.to_csv(output_dir / "m27_val_pred_class_reliability.csv", index=False)

    print("\n=== LOW M26 PREDICTED-CLASS RELIABILITY TOP 30 ===")
    print(
        rel_df[rel_df["m26_pred_support"] > 0]
        .sort_values(["m26_pred_precision", "m26_pred_support"], ascending=[True, False])
        .head(30)
        .to_string(index=False)
    )

    grid_df = tune_rules(
        val_bundle=val_bundle,
        rel_df=rel_df,
        num_classes=num_classes,
        args=args,
    )

    grid_path = output_dir / "m27_val_grid_results.csv"
    grid_df.to_csv(grid_path, index=False)

    print("\n=== TOP 20 M27 VAL CONFIGS ===")
    display_cols = [
        "min_pred_support",
        "m26_bad_precision_threshold",
        "m22_good_precision_threshold",
        "precision_margin",
        "m22_conf_threshold",
        "conf_margin",
        "num_switched",
        "accuracy",
        "macro_f1_present",
        "macro_f1_all",
        "weighted_f1",
    ]
    print(grid_df.head(20)[display_cols].to_string(index=False))

    selected = grid_df.iloc[0].to_dict()

    selected_config = {
        "min_pred_support": int(selected["min_pred_support"]),
        "m26_bad_precision_threshold": float(selected["m26_bad_precision_threshold"]),
        "m22_good_precision_threshold": float(selected["m22_good_precision_threshold"]),
        "precision_margin": float(selected["precision_margin"]),
        "m22_conf_threshold": float(selected["m22_conf_threshold"]),
        "conf_margin": float(selected["conf_margin"]),
    }

    print("\n=== SELECTED M27 CONFIG BY VAL ===")
    for k, v in selected_config.items():
        print(f"{k}: {v}")

    save_json(
        {
            "selected_config": selected_config,
            "selection_rule": "Selected by validation macro_f1_present, then accuracy, then weighted_f1.",
            "note": "Base model is M26. Switch to M22 only when predicted-class reliability and confidence conditions pass.",
        },
        output_dir / "m27_selected_config.json",
    )

    val_probs, val_pred, val_switch = apply_rule(
        val_bundle,
        rel_df,
        selected_config,
    )

    test_probs, test_pred, test_switch = apply_rule(
        test_bundle,
        rel_df,
        selected_config,
    )

    val_result = save_outputs(
        output_dir=output_dir,
        prefix="val",
        bundle=val_bundle,
        probs=val_probs,
        pred=val_pred,
        switch_mask=val_switch,
        config=selected_config,
    )

    test_result = save_outputs(
        output_dir=output_dir,
        prefix="test",
        bundle=test_bundle,
        probs=test_probs,
        pred=test_pred,
        switch_mask=test_switch,
        config=selected_config,
    )

    summary = pd.DataFrame([val_result, test_result])
    summary.to_csv(output_dir / "m27_summary.csv", index=False)

    print("\nSaved summary:", output_dir / "m27_summary.csv")
    print("Saved grid:", grid_path)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M27 class-wise fallback from M26 to M22 using validation predicted-class reliability."
    )

    parser.add_argument("--m22_dir", type=str, required=True)
    parser.add_argument("--m26_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--min_pred_support_values", type=str, default="3,5,10,20")
    parser.add_argument("--m26_bad_precision_thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--m22_good_precision_thresholds", type=str, default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--precision_margin_values", type=str, default="0.00,0.05,0.10,0.15,0.20")
    parser.add_argument("--m22_conf_thresholds", type=str, default="0.00,0.20,0.30,0.40,0.50")
    parser.add_argument("--conf_margin_values", type=str, default="-0.20,-0.10,0.00,0.05,0.10")

    args = parser.parse_args()
    main(args)

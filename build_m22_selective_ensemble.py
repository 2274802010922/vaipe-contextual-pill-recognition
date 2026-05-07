import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
    }


def per_class_f1(y_true, y_pred, num_classes):
    labels = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return pd.DataFrame({
        "class_idx": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def top2_margin(probs):
    part = np.partition(probs, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return top1 - top2


def load_bundle(folder):
    folder = Path(folder)

    data = {
        "m19_probs": np.load(folder / "m19_val_probs.npy"),
        "m21_probs": np.load(folder / "m21_val_probs.npy"),
        "y_true": np.load(folder / "m19_val_y_true.npy"),
        "m19_pred": np.load(folder / "m19_val_y_pred.npy"),
        "m21_pred": np.load(folder / "m21_val_y_pred.npy"),
        "m19_csv": pd.read_csv(folder / "m19_val_predictions.csv"),
        "m21_csv": pd.read_csv(folder / "m21_val_predictions.csv"),
    }

    y_true_21 = np.load(folder / "m21_val_y_true.npy")

    if not np.array_equal(data["y_true"], y_true_21):
        raise RuntimeError(f"y_true mismatch in folder: {folder}")

    if data["m19_probs"].shape != data["m21_probs"].shape:
        raise RuntimeError(f"prob shape mismatch in folder: {folder}")

    return data


def build_idx_to_original(pred_df, num_classes):
    idx_to_original = {i: str(i) for i in range(num_classes)}

    for _, row in pred_df.iterrows():
        if "true_mapped_label" in pred_df.columns and "true_original_label" in pred_df.columns:
            idx_to_original[int(row["true_mapped_label"])] = str(row["true_original_label"])

        if "pred_mapped_label" in pred_df.columns and "pred_original_label" in pred_df.columns:
            idx_to_original[int(row["pred_mapped_label"])] = str(row["pred_original_label"])

    return idx_to_original


def choose_trusted_m21_classes(y_true, pred_m19, pred_m21, num_classes, min_support, min_f1_gain):
    pc19 = per_class_f1(y_true, pred_m19, num_classes)
    pc21 = per_class_f1(y_true, pred_m21, num_classes)

    merged = pc19.merge(
        pc21,
        on="class_idx",
        suffixes=("_m19", "_m21"),
    )

    merged["f1_gain_m21_minus_m19"] = merged["f1_m21"] - merged["f1_m19"]
    merged["recall_gain_m21_minus_m19"] = merged["recall_m21"] - merged["recall_m19"]

    trusted = merged[
        (merged["support_m19"] >= min_support)
        & (merged["f1_gain_m21_minus_m19"] >= min_f1_gain)
    ].copy()

    trusted_classes = sorted(trusted["class_idx"].astype(int).tolist())

    return trusted_classes, merged


def evaluate_global_grid(val, num_classes, grid_step):
    y_true = val["y_true"]
    rows = []

    weights = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 4)

    for w_m19 in weights:
        w_m21 = 1.0 - w_m19
        probs = w_m19 * val["m19_probs"] + w_m21 * val["m21_probs"]
        pred = probs.argmax(axis=1)

        metrics = compute_metrics(y_true, pred, num_classes)

        rows.append({
            "mode": "global_weighted",
            "w_m19": float(w_m19),
            "w_m21": float(w_m21),
            "threshold": None,
            "margin": None,
            "num_switched": 0,
            **metrics,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["macro_f1_present", "accuracy", "weighted_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def apply_selective_rule(
    m19_probs,
    m21_probs,
    trusted_classes,
    base_w_m19,
    threshold,
    margin_threshold,
):
    base_w_m21 = 1.0 - base_w_m19
    base_probs = base_w_m19 * m19_probs + base_w_m21 * m21_probs

    m21_pred = m21_probs.argmax(axis=1)
    m21_conf = m21_probs.max(axis=1)
    m21_margin = top2_margin(m21_probs)

    trusted_mask = np.isin(m21_pred, np.array(trusted_classes, dtype=np.int64))
    conf_mask = m21_conf >= threshold
    margin_mask = m21_margin >= margin_threshold

    switch_mask = trusted_mask & conf_mask & margin_mask

    final_probs = base_probs.copy()
    final_probs[switch_mask] = m21_probs[switch_mask]

    return final_probs, switch_mask


def evaluate_selective_grid(
    val,
    num_classes,
    trusted_classes,
    base_weights,
    thresholds,
    margins,
):
    y_true = val["y_true"]
    rows = []

    for base_w_m19 in base_weights:
        for threshold in thresholds:
            for margin in margins:
                probs, switch_mask = apply_selective_rule(
                    m19_probs=val["m19_probs"],
                    m21_probs=val["m21_probs"],
                    trusted_classes=trusted_classes,
                    base_w_m19=base_w_m19,
                    threshold=threshold,
                    margin_threshold=margin,
                )

                pred = probs.argmax(axis=1)
                metrics = compute_metrics(y_true, pred, num_classes)

                rows.append({
                    "mode": "selective_m21_on_trusted_pred",
                    "w_m19": float(base_w_m19),
                    "w_m21": float(1.0 - base_w_m19),
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "num_switched": int(switch_mask.sum()),
                    **metrics,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["macro_f1_present", "accuracy", "weighted_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def apply_selected_config(bundle, config, trusted_classes):
    mode = config["mode"]

    if mode == "global_weighted":
        probs = config["w_m19"] * bundle["m19_probs"] + config["w_m21"] * bundle["m21_probs"]
        switch_mask = np.zeros(len(bundle["y_true"]), dtype=bool)

    elif mode == "selective_m21_on_trusted_pred":
        probs, switch_mask = apply_selective_rule(
            m19_probs=bundle["m19_probs"],
            m21_probs=bundle["m21_probs"],
            trusted_classes=trusted_classes,
            base_w_m19=config["w_m19"],
            threshold=config["threshold"],
            margin_threshold=config["margin"],
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    pred = probs.argmax(axis=1)
    return probs, pred, switch_mask


def save_ensemble_outputs(output_dir, prefix, bundle, pred, probs, switch_mask, num_classes, idx_to_original, config):
    output_dir = Path(output_dir)

    y_true = bundle["y_true"]
    metrics = compute_metrics(y_true, pred, num_classes)

    metrics.update({
        "prefix": prefix,
        "selected_mode": config["mode"],
        "w_m19": config["w_m19"],
        "w_m21": config["w_m21"],
        "threshold": config.get("threshold"),
        "margin": config.get("margin"),
        "num_switched": int(switch_mask.sum()),
        "num_rows": int(len(y_true)),
        "num_classes": int(num_classes),
    })

    pred_df = bundle["m19_csv"].copy()
    pred_df["ensemble_pred_mapped_label"] = pred
    pred_df["ensemble_pred_original_label"] = [idx_to_original.get(int(x), str(int(x))) for x in pred]
    pred_df["ensemble_confidence"] = probs.max(axis=1)
    pred_df["ensemble_switched_to_m21"] = switch_mask
    pred_df["ensemble_is_correct"] = pred_df["true_mapped_label"].astype(int).values == pred

    pred_df.to_csv(output_dir / f"{prefix}_ensemble_predictions.csv", index=False)
    np.save(output_dir / f"{prefix}_ensemble_probs.npy", probs)
    save_json(metrics, output_dir / f"{prefix}_ensemble_metrics.json")

    print(f"\n=== {prefix.upper()} ENSEMBLE RESULTS ===")
    print(f"Mode             : {metrics['selected_mode']}")
    print(f"w_m19 / w_m21    : {metrics['w_m19']} / {metrics['w_m21']}")
    print(f"threshold        : {metrics['threshold']}")
    print(f"margin           : {metrics['margin']}")
    print(f"num switched     : {metrics['num_switched']}")
    print(f"Accuracy         : {metrics['accuracy']:.6f}")
    print(f"Macro F1 Present : {metrics['macro_f1_present']:.6f}")
    print(f"Macro F1 All     : {metrics['macro_f1_all']:.6f}")
    print(f"Weighted F1      : {metrics['weighted_f1']:.6f}")

    return metrics


def main(args):
    ensure_dir(args.output_dir)

    output_dir = Path(args.output_dir)

    print("=== BUILD M22 SELECTIVE ENSEMBLE ===")
    print("Val prediction dir :", args.val_dir)
    print("Test prediction dir:", args.test_dir)
    print("Output dir         :", output_dir)

    val = load_bundle(args.val_dir)
    test = load_bundle(args.test_dir)

    num_classes = val["m19_probs"].shape[1]

    if test["m19_probs"].shape[1] != num_classes:
        raise RuntimeError("Val/test num_classes mismatch.")

    idx_to_original = build_idx_to_original(val["m19_csv"], num_classes)
    idx_to_original.update(build_idx_to_original(test["m19_csv"], num_classes))

    print("Num classes:", num_classes)
    print("Val rows:", len(val["y_true"]))
    print("Test rows:", len(test["y_true"]))

    m19_val_metrics = compute_metrics(val["y_true"], val["m19_pred"], num_classes)
    m21_val_metrics = compute_metrics(val["y_true"], val["m21_pred"], num_classes)

    print("\n=== BASE VAL METRICS ===")
    print("M19 Val Macro F1:", round(m19_val_metrics["macro_f1_present"], 6))
    print("M21 Val Macro F1:", round(m21_val_metrics["macro_f1_present"], 6))

    trusted_classes, per_class_compare = choose_trusted_m21_classes(
        y_true=val["y_true"],
        pred_m19=val["m19_pred"],
        pred_m21=val["m21_pred"],
        num_classes=num_classes,
        min_support=args.min_support,
        min_f1_gain=args.min_f1_gain,
    )

    print("\nTrusted M21 classes:", trusted_classes)
    print("Num trusted M21 classes:", len(trusted_classes))

    per_class_compare.to_csv(output_dir / "m19_m21_val_per_class_compare.csv", index=False)

    global_grid = evaluate_global_grid(
        val=val,
        num_classes=num_classes,
        grid_step=args.grid_step,
    )

    global_grid.to_csv(output_dir / "global_weight_grid_val.csv", index=False)

    best_global = global_grid.iloc[0].to_dict()

    print("\n=== BEST GLOBAL VAL ===")
    print(global_grid.head(10).to_string(index=False))

    base_weights = sorted(set([
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        float(best_global["w_m19"]),
    ]))

    thresholds = [float(x) for x in args.thresholds.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

    selective_grid = evaluate_selective_grid(
        val=val,
        num_classes=num_classes,
        trusted_classes=trusted_classes,
        base_weights=base_weights,
        thresholds=thresholds,
        margins=margins,
    )

    selective_grid.to_csv(output_dir / "selective_grid_val.csv", index=False)

    print("\n=== BEST SELECTIVE VAL ===")
    print(selective_grid.head(10).to_string(index=False))

    all_candidates = pd.concat([global_grid, selective_grid], ignore_index=True)
    all_candidates = all_candidates.sort_values(
        ["macro_f1_present", "accuracy", "weighted_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    all_candidates.to_csv(output_dir / "all_ensemble_candidates_val.csv", index=False)

    selected = all_candidates.iloc[0].to_dict()

    print("\n=== SELECTED CONFIG BY VAL ===")
    for k, v in selected.items():
        print(f"{k}: {v}")

    save_json(
        {
            "selected_config": selected,
            "trusted_m21_classes": trusted_classes,
            "min_support": args.min_support,
            "min_f1_gain": args.min_f1_gain,
            "note": "Selected only by validation metrics. Test is used only for final application.",
        },
        output_dir / "m22_selected_config.json",
    )

    val_probs, val_pred, val_switch = apply_selected_config(
        val,
        selected,
        trusted_classes,
    )

    test_probs, test_pred, test_switch = apply_selected_config(
        test,
        selected,
        trusted_classes,
    )

    val_metrics = save_ensemble_outputs(
        output_dir=output_dir,
        prefix="val",
        bundle=val,
        pred=val_pred,
        probs=val_probs,
        switch_mask=val_switch,
        num_classes=num_classes,
        idx_to_original=idx_to_original,
        config=selected,
    )

    test_metrics = save_ensemble_outputs(
        output_dir=output_dir,
        prefix="test",
        bundle=test,
        pred=test_pred,
        probs=test_probs,
        switch_mask=test_switch,
        num_classes=num_classes,
        idx_to_original=idx_to_original,
        config=selected,
    )

    summary = pd.DataFrame([
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])

    summary.to_csv(output_dir / "m22_ensemble_summary.csv", index=False)

    print("\nSaved summary:", output_dir / "m22_ensemble_summary.csv")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build M22 selective ensemble from M19 and M21 probabilities."
    )

    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--grid_step", type=float, default=0.02)

    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--min_f1_gain", type=float, default=0.05)

    parser.add_argument("--thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--margins", type=str, default="0.00,0.05,0.10,0.15,0.20")

    args = parser.parse_args()
    main(args)

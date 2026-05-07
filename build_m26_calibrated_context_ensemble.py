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


def load_prediction_bundle(m19_m21_dir, m22_dir, prefix):
    """
    prefix:
      - val  -> M22 file val_ensemble_probs.npy
      - test -> M22 file test_ensemble_probs.npy

    M19/M21 folder uses names m19_val_probs.npy even for test folder
    because the old script saved eval outputs with val naming.
    """
    m19_m21_dir = Path(m19_m21_dir)
    m22_dir = Path(m22_dir)

    m19_probs = np.load(m19_m21_dir / "m19_val_probs.npy").astype(np.float64)
    m21_probs = np.load(m19_m21_dir / "m21_val_probs.npy").astype(np.float64)

    m19_y_true = np.load(m19_m21_dir / "m19_val_y_true.npy").astype(np.int64)
    m21_y_true = np.load(m19_m21_dir / "m21_val_y_true.npy").astype(np.int64)

    if not np.array_equal(m19_y_true, m21_y_true):
        raise RuntimeError(f"M19/M21 y_true mismatch in {m19_m21_dir}")

    m22_probs = np.load(m22_dir / f"{prefix}_ensemble_probs.npy").astype(np.float64)
    m22_csv = pd.read_csv(m22_dir / f"{prefix}_ensemble_predictions.csv")

    if m19_probs.shape != m21_probs.shape:
        raise RuntimeError(f"M19/M21 prob shape mismatch in {m19_m21_dir}")

    if m19_probs.shape != m22_probs.shape:
        raise RuntimeError(f"M19/M22 prob shape mismatch: {m19_probs.shape} vs {m22_probs.shape}")

    if len(m19_y_true) != m22_probs.shape[0]:
        raise RuntimeError("y_true length does not match M22 probs.")

    return {
        "m19_probs": normalize_probs(m19_probs),
        "m21_probs": normalize_probs(m21_probs),
        "m22_probs": normalize_probs(m22_probs),
        "y_true": m19_y_true,
        "csv": m22_csv,
    }


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def probs_to_logits(probs):
    probs = normalize_probs(probs)
    return np.log(probs + EPS)


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def build_train_prior(train_metadata, num_classes, smoothing=1.0):
    df = pd.read_csv(train_metadata)

    if "mapped_label" not in df.columns:
        raise ValueError("train_metadata must contain mapped_label column.")

    counts = np.zeros(num_classes, dtype=np.float64)

    for y in df["mapped_label"].astype(int).tolist():
        if 0 <= y < num_classes:
            counts[y] += 1.0

    counts = counts + float(smoothing)
    prior = counts / counts.sum()

    return prior


def combine_probs(bundle, config, train_prior):
    mode = config["mode"]
    w19 = float(config["w_m19"])
    w21 = float(config["w_m21"])
    w22 = float(config["w_m22"])
    alpha = float(config["logit_adjust_alpha"])
    temperature = float(config["temperature"])

    p19 = bundle["m19_probs"]
    p21 = bundle["m21_probs"]
    p22 = bundle["m22_probs"]

    if mode == "prob_average":
        probs = w19 * p19 + w21 * p21 + w22 * p22
        logits = probs_to_logits(probs)

    elif mode == "logit_average":
        l19 = probs_to_logits(p19)
        l21 = probs_to_logits(p21)
        l22 = probs_to_logits(p22)

        logits = w19 * l19 + w21 * l21 + w22 * l22

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # logit adjustment:
    # logits - alpha * log(train_prior)
    # alpha > 0 boosts rare classes because log(prior) is more negative.
    log_prior = np.log(np.clip(train_prior, EPS, 1.0))[None, :]
    logits = logits - alpha * log_prior

    logits = logits / max(temperature, 1e-6)

    final_probs = softmax_np(logits)
    pred = final_probs.argmax(axis=1)

    return final_probs, pred


def generate_weight_grid(step):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    rows = []

    for w19 in values:
        for w21 in values:
            w22 = 1.0 - w19 - w21

            if w22 < -1e-9:
                continue

            w22 = round(float(w22), 4)

            if w22 < 0:
                continue

            rows.append((float(w19), float(w21), float(w22)))

    return rows


def tune_on_validation(val_bundle, train_prior, num_classes, args):
    rows = []

    weight_grid = generate_weight_grid(args.weight_step)

    modes = ["prob_average", "logit_average"]

    alphas = [float(x) for x in args.logit_adjust_alphas.split(",")]
    temperatures = [float(x) for x in args.temperatures.split(",")]

    y_true = val_bundle["y_true"]

    print("Grid modes:", modes)
    print("Num weight combinations:", len(weight_grid))
    print("Alphas:", alphas)
    print("Temperatures:", temperatures)

    for mode in modes:
        for w19, w21, w22 in weight_grid:
            for alpha in alphas:
                for temp in temperatures:
                    config = {
                        "mode": mode,
                        "w_m19": w19,
                        "w_m21": w21,
                        "w_m22": w22,
                        "logit_adjust_alpha": alpha,
                        "temperature": temp,
                    }

                    _, pred = combine_probs(
                        bundle=val_bundle,
                        config=config,
                        train_prior=train_prior,
                    )

                    metrics = compute_metrics(
                        y_true=y_true,
                        y_pred=pred,
                        num_classes=num_classes,
                    )

                    rows.append({
                        **config,
                        **metrics,
                    })

    grid_df = pd.DataFrame(rows)

    grid_df = grid_df.sort_values(
        ["macro_f1_present", "accuracy", "weighted_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grid_df


def build_idx_to_original(pred_df, num_classes):
    idx_to_original = {i: str(i) for i in range(num_classes)}

    if "true_mapped_label" in pred_df.columns and "true_original_label" in pred_df.columns:
        for _, row in pred_df[["true_mapped_label", "true_original_label"]].drop_duplicates().iterrows():
            idx_to_original[int(row["true_mapped_label"])] = str(row["true_original_label"])

    if "ensemble_pred_mapped_label" in pred_df.columns and "ensemble_pred_original_label" in pred_df.columns:
        for _, row in pred_df[["ensemble_pred_mapped_label", "ensemble_pred_original_label"]].drop_duplicates().iterrows():
            idx_to_original[int(row["ensemble_pred_mapped_label"])] = str(row["ensemble_pred_original_label"])

    return idx_to_original


def save_outputs(output_dir, prefix, bundle, probs, pred, config, train_prior):
    output_dir = Path(output_dir)
    num_classes = probs.shape[1]
    y_true = bundle["y_true"]

    metrics = compute_metrics(y_true, pred, num_classes)

    idx_to_original = build_idx_to_original(bundle["csv"], num_classes)

    out_df = bundle["csv"].copy()

    out_df["m26_pred_mapped_label"] = pred
    out_df["m26_pred_original_label"] = [idx_to_original.get(int(x), str(int(x))) for x in pred]
    out_df["m26_confidence"] = probs.max(axis=1)
    out_df["m26_is_correct"] = out_df["true_mapped_label"].astype(int).values == pred

    # Also keep top-3 for error analysis.
    top3 = np.argsort(-probs, axis=1)[:, :3]
    out_df["m26_top1"] = top3[:, 0]
    out_df["m26_top2"] = top3[:, 1]
    out_df["m26_top3"] = top3[:, 2]

    pred_csv = output_dir / f"{prefix}_m26_predictions.csv"
    probs_npy = output_dir / f"{prefix}_m26_probs.npy"
    metrics_json = output_dir / f"{prefix}_m26_metrics.json"

    out_df.to_csv(pred_csv, index=False)
    np.save(probs_npy, probs)

    result = {
        "split": prefix,
        "num_rows": int(len(y_true)),
        "num_classes": int(num_classes),
        "selected_config": config,
        "train_prior_min": float(train_prior.min()),
        "train_prior_max": float(train_prior.max()),
        **metrics,
    }

    save_json(result, metrics_json)

    print(f"\n=== {prefix.upper()} M26 RESULTS ===")
    print("Mode             :", config["mode"])
    print("Weights M19/M21/M22:", config["w_m19"], config["w_m21"], config["w_m22"])
    print("Logit alpha      :", config["logit_adjust_alpha"])
    print("Temperature      :", config["temperature"])
    print(f"Accuracy         : {metrics['accuracy']:.6f}")
    print(f"Macro F1 Present : {metrics['macro_f1_present']:.6f}")
    print(f"Macro F1 All     : {metrics['macro_f1_all']:.6f}")
    print(f"Weighted F1      : {metrics['weighted_f1']:.6f}")

    print("Saved:")
    print(pred_csv)
    print(probs_npy)
    print(metrics_json)

    return result


def compare_baselines(bundle, num_classes, prefix):
    y_true = bundle["y_true"]

    rows = []

    for name in ["m19", "m21", "m22"]:
        probs = bundle[f"{name}_probs"]
        pred = probs.argmax(axis=1)

        metrics = compute_metrics(y_true, pred, num_classes)

        rows.append({
            "split": prefix,
            "model": name.upper(),
            **metrics,
        })

    return pd.DataFrame(rows)


def main(args):
    ensure_dir(args.output_dir)
    output_dir = Path(args.output_dir)

    print("=== M26 CALIBRATED CONTEXT ENSEMBLE ===")
    print("Val M19/M21 dir :", args.val_m19_m21_dir)
    print("Test M19/M21 dir:", args.test_m19_m21_dir)
    print("M22 dir         :", args.m22_dir)
    print("Train metadata  :", args.train_metadata)
    print("Output dir      :", output_dir)

    val_bundle = load_prediction_bundle(
        m19_m21_dir=args.val_m19_m21_dir,
        m22_dir=args.m22_dir,
        prefix="val",
    )

    test_bundle = load_prediction_bundle(
        m19_m21_dir=args.test_m19_m21_dir,
        m22_dir=args.m22_dir,
        prefix="test",
    )

    num_classes = val_bundle["m19_probs"].shape[1]

    if test_bundle["m19_probs"].shape[1] != num_classes:
        raise RuntimeError("Val/test num_classes mismatch.")

    train_prior = build_train_prior(
        train_metadata=args.train_metadata,
        num_classes=num_classes,
        smoothing=args.prior_smoothing,
    )

    print("Num classes:", num_classes)
    print("Val rows:", len(val_bundle["y_true"]))
    print("Test rows:", len(test_bundle["y_true"]))
    print("Train prior min/max:", train_prior.min(), train_prior.max())

    val_base = compare_baselines(val_bundle, num_classes, "val")
    test_base = compare_baselines(test_bundle, num_classes, "test")

    base_df = pd.concat([val_base, test_base], ignore_index=True)
    base_df.to_csv(output_dir / "m26_baseline_metrics.csv", index=False)

    print("\n=== BASELINE METRICS ===")
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

    grid_df = tune_on_validation(
        val_bundle=val_bundle,
        train_prior=train_prior,
        num_classes=num_classes,
        args=args,
    )

    grid_csv = output_dir / "m26_val_grid_results.csv"
    grid_df.to_csv(grid_csv, index=False)

    print("\n=== TOP 20 VAL CONFIGS ===")
    display_cols = [
        "mode",
        "w_m19",
        "w_m21",
        "w_m22",
        "logit_adjust_alpha",
        "temperature",
        "accuracy",
        "macro_f1_present",
        "macro_f1_all",
        "weighted_f1",
    ]
    print(grid_df.head(20)[display_cols].to_string(index=False))

    selected = grid_df.iloc[0].to_dict()

    selected_config = {
        "mode": selected["mode"],
        "w_m19": float(selected["w_m19"]),
        "w_m21": float(selected["w_m21"]),
        "w_m22": float(selected["w_m22"]),
        "logit_adjust_alpha": float(selected["logit_adjust_alpha"]),
        "temperature": float(selected["temperature"]),
    }

    save_json(
        {
            "selected_config": selected_config,
            "selection_rule": "Selected by validation macro_f1_present, then accuracy, then weighted_f1.",
            "weight_step": args.weight_step,
            "logit_adjust_alphas": args.logit_adjust_alphas,
            "temperatures": args.temperatures,
            "prior_smoothing": args.prior_smoothing,
        },
        output_dir / "m26_selected_config.json",
    )

    print("\n=== SELECTED CONFIG BY VAL ===")
    for k, v in selected_config.items():
        print(f"{k}: {v}")

    val_probs, val_pred = combine_probs(
        bundle=val_bundle,
        config=selected_config,
        train_prior=train_prior,
    )

    test_probs, test_pred = combine_probs(
        bundle=test_bundle,
        config=selected_config,
        train_prior=train_prior,
    )

    val_result = save_outputs(
        output_dir=output_dir,
        prefix="val",
        bundle=val_bundle,
        probs=val_probs,
        pred=val_pred,
        config=selected_config,
        train_prior=train_prior,
    )

    test_result = save_outputs(
        output_dir=output_dir,
        prefix="test",
        bundle=test_bundle,
        probs=test_probs,
        pred=test_pred,
        config=selected_config,
        train_prior=train_prior,
    )

    summary_df = pd.DataFrame([val_result, test_result])
    summary_df.to_csv(output_dir / "m26_summary.csv", index=False)

    print("\nSaved summary:", output_dir / "m26_summary.csv")
    print("Saved grid:", grid_csv)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M26 calibrated/logit-adjusted ensemble for M19, M21, M22 context predictions."
    )

    parser.add_argument("--val_m19_m21_dir", type=str, required=True)
    parser.add_argument("--test_m19_m21_dir", type=str, required=True)
    parser.add_argument("--m22_dir", type=str, required=True)
    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--weight_step", type=float, default=0.05)
    parser.add_argument("--prior_smoothing", type=float, default=1.0)

    parser.add_argument(
        "--logit_adjust_alphas",
        type=str,
        default="-0.50,-0.25,0.00,0.10,0.20,0.30,0.40,0.50,0.75,1.00",
    )

    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.75,1.00,1.25,1.50,2.00",
    )

    args = parser.parse_args()
    main(args)

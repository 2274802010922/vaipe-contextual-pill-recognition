import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HARD_THRESHOLD = 0.45
CONF_GAP_THRESHOLD = 0.08
M30_BOOST = 0.65
TOPK_CHECK = 5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_probs(probs):
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    present_labels = sorted(list(set(y_true)))

    macro_present = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=present_labels,
        zero_division=0
    )

    macro_all = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1_present": float(macro_present),
        "macro_f1_all": float(macro_all),
        "weighted_f1": float(weighted)
    }


def get_hard_classes(weights_csv):
    df = pd.read_csv(weights_csv)

    hard_classes = set()

    for _, row in df.iterrows():
        recall = row["m26_val_recall"]
        support = row["m26_val_support"]
        weight = row["final_weight"]

        if recall <= 0.25:
            hard_classes.add(int(row["class_id"]))

        elif weight >= 1.5:
            hard_classes.add(int(row["class_id"]))

        elif support <= 10:
            hard_classes.add(int(row["class_id"]))

    return hard_classes


def dynamic_ensemble(
    probs_m26,
    probs_m30,
    hard_classes
):
    final_probs = []

    for i in range(len(probs_m26)):
        p26 = probs_m26[i]
        p30 = probs_m30[i]

        top1 = int(np.argmax(p26))

        sorted_idx = np.argsort(p26)[::-1]
        top1_score = p26[sorted_idx[0]]
        top2_score = p26[sorted_idx[1]]

        confidence_gap = top1_score - top2_score

        use_m30 = False

        if top1_score < HARD_THRESHOLD:
            use_m30 = True

        if confidence_gap < CONF_GAP_THRESHOLD:
            use_m30 = True

        if top1 in hard_classes:
            use_m30 = True

        topk = sorted_idx[:TOPK_CHECK]

        if any(cls in hard_classes for cls in topk):
            use_m30 = True

        if use_m30:
            blend = (1.0 - M30_BOOST) * p26 + M30_BOOST * p30
        else:
            blend = 0.85 * p26 + 0.15 * p30

        blend = blend / blend.sum()

        final_probs.append(blend)

    return np.array(final_probs)


def evaluate_split(
    split_name,
    pred_csv_m26,
    probs_npy_m26,
    pred_csv_m30,
    probs_npy_m30,
    hard_classes,
    output_dir
):
    df26 = pd.read_csv(pred_csv_m26)
    df30 = pd.read_csv(pred_csv_m30)

    probs26 = np.load(probs_npy_m26)
    probs30 = np.load(probs_npy_m30)

    probs26 = normalize_probs(probs26)
    probs30 = normalize_probs(probs30)

    y_true = df26["true_label"].values

    ensemble_probs = dynamic_ensemble(
        probs26,
        probs30,
        hard_classes
    )

    y_pred = np.argmax(ensemble_probs, axis=1)

    metrics = compute_metrics(y_true, y_pred)

    out_df = df26.copy()

    out_df["m31_pred"] = y_pred
    out_df["m26_pred"] = np.argmax(probs26, axis=1)
    out_df["m30_pred"] = np.argmax(probs30, axis=1)

    out_csv = os.path.join(
        output_dir,
        f"{split_name}_m31_predictions.csv"
    )

    out_probs = os.path.join(
        output_dir,
        f"{split_name}_m31_probs.npy"
    )

    out_metrics = os.path.join(
        output_dir,
        f"{split_name}_m31_metrics.json"
    )

    out_df.to_csv(out_csv, index=False)
    np.save(out_probs, ensemble_probs)

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== M31 RESULTS:", split_name.upper(), "===")

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m26_dir", type=str, required=True)
    parser.add_argument("--m30_dir", type=str, required=True)
    parser.add_argument("--weights_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hard_classes = get_hard_classes(args.weights_csv)

    print("\n=== M31 DYNAMIC HARD ENSEMBLE ===")
    print("Hard classes:", len(hard_classes))

    val_metrics = evaluate_split(
        split_name="val",
        pred_csv_m26=os.path.join(args.m26_dir, "val_m26_predictions.csv"),
        probs_npy_m26=os.path.join(args.m26_dir, "val_m26_probs.npy"),
        pred_csv_m30=os.path.join(args.m30_dir, "val_m30_predictions.csv"),
        probs_npy_m30=os.path.join(args.m30_dir, "val_m30_probs.npy"),
        hard_classes=hard_classes,
        output_dir=args.output_dir
    )

    test_metrics = evaluate_split(
        split_name="test",
        pred_csv_m26=os.path.join(args.m26_dir, "test_m26_predictions.csv"),
        probs_npy_m26=os.path.join(args.m26_dir, "test_m26_probs.npy"),
        pred_csv_m30=os.path.join(args.m30_dir, "test_m30_predictions.csv"),
        probs_npy_m30=os.path.join(args.m30_dir, "test_m30_probs.npy"),
        hard_classes=hard_classes,
        output_dir=args.output_dir
    )

    summary = pd.DataFrame([
        {
            "split": "val",
            **val_metrics
        },
        {
            "split": "test",
            **test_metrics
        }
    ])

    summary_csv = os.path.join(
        args.output_dir,
        "m31_summary.csv"
    )

    summary.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(summary_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()

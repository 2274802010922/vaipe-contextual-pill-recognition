import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

LABEL_CANDIDATES = [
    "true_label",
    "mapped_label",
    "label",
    "pill_label",
]

PRED_CANDIDATES = [
    "pred_label",
    "prediction",
    "pred",
]

PROB_CANDIDATES = [
    "prob",
    "confidence",
    "max_prob",
]


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_hard_classes(metrics_json, recall_threshold=0.50):
    hard_classes = set()

    per_class = metrics_json.get("per_class_metrics", [])

    for row in per_class:
        recall = row.get("recall", 1.0)
        class_id = row.get("class_id")

        if recall < recall_threshold:
            hard_classes.add(int(class_id))

    return hard_classes


def ensemble_predictions(
    y26,
    y30,
    p26,
    hard_classes,
    confidence_threshold=0.60,
):
    final_preds = []

    for pred26, pred30, conf26 in zip(y26, y30, p26):

        use_m30 = False

        if pred26 in hard_classes:
            use_m30 = True

        if conf26 < confidence_threshold:
            use_m30 = True

        if use_m30:
            final_preds.append(pred30)
        else:
            final_preds.append(pred26)

    return np.array(final_preds)


def evaluate_split(
    df26,
    df30,
    split_name,
    hard_classes,
    confidence_threshold,
):
    label_col = first_existing(df26, LABEL_CANDIDATES)

    pred26_col = first_existing(df26, PRED_CANDIDATES)
    pred30_col = first_existing(df30, PRED_CANDIDATES)

    prob26_col = first_existing(df26, PROB_CANDIDATES)

    if label_col is None:
        raise ValueError(f"No label column found in {split_name}")

    if pred26_col is None:
        raise ValueError(f"No prediction column found for M26 in {split_name}")

    if pred30_col is None:
        raise ValueError(f"No prediction column found for M30 in {split_name}")

    if prob26_col is None:
        raise ValueError(f"No probability column found for M26 in {split_name}")

    y_true = df26[label_col].values
    y26 = df26[pred26_col].values
    y30 = df30[pred30_col].values
    p26 = df26[prob26_col].values

    final_preds = ensemble_predictions(
        y26=y26,
        y30=y30,
        p26=p26,
        hard_classes=hard_classes,
        confidence_threshold=confidence_threshold,
    )

    acc = accuracy_score(y_true, final_preds)

    macro_f1_present = f1_score(
        y_true,
        final_preds,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        final_preds,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "split": split_name,
        "accuracy": float(acc),
        "macro_f1_present": float(macro_f1_present),
        "weighted_f1": float(weighted_f1),
    }

    return metrics, final_preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m26_val_csv", type=str, required=True)
    parser.add_argument("--m26_test_csv", type=str, required=True)

    parser.add_argument("--m30_val_csv", type=str, required=True)
    parser.add_argument("--m30_test_csv", type=str, required=True)

    parser.add_argument("--m26_metrics_json", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--recall_threshold", type=float, default=0.50)
    parser.add_argument("--confidence_threshold", type=float, default=0.60)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    m26_metrics = load_metrics(args.m26_metrics_json)

    hard_classes = compute_hard_classes(
        m26_metrics,
        recall_threshold=args.recall_threshold,
    )

    print("\n=== M31 DYNAMIC HARD ENSEMBLE ===")
    print(f"Hard classes: {len(hard_classes)}")

    val26 = pd.read_csv(args.m26_val_csv)
    test26 = pd.read_csv(args.m26_test_csv)

    val30 = pd.read_csv(args.m30_val_csv)
    test30 = pd.read_csv(args.m30_test_csv)

    val_metrics, val_preds = evaluate_split(
        val26,
        val30,
        "val",
        hard_classes,
        args.confidence_threshold,
    )

    test_metrics, test_preds = evaluate_split(
        test26,
        test30,
        "test",
        hard_classes,
        args.confidence_threshold,
    )

    print("\n=== RESULTS ===")

    for row in [val_metrics, test_metrics]:
        print(json.dumps(row, indent=2))

    val26["m31_pred"] = val_preds
    test26["m31_pred"] = test_preds

    val26.to_csv(
        os.path.join(args.output_dir, "val_m31_predictions.csv"),
        index=False,
    )

    test26.to_csv(
        os.path.join(args.output_dir, "test_m31_predictions.csv"),
        index=False,
    )

    with open(
        os.path.join(args.output_dir, "m31_metrics.json"),
        "w",
    ) as f:
        json.dump(
            {
                "val": val_metrics,
                "test": test_metrics,
                "hard_classes": sorted(list(hard_classes)),
            },
            f,
            indent=2,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

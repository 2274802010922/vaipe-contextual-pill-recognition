import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


def find_true_col(df):
    candidates = [
        "true_label",
        "mapped_label",
        "label",
        "pill_label",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"No label column found. Available columns: {list(df.columns)}"
    )


def find_pred_col(df):
    candidates = [
        "pred_label",
        "prediction",
        "pred",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"No prediction column found. Available columns: {list(df.columns)}"
    )


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def load_hard_classes(metadata_csv):
    df = pd.read_csv(metadata_csv)

    if "is_hard_class" not in df.columns:
        raise ValueError("Missing is_hard_class column")

    if "mapped_label" not in df.columns:
        raise ValueError("Missing mapped_label column")

    hard_classes = (
        df[df["is_hard_class"] == 1]["mapped_label"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    hard_classes = sorted(list(set(hard_classes)))

    return hard_classes


def build_dynamic_predictions(
    df26,
    df30,
    hard_classes,
):
    true_col = find_true_col(df26)

    pred26_col = find_pred_col(df26)
    pred30_col = find_pred_col(df30)

    y_true = df26[true_col].astype(int).values

    pred26 = df26[pred26_col].astype(int).values
    pred30 = df30[pred30_col].astype(int).values

    final_preds = []

    use_m30_count = 0
    use_m26_count = 0

    for p26, p30 in zip(pred26, pred30):

        if int(p26) in hard_classes:
            final_preds.append(int(p30))
            use_m30_count += 1
        else:
            final_preds.append(int(p26))
            use_m26_count += 1

    final_preds = np.array(final_preds)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, final_preds)
        ),
        "macro_f1_present": float(
            f1_score(
                y_true,
                final_preds,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                final_preds,
                average="weighted",
                zero_division=0,
            )
        ),
        "num_samples": int(len(y_true)),
        "used_m30": int(use_m30_count),
        "used_m26": int(use_m26_count),
    }

    out_df = df26.copy()

    out_df["m26_pred"] = pred26
    out_df["m30_pred"] = pred30
    out_df["final_pred"] = final_preds

    return out_df, metrics


def save_metrics(path, metrics):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--m26_val_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m26_test_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m30_val_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m30_test_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m26_metrics_json",
        type=str,
        default="/content/drive/MyDrive/model/M26_calibrated_context_ensemble/run_v1/val_m26_metrics.json",
    )

    parser.add_argument(
        "--hard_metadata_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hard_classes = load_hard_classes(
        args.hard_metadata_csv
    )

    print("\n=== M31 DYNAMIC HARD ENSEMBLE ===")
    print(f"Hard classes: {len(hard_classes)}")

    m26_metrics = load_metrics(
        args.m26_metrics_json
    )

    print("\n=== BASE M26 METRICS ===")
    print(json.dumps(m26_metrics, indent=2))

    df26_val = pd.read_csv(args.m26_val_csv)
    df30_val = pd.read_csv(args.m30_val_csv)

    df26_test = pd.read_csv(args.m26_test_csv)
    df30_test = pd.read_csv(args.m30_test_csv)

    val_df, val_metrics = build_dynamic_predictions(
        df26_val,
        df30_val,
        hard_classes,
    )

    test_df, test_metrics = build_dynamic_predictions(
        df26_test,
        df30_test,
        hard_classes,
    )

    print("\n=== M31 VALIDATION ===")
    print(json.dumps(val_metrics, indent=2))

    print("\n=== M31 TEST ===")
    print(json.dumps(test_metrics, indent=2))

    val_path = os.path.join(
        args.output_dir,
        "val_m31_predictions.csv",
    )

    test_path = os.path.join(
        args.output_dir,
        "test_m31_predictions.csv",
    )

    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    save_metrics(
        os.path.join(
            args.output_dir,
            "val_m31_metrics.json",
        ),
        val_metrics,
    )

    save_metrics(
        os.path.join(
            args.output_dir,
            "test_m31_metrics.json",
        ),
        test_metrics,
    )

    summary_df = pd.DataFrame([
        {
            "split": "val",
            **val_metrics,
        },
        {
            "split": "test",
            **test_metrics,
        },
    ])

    summary_path = os.path.join(
        args.output_dir,
        "m31_summary.csv",
    )

    summary_df.to_csv(summary_path, index=False)

    print("\nSaved files:")
    print(val_path)
    print(test_path)
    print(summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

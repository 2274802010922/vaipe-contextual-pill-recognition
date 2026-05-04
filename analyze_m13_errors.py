import argparse
import json
from pathlib import Path

import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    predictions_path = Path(args.predictions_csv)
    per_class_path = Path(args.per_class_csv)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    if not per_class_path.exists():
        raise FileNotFoundError(f"Per-class metrics CSV not found: {per_class_path}")

    pred_df = pd.read_csv(predictions_path)
    per_class_df = pd.read_csv(per_class_path)

    print("Predictions CSV:", predictions_path)
    print("Per-class CSV  :", per_class_path)
    print("Output dir     :", output_dir)

    print("\nPrediction columns:")
    print(pred_df.columns.tolist())

    print("\nPer-class columns:")
    print(per_class_df.columns.tolist())

    required_pred_cols = [
        "true_mapped_label",
        "pred_mapped_label",
        "true_original_label",
        "pred_original_label",
        "is_correct",
    ]

    for col in required_pred_cols:
        if col not in pred_df.columns:
            raise ValueError(f"Missing required prediction column: {col}")

    required_metric_cols = ["mapped_label", "original_label", "precision", "recall", "f1", "support"]

    for col in required_metric_cols:
        if col not in per_class_df.columns:
            raise ValueError(f"Missing required per-class column: {col}")

    total_samples = len(pred_df)
    correct_samples = int(pred_df["is_correct"].sum())
    wrong_samples = total_samples - correct_samples
    accuracy = correct_samples / max(1, total_samples)

    print("\n=== OVERALL ===")
    print("Total samples :", total_samples)
    print("Correct       :", correct_samples)
    print("Wrong         :", wrong_samples)
    print("Accuracy      :", round(accuracy, 6))

    summary = {
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "wrong_samples": wrong_samples,
        "accuracy": accuracy,
    }

    with open(output_dir / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Only classes that appear in test.
    present_df = per_class_df[per_class_df["support"] > 0].copy()
    missing_df = per_class_df[per_class_df["support"] == 0].copy()

    present_df = present_df.sort_values(["f1", "support"], ascending=[True, False])
    missing_df = missing_df.sort_values("mapped_label")

    present_df.to_csv(output_dir / "classes_present_sorted_by_f1.csv", index=False)
    missing_df.to_csv(output_dir / "classes_missing_in_test.csv", index=False)

    low_f1_df = present_df[present_df["f1"] < args.low_f1_threshold].copy()
    low_f1_df.to_csv(output_dir / "low_f1_classes.csv", index=False)

    zero_recall_df = present_df[present_df["recall"] == 0].copy()
    zero_recall_df.to_csv(output_dir / "zero_recall_classes.csv", index=False)

    high_support_low_f1_df = present_df[
        (present_df["support"] >= args.min_support) &
        (present_df["f1"] < args.low_f1_threshold)
    ].copy()

    high_support_low_f1_df.to_csv(
        output_dir / "high_support_low_f1_classes.csv",
        index=False,
    )

    wrong_df = pred_df[pred_df["is_correct"] == False].copy()
    wrong_df.to_csv(output_dir / "wrong_predictions.csv", index=False)

    # Confusion pairs.
    confusion_df = (
        wrong_df
        .groupby([
            "true_mapped_label",
            "true_original_label",
            "pred_mapped_label",
            "pred_original_label",
        ])
        .size()
        .reset_index(name="wrong_count")
        .sort_values("wrong_count", ascending=False)
    )

    true_support = (
        pred_df
        .groupby(["true_mapped_label", "true_original_label"])
        .size()
        .reset_index(name="true_support")
    )

    confusion_df = confusion_df.merge(
        true_support,
        on=["true_mapped_label", "true_original_label"],
        how="left",
    )

    confusion_df["wrong_rate_within_true_class"] = (
        confusion_df["wrong_count"] / confusion_df["true_support"].clip(lower=1)
    )

    confusion_df.to_csv(output_dir / "top_confusion_pairs.csv", index=False)

    # Classes most often predicted.
    pred_distribution = (
        pred_df
        .groupby(["pred_mapped_label", "pred_original_label"])
        .size()
        .reset_index(name="predicted_count")
        .sort_values("predicted_count", ascending=False)
    )

    pred_distribution.to_csv(output_dir / "prediction_distribution.csv", index=False)

    # True distribution.
    true_distribution = (
        pred_df
        .groupby(["true_mapped_label", "true_original_label"])
        .size()
        .reset_index(name="true_count")
        .sort_values("true_count", ascending=False)
    )

    true_distribution.to_csv(output_dir / "true_distribution.csv", index=False)

    # Bias toward dominant predicted class.
    dominant_pred_label = int(pred_distribution.iloc[0]["pred_mapped_label"])
    dominant_pred_original = pred_distribution.iloc[0]["pred_original_label"]

    predicted_as_dominant = pred_df[
        pred_df["pred_mapped_label"] == dominant_pred_label
    ].copy()

    predicted_as_dominant_summary = (
        predicted_as_dominant
        .groupby(["true_mapped_label", "true_original_label"])
        .size()
        .reset_index(name="count_predicted_as_dominant")
        .sort_values("count_predicted_as_dominant", ascending=False)
    )

    predicted_as_dominant_summary.to_csv(
        output_dir / "predicted_as_dominant_class.csv",
        index=False,
    )

    print("\n=== DOMINANT PREDICTED CLASS ===")
    print("Mapped label  :", dominant_pred_label)
    print("Original label:", dominant_pred_original)
    print("Predicted count:", int(pred_distribution.iloc[0]["predicted_count"]))

    # Optional merge with train/val counts.
    if args.train_csv and Path(args.train_csv).exists():
        train_df = pd.read_csv(args.train_csv)

        if args.label_col in train_df.columns:
            train_counts = (
                train_df[args.label_col]
                .value_counts()
                .rename_axis("original_label")
                .reset_index(name="train_count")
            )

            train_counts["original_label"] = train_counts["original_label"].astype(int)
            per_class_df["original_label"] = per_class_df["original_label"].astype(int)

            merged_df = per_class_df.merge(
                train_counts,
                on="original_label",
                how="left",
            )

            merged_df["train_count"] = merged_df["train_count"].fillna(0).astype(int)
            merged_df = merged_df.sort_values(["f1", "support"], ascending=[True, False])

            merged_df.to_csv(
                output_dir / "per_class_metrics_with_train_count.csv",
                index=False,
            )

            rare_bad_df = merged_df[
                (merged_df["support"] > 0) &
                (merged_df["train_count"] <= args.rare_train_threshold) &
                (merged_df["f1"] < args.low_f1_threshold)
            ].copy()

            rare_bad_df.to_csv(
                output_dir / "rare_train_low_f1_classes.csv",
                index=False,
            )

    print("\n=== WORST PRESENT CLASSES BY F1 ===")
    print(present_df.head(args.print_top).to_string(index=False))

    print("\n=== HIGH SUPPORT LOW F1 CLASSES ===")
    if len(high_support_low_f1_df) == 0:
        print("None")
    else:
        print(high_support_low_f1_df.head(args.print_top).to_string(index=False))

    print("\n=== TOP CONFUSION PAIRS ===")
    print(confusion_df.head(args.print_top).to_string(index=False))

    print("\n=== PREDICTION DISTRIBUTION TOP ===")
    print(pred_distribution.head(args.print_top).to_string(index=False))

    print("\nSaved analysis files:")
    for p in sorted(output_dir.glob("*.csv")):
        print(p)
    print(output_dir / "error_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze M13 ensemble errors by class and confusion pairs."
    )

    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--per_class_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M13_ensemble_m10_m11/error_analysis",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/best_pika_split_metadata/train.csv",
    )

    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--low_f1_threshold", type=float, default=0.30)
    parser.add_argument("--min_support", type=int, default=20)
    parser.add_argument("--rare_train_threshold", type=int, default=30)
    parser.add_argument("--print_top", type=int, default=25)

    args = parser.parse_args()
    main(args)

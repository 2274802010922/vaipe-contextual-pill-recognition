import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label(x):
    if pd.isna(x):
        return "__NA__"

    s = str(x).strip()

    # Convert "12.0" -> "12"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def sort_label_key(x):
    try:
        return (0, int(x))
    except Exception:
        return (1, str(x))


def load_split_csv(path, split_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} CSV: {path}")

    df = pd.read_csv(path)

    if "pill_label" not in df.columns:
        raise ValueError(f"{split_name} CSV must contain pill_label column.")

    df = df.copy()
    df["label_norm"] = df["pill_label"].apply(normalize_label)
    df["split_name"] = split_name

    return df


def class_distribution(train_df, val_df, test_df):
    train_counts = train_df["label_norm"].value_counts().rename("train_count")
    val_counts = val_df["label_norm"].value_counts().rename("val_count")
    test_counts = test_df["label_norm"].value_counts().rename("test_count")

    labels = sorted(
        set(train_counts.index) | set(val_counts.index) | set(test_counts.index),
        key=sort_label_key,
    )

    dist = pd.DataFrame({"label": labels})
    dist = dist.merge(train_counts, left_on="label", right_index=True, how="left")
    dist = dist.merge(val_counts, left_on="label", right_index=True, how="left")
    dist = dist.merge(test_counts, left_on="label", right_index=True, how="left")

    for col in ["train_count", "val_count", "test_count"]:
        dist[col] = dist[col].fillna(0).astype(int)

    dist["total_count"] = dist["train_count"] + dist["val_count"] + dist["test_count"]

    dist["in_train"] = dist["train_count"] > 0
    dist["in_val"] = dist["val_count"] > 0
    dist["in_test"] = dist["test_count"] > 0

    dist["missing_in_val"] = dist["in_train"] & (~dist["in_val"])
    dist["missing_in_test"] = dist["in_train"] & (~dist["in_test"])

    dist["train_ratio"] = dist["train_count"] / max(1, train_df.shape[0])
    dist["val_ratio"] = dist["val_count"] / max(1, val_df.shape[0])
    dist["test_ratio"] = dist["test_count"] / max(1, test_df.shape[0])

    dist["val_to_train_ratio"] = dist["val_count"] / dist["train_count"].replace(0, np.nan)
    dist["test_to_train_ratio"] = dist["test_count"] / dist["train_count"].replace(0, np.nan)

    return dist


def split_summary(train_df, val_df, test_df, dist_df):
    summary = []

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        labels_present = sorted(df["label_norm"].unique().tolist(), key=sort_label_key)

        summary.append({
            "split": name,
            "rows": int(len(df)),
            "labels_present": int(len(labels_present)),
            "min_class_count": int(df["label_norm"].value_counts().min()),
            "max_class_count": int(df["label_norm"].value_counts().max()),
        })

    summary_df = pd.DataFrame(summary)

    train_labels = set(train_df["label_norm"].unique())
    val_labels = set(val_df["label_norm"].unique())
    test_labels = set(test_df["label_norm"].unique())

    label_coverage = {
        "train_labels": len(train_labels),
        "val_labels": len(val_labels),
        "test_labels": len(test_labels),
        "train_missing_in_val_count": len(train_labels - val_labels),
        "train_missing_in_val": sorted(list(train_labels - val_labels), key=sort_label_key),
        "train_missing_in_test_count": len(train_labels - test_labels),
        "train_missing_in_test": sorted(list(train_labels - test_labels), key=sort_label_key),
        "test_labels_not_in_train_count": len(test_labels - train_labels),
        "test_labels_not_in_train": sorted(list(test_labels - train_labels), key=sort_label_key),
    }

    return summary_df, label_coverage


def infer_prediction_columns(df):
    true_candidates = [
        "true_original_label",
        "true_label",
        "label",
        "pill_label",
        "true_mapped_label",
    ]

    pred_candidates = [
        "pred_original_label",
        "pred_label",
        "pred_mapped_label",
        "prediction",
        "pred",
    ]

    true_col = None
    pred_col = None

    for c in true_candidates:
        if c in df.columns:
            true_col = c
            break

    for c in pred_candidates:
        if c in df.columns:
            pred_col = c
            break

    if true_col is None or pred_col is None:
        raise ValueError(
            f"Cannot infer prediction columns. Columns found: {df.columns.tolist()}"
        )

    return true_col, pred_col


def evaluate_prediction_file(pred_path, all_labels, output_prefix, output_dir, class_dist=None):
    pred_path = Path(pred_path)

    if not pred_path.exists():
        print(f"[Warning] Prediction file not found: {pred_path}")
        return None

    df = pd.read_csv(pred_path)
    true_col, pred_col = infer_prediction_columns(df)

    df = df.copy()
    df["true_norm"] = df[true_col].apply(normalize_label)
    df["pred_norm"] = df[pred_col].apply(normalize_label)
    df["is_correct_diag"] = df["true_norm"] == df["pred_norm"]

    y_true = df["true_norm"].tolist()
    y_pred = df["pred_norm"].tolist()

    present_labels = sorted(set(y_true), key=sort_label_key)

    accuracy = accuracy_score(y_true, y_pred)

    p_present, r_present, f1_present, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    p_all, r_all, f1_all, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0,
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "prediction_file": str(pred_path),
        "true_column": true_col,
        "pred_column": pred_col,
        "num_rows": int(len(df)),
        "num_present_labels": int(len(present_labels)),
        "accuracy": float(accuracy),
        "macro_precision_present": float(p_present),
        "macro_recall_present": float(r_present),
        "macro_f1_present": float(f1_present),
        "macro_precision_all_train_labels": float(p_all),
        "macro_recall_all_train_labels": float(r_all),
        "macro_f1_all_train_labels": float(f1_all),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
    }

    save_json(metrics, output_dir / f"{output_prefix}_metrics.json")

    labels_for_report = all_labels

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_for_report,
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "label": labels_for_report,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    pred_count = pd.Series(y_pred).value_counts().rename("pred_count")
    per_class = per_class.merge(
        pred_count,
        left_on="label",
        right_index=True,
        how="left",
    )

    per_class["pred_count"] = per_class["pred_count"].fillna(0).astype(int)

    if class_dist is not None:
        per_class = per_class.merge(
            class_dist[
                [
                    "label",
                    "train_count",
                    "val_count",
                    "test_count",
                    "missing_in_val",
                    "missing_in_test",
                ]
            ],
            on="label",
            how="left",
        )

    per_class["error_count_est"] = per_class["support"] - (per_class["recall"] * per_class["support"])

    per_class.to_csv(output_dir / f"{output_prefix}_per_class_metrics.csv", index=False)

    hard_classes = per_class[per_class["support"] > 0].copy()
    hard_classes = hard_classes.sort_values(
        ["f1", "support"],
        ascending=[True, False],
    )
    hard_classes.to_csv(output_dir / f"{output_prefix}_hard_classes.csv", index=False)

    wrong = df[df["true_norm"] != df["pred_norm"]].copy()

    confusion_pairs = (
        wrong.groupby(["true_norm", "pred_norm"])
        .size()
        .reset_index(name="wrong_count")
        .sort_values("wrong_count", ascending=False)
    )

    true_support = df["true_norm"].value_counts().rename("true_support")
    pred_support = df["pred_norm"].value_counts().rename("pred_support")

    confusion_pairs = confusion_pairs.merge(
        true_support,
        left_on="true_norm",
        right_index=True,
        how="left",
    )

    confusion_pairs = confusion_pairs.merge(
        pred_support,
        left_on="pred_norm",
        right_index=True,
        how="left",
    )

    confusion_pairs.to_csv(output_dir / f"{output_prefix}_confusion_pairs.csv", index=False)

    # Save normalized prediction file for later comparison
    df.to_csv(output_dir / f"{output_prefix}_predictions_normalized.csv", index=False)

    print(f"\n=== {output_prefix.upper()} METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print(f"Saved: {output_dir / f'{output_prefix}_metrics.json'}")
    print(f"Saved: {output_dir / f'{output_prefix}_per_class_metrics.csv'}")
    print(f"Saved: {output_dir / f'{output_prefix}_hard_classes.csv'}")
    print(f"Saved: {output_dir / f'{output_prefix}_confusion_pairs.csv'}")

    return {
        "name": output_prefix,
        "metrics": metrics,
        "df": df,
        "per_class": per_class,
        "hard_classes": hard_classes,
        "confusion_pairs": confusion_pairs,
    }


def compare_two_prediction_files(pred_a_result, pred_b_result, output_dir):
    if pred_a_result is None or pred_b_result is None:
        print("[Info] Cannot compare predictions because one result is missing.")
        return None

    name_a = pred_a_result["name"]
    name_b = pred_b_result["name"]

    a = pred_a_result["df"].copy()
    b = pred_b_result["df"].copy()

    # Prefer pill_crop_path as stable key.
    if "pill_crop_path" in a.columns and "pill_crop_path" in b.columns:
        key = "pill_crop_path"

        a_small = a[[key, "true_norm", "pred_norm", "is_correct_diag"]].copy()
        b_small = b[[key, "true_norm", "pred_norm", "is_correct_diag"]].copy()

        if "confidence" in a.columns:
            a_small[f"{name_a}_confidence"] = a["confidence"]

        if "confidence" in b.columns:
            b_small[f"{name_b}_confidence"] = b["confidence"]

        merged = a_small.merge(
            b_small,
            on=key,
            how="inner",
            suffixes=(f"_{name_a}", f"_{name_b}"),
        )
    else:
        if len(a) != len(b):
            print("[Warning] Cannot align prediction files by index because lengths differ.")
            return None

        merged = pd.DataFrame({
            "row_index": np.arange(len(a)),
            f"true_norm_{name_a}": a["true_norm"],
            f"pred_norm_{name_a}": a["pred_norm"],
            f"is_correct_diag_{name_a}": a["is_correct_diag"],
            f"true_norm_{name_b}": b["true_norm"],
            f"pred_norm_{name_b}": b["pred_norm"],
            f"is_correct_diag_{name_b}": b["is_correct_diag"],
        })

        key = "row_index"

    true_a_col = f"true_norm_{name_a}"
    true_b_col = f"true_norm_{name_b}"

    pred_a_col = f"pred_norm_{name_a}"
    pred_b_col = f"pred_norm_{name_b}"

    correct_a_col = f"is_correct_diag_{name_a}"
    correct_b_col = f"is_correct_diag_{name_b}"

    if true_a_col not in merged.columns:
        true_a_col = "true_norm_" + name_a

    # In merge by key, true_norm columns are suffixed.
    if f"true_norm_{name_a}" not in merged.columns and "true_norm" in merged.columns:
        raise RuntimeError("Unexpected merged columns. Cannot compare.")

    merged["same_true_label"] = merged[true_a_col] == merged[true_b_col]

    merged["case"] = "unknown"
    merged.loc[merged[correct_a_col] & merged[correct_b_col], "case"] = "both_correct"
    merged.loc[merged[correct_a_col] & (~merged[correct_b_col]), "case"] = f"{name_a}_only_correct"
    merged.loc[(~merged[correct_a_col]) & merged[correct_b_col], "case"] = f"{name_b}_only_correct"
    merged.loc[(~merged[correct_a_col]) & (~merged[correct_b_col]), "case"] = "both_wrong"

    both_wrong = merged["case"] == "both_wrong"
    merged.loc[both_wrong & (merged[pred_a_col] == merged[pred_b_col]), "case"] = "both_wrong_same_prediction"
    merged.loc[both_wrong & (merged[pred_a_col] != merged[pred_b_col]), "case"] = "both_wrong_different_prediction"

    case_summary = (
        merged["case"]
        .value_counts()
        .rename_axis("case")
        .reset_index(name="count")
    )

    case_summary["ratio"] = case_summary["count"] / max(1, len(merged))

    per_class_compare = (
        merged.groupby(true_a_col)
        .agg(
            samples=(key, "count"),
            a_correct=(correct_a_col, "sum"),
            b_correct=(correct_b_col, "sum"),
        )
        .reset_index()
        .rename(columns={true_a_col: "label"})
    )

    per_class_compare["a_acc"] = per_class_compare["a_correct"] / per_class_compare["samples"]
    per_class_compare["b_acc"] = per_class_compare["b_correct"] / per_class_compare["samples"]
    per_class_compare["b_minus_a_acc"] = per_class_compare["b_acc"] - per_class_compare["a_acc"]

    per_class_compare = per_class_compare.sort_values(
        ["b_minus_a_acc", "samples"],
        ascending=[False, False],
    )

    merged.to_csv(output_dir / f"compare_{name_a}_vs_{name_b}_row_level.csv", index=False)
    case_summary.to_csv(output_dir / f"compare_{name_a}_vs_{name_b}_case_summary.csv", index=False)
    per_class_compare.to_csv(output_dir / f"compare_{name_a}_vs_{name_b}_per_class.csv", index=False)

    a_only = merged[merged["case"] == f"{name_a}_only_correct"].copy()
    b_only = merged[merged["case"] == f"{name_b}_only_correct"].copy()

    a_only.to_csv(output_dir / f"{name_a}_correct_{name_b}_wrong.csv", index=False)
    b_only.to_csv(output_dir / f"{name_b}_correct_{name_a}_wrong.csv", index=False)

    print(f"\n=== COMPARE {name_a.upper()} VS {name_b.upper()} ===")
    print(case_summary.to_string(index=False))
    print(f"Saved comparison files to: {output_dir}")

    return {
        "merged": merged,
        "case_summary": case_summary,
        "per_class_compare": per_class_compare,
    }


def write_markdown_report(
    output_dir,
    split_summary_df,
    label_coverage,
    class_dist,
    eval_results,
    compare_result,
):
    report_path = output_dir / "diagnosis_report.md"

    lines = []
    lines.append("# Diagnosis Report - VAIPE Clean Split and Model Errors\n")

    lines.append("## 1. Split Summary\n")
    lines.append(split_summary_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## 2. Label Coverage\n")
    lines.append(f"- Train labels: {label_coverage['train_labels']}")
    lines.append(f"- Val labels: {label_coverage['val_labels']}")
    lines.append(f"- Test labels: {label_coverage['test_labels']}")
    lines.append(f"- Labels in train but missing in val: {label_coverage['train_missing_in_val_count']}")
    lines.append(f"- Labels in train but missing in test: {label_coverage['train_missing_in_test_count']}")
    lines.append(f"- Labels in test but missing in train: {label_coverage['test_labels_not_in_train_count']}")
    lines.append("\n")

    lines.append("### Train labels missing in val\n")
    lines.append("```text")
    lines.append(str(label_coverage["train_missing_in_val"]))
    lines.append("```\n")

    lines.append("### Train labels missing in test\n")
    lines.append("```text")
    lines.append(str(label_coverage["train_missing_in_test"]))
    lines.append("```\n")

    lines.append("## 3. Most Frequent Classes\n")
    frequent = class_dist.sort_values("train_count", ascending=False).head(20)
    lines.append(frequent[["label", "train_count", "val_count", "test_count", "total_count"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## 4. Rare / Risky Classes\n")
    risky = class_dist[
        (class_dist["train_count"] <= 30) | ((class_dist["test_count"] > 0) & (class_dist["train_count"] <= 60))
    ].copy()
    risky = risky.sort_values(["train_count", "test_count"], ascending=[True, False]).head(40)
    lines.append(risky[["label", "train_count", "val_count", "test_count", "missing_in_val", "missing_in_test"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## 5. Model Metrics\n")

    for name, result in eval_results.items():
        if result is None:
            continue

        m = result["metrics"]
        lines.append(f"### {name}\n")
        lines.append(f"- Accuracy: {m['accuracy']:.6f}")
        lines.append(f"- Macro F1 Present: {m['macro_f1_present']:.6f}")
        lines.append(f"- Macro F1 All Train Labels: {m['macro_f1_all_train_labels']:.6f}")
        lines.append(f"- Weighted F1: {m['weighted_f1']:.6f}")
        lines.append("\n")

        hard = result["hard_classes"].copy()
        hard = hard[hard["support"] > 0].head(20)

        lines.append("Hardest classes:")
        lines.append(hard[["label", "support", "precision", "recall", "f1", "train_count", "val_count", "test_count"]].to_markdown(index=False))
        lines.append("\n")

        conf = result["confusion_pairs"].head(20)
        lines.append("Top confusion pairs:")
        lines.append(conf.to_markdown(index=False))
        lines.append("\n")

    if compare_result is not None:
        lines.append("## 6. Model Comparison\n")
        lines.append("### Case Summary\n")
        lines.append(compare_result["case_summary"].to_markdown(index=False))
        lines.append("\n")

        lines.append("### Classes where second model improves over first\n")
        tmp = compare_result["per_class_compare"].sort_values(
            ["b_minus_a_acc", "samples"],
            ascending=[False, False],
        ).head(20)
        lines.append(tmp.to_markdown(index=False))
        lines.append("\n")

        lines.append("### Classes where second model is worse than first\n")
        tmp = compare_result["per_class_compare"].sort_values(
            ["b_minus_a_acc", "samples"],
            ascending=[True, False],
        ).head(20)
        lines.append(tmp.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 7. Initial Interpretation\n")
    lines.append("- If validation labels and test labels differ strongly, validation may not predict test performance reliably.")
    lines.append("- If many test-supported classes have low train counts, Macro F1 will remain low unless imbalance is handled directly.")
    lines.append("- If a stronger model improves validation but worsens test, the model may be overfitting validation distribution.")
    lines.append("- If two models make different mistakes, ensemble may help. If they fail on the same classes, visual pretraining or split redesign is more important.")
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved markdown report: {report_path}")


def main(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("=== DIAGNOSE SPLIT AND ERRORS ===")
    print("Train CSV:", args.train_csv)
    print("Val CSV:", args.val_csv)
    print("Test CSV:", args.test_csv)
    print("M19 predictions:", args.m19_predictions)
    print("M21 predictions:", args.m21_predictions)
    print("Output dir:", output_dir)

    train_df = load_split_csv(args.train_csv, "train")
    val_df = load_split_csv(args.val_csv, "val")
    test_df = load_split_csv(args.test_csv, "test")

    class_dist = class_distribution(train_df, val_df, test_df)
    class_dist.to_csv(output_dir / "class_distribution_train_val_test.csv", index=False)

    summary_df, label_coverage = split_summary(train_df, val_df, test_df, class_dist)
    summary_df.to_csv(output_dir / "split_summary.csv", index=False)
    save_json(label_coverage, output_dir / "label_coverage.json")

    all_train_labels = sorted(train_df["label_norm"].unique().tolist(), key=sort_label_key)

    print("\n=== SPLIT SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== LABEL COVERAGE ===")
    print(json.dumps(label_coverage, ensure_ascii=False, indent=2))

    print("\nSaved split files:")
    print(output_dir / "class_distribution_train_val_test.csv")
    print(output_dir / "split_summary.csv")
    print(output_dir / "label_coverage.json")

    eval_results = {}

    if args.m19_predictions:
        eval_results["m19"] = evaluate_prediction_file(
            args.m19_predictions,
            all_train_labels,
            "m19",
            output_dir,
            class_dist=class_dist,
        )
    else:
        eval_results["m19"] = None

    if args.m21_predictions:
        eval_results["m21"] = evaluate_prediction_file(
            args.m21_predictions,
            all_train_labels,
            "m21",
            output_dir,
            class_dist=class_dist,
        )
    else:
        eval_results["m21"] = None

    compare_result = None

    if eval_results.get("m19") is not None and eval_results.get("m21") is not None:
        compare_result = compare_two_prediction_files(
            eval_results["m19"],
            eval_results["m21"],
            output_dir,
        )

    write_markdown_report(
        output_dir=output_dir,
        split_summary_df=summary_df,
        label_coverage=label_coverage,
        class_dist=class_dist,
        eval_results=eval_results,
        compare_result=compare_result,
    )

    print("\n=== DONE ===")
    print("Main report:", output_dir / "diagnosis_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose VAIPE split distribution and model errors."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--m19_predictions", type=str, default="")
    parser.add_argument("--m21_predictions", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/diagnosis_split_errors_v1",
    )

    args = parser.parse_args()
    main(args)

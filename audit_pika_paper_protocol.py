"""
Audit clean paper-like split for PIKA protocol / paper-compatible benchmarks.

Typical Colab run (after Drive mount + repo clone):
  python audit_pika_paper_protocol.py \\
    --train_csv /content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2/train_clean.csv \\
    --val_csv   /content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2/val_clean.csv \\
    --test_csv  /content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2/test_clean.csv \\
    --output_dir /content/drive/MyDrive/model/audit_pika_protocol_v1

Outputs: audit_report.md, audit_candidate_benchmarks.csv (for evaluate_candidate_benchmarks.py),
leakage CSVs, class distribution.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label(x):
    if pd.isna(x):
        return "__NA__"

    s = str(x).strip()

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


def load_csv(path, split_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} CSV: {path}")

    df = pd.read_csv(path)

    required = ["pill_label", "pill_crop_path", "prescription_image_path"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{split_name} CSV missing columns: {missing}")

    df = df.copy()
    df["label_norm"] = df["pill_label"].apply(normalize_label)
    df["split_name"] = split_name

    if "prescription_key" not in df.columns:
        df["prescription_key"] = df["prescription_image_path"].apply(
            lambda p: Path(str(p)).stem
        )

    return df


def check_paths(df, split_name):
    out = {}

    for col in ["pill_crop_path", "prescription_image_path"]:
        exists = df[col].apply(lambda p: Path(str(p)).exists())
        out[f"{col}_existing"] = int(exists.sum())
        out[f"{col}_missing"] = int((~exists).sum())
        out[f"{col}_total"] = int(len(df))

        if int((~exists).sum()) > 0:
            missing_examples = df.loc[~exists, [col, "pill_label", "prescription_key"]].head(10)
            out[f"{col}_missing_examples"] = missing_examples.to_dict(orient="records")
        else:
            out[f"{col}_missing_examples"] = []

    return out


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

    for c in ["train_count", "val_count", "test_count"]:
        dist[c] = dist[c].fillna(0).astype(int)

    dist["total_count"] = dist["train_count"] + dist["val_count"] + dist["test_count"]

    dist["in_train"] = dist["train_count"] > 0
    dist["in_val"] = dist["val_count"] > 0
    dist["in_test"] = dist["test_count"] > 0

    dist["in_all_splits"] = dist["in_train"] & dist["in_val"] & dist["in_test"]
    dist["missing_in_val"] = dist["in_train"] & (~dist["in_val"])
    dist["missing_in_test"] = dist["in_train"] & (~dist["in_test"])
    dist["test_not_in_train"] = dist["in_test"] & (~dist["in_train"])

    dist["train_ratio"] = dist["train_count"] / max(1, int(dist["train_count"].sum()))
    dist["val_ratio"] = dist["val_count"] / max(1, int(dist["val_count"].sum()))
    dist["test_ratio"] = dist["test_count"] / max(1, int(dist["test_count"].sum()))

    dist["val_to_train_ratio"] = dist["val_count"] / dist["train_count"].replace(0, np.nan)
    dist["test_to_train_ratio"] = dist["test_count"] / dist["train_count"].replace(0, np.nan)

    return dist


def split_summary(df, split_name):
    counts = df["label_norm"].value_counts()

    return {
        "split": split_name,
        "rows": int(len(df)),
        "groups": int(df["prescription_key"].nunique()),
        "labels_present": int(df["label_norm"].nunique()),
        "min_class_count": int(counts.min()) if len(counts) else 0,
        "max_class_count": int(counts.max()) if len(counts) else 0,
        "median_class_count": float(counts.median()) if len(counts) else 0,
        "mean_class_count": float(counts.mean()) if len(counts) else 0,
    }


def leakage_check(train_df, val_df, test_df):
    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    group_map = defaultdict(list)
    pill_path_map = defaultdict(list)
    pres_path_map = defaultdict(list)

    for split, df in split_dfs.items():
        for g in df["prescription_key"].dropna().unique():
            group_map[str(g)].append(split)

        for p in df["pill_crop_path"].dropna().unique():
            pill_path_map[str(p)].append(split)

        for p in df["prescription_image_path"].dropna().unique():
            pres_path_map[str(p)].append(split)

    group_leaks = []
    for k, splits in group_map.items():
        unique_splits = sorted(set(splits))
        if len(unique_splits) > 1:
            group_leaks.append({
                "key": k,
                "splits": ",".join(unique_splits),
                "num_splits": len(unique_splits),
            })

    pill_path_leaks = []
    for k, splits in pill_path_map.items():
        unique_splits = sorted(set(splits))
        if len(unique_splits) > 1:
            pill_path_leaks.append({
                "path": k,
                "splits": ",".join(unique_splits),
                "num_splits": len(unique_splits),
            })

    pres_path_leaks = []
    for k, splits in pres_path_map.items():
        unique_splits = sorted(set(splits))
        if len(unique_splits) > 1:
            pres_path_leaks.append({
                "path": k,
                "splits": ",".join(unique_splits),
                "num_splits": len(unique_splits),
            })

    return (
        pd.DataFrame(group_leaks),
        pd.DataFrame(pill_path_leaks),
        pd.DataFrame(pres_path_leaks),
    )


def support_bins(dist):
    bins = [
        ("train_1", dist["train_count"] == 1),
        ("train_2_5", (dist["train_count"] >= 2) & (dist["train_count"] <= 5)),
        ("train_6_10", (dist["train_count"] >= 6) & (dist["train_count"] <= 10)),
        ("train_11_30", (dist["train_count"] >= 11) & (dist["train_count"] <= 30)),
        ("train_31_100", (dist["train_count"] >= 31) & (dist["train_count"] <= 100)),
        ("train_101_500", (dist["train_count"] >= 101) & (dist["train_count"] <= 500)),
        ("train_501_plus", dist["train_count"] >= 501),
    ]

    rows = []

    for name, mask in bins:
        sub = dist[mask]
        rows.append({
            "bin": name,
            "num_labels": int(len(sub)),
            "train_rows": int(sub["train_count"].sum()),
            "val_rows": int(sub["val_count"].sum()),
            "test_rows": int(sub["test_count"].sum()),
            "labels_in_all_splits": int(sub["in_all_splits"].sum()),
            "labels_missing_in_val": int(sub["missing_in_val"].sum()),
            "labels_missing_in_test": int(sub["missing_in_test"].sum()),
        })

    return pd.DataFrame(rows)


def candidate_benchmarks(dist, train_df, val_df, test_df):
    rows = []

    candidates = [
        ("all_train_labels", dist["in_train"]),
        ("labels_present_in_train_val", dist["in_train"] & dist["in_val"]),
        ("labels_present_in_train_test", dist["in_train"] & dist["in_test"]),
        ("labels_present_in_all_splits", dist["in_all_splits"]),
        ("train_support_ge_5_and_in_test", (dist["train_count"] >= 5) & dist["in_test"]),
        ("train_support_ge_10_and_in_test", (dist["train_count"] >= 10) & dist["in_test"]),
        ("train_support_ge_30_and_in_test", (dist["train_count"] >= 30) & dist["in_test"]),
        ("train_support_ge_50_and_in_test", (dist["train_count"] >= 50) & dist["in_test"]),
        ("train_support_ge_100_and_in_test", (dist["train_count"] >= 100) & dist["in_test"]),
        ("train_support_ge_30_and_all_splits", (dist["train_count"] >= 30) & dist["in_all_splits"]),
        ("train_support_ge_50_and_all_splits", (dist["train_count"] >= 50) & dist["in_all_splits"]),
    ]

    for name, mask in candidates:
        labels = set(dist.loc[mask, "label"].tolist())

        rows.append({
            "candidate": name,
            "num_labels": int(len(labels)),
            "train_rows": int(train_df["label_norm"].isin(labels).sum()),
            "val_rows": int(val_df["label_norm"].isin(labels).sum()),
            "test_rows": int(test_df["label_norm"].isin(labels).sum()),
            "min_train_count": int(dist.loc[mask, "train_count"].min()) if len(labels) else 0,
            "max_train_count": int(dist.loc[mask, "train_count"].max()) if len(labels) else 0,
            "labels": ",".join(sorted(labels, key=sort_label_key)),
        })

    return pd.DataFrame(rows)


def compare_prediction_metrics(pred_path, num_classes, output_prefix, output_dir):
    pred_path = Path(pred_path)

    if not pred_path.exists():
        print(f"[Warning] Prediction file not found: {pred_path}")
        return None

    pred_df = pd.read_csv(pred_path)

    true_col = None
    pred_col = None

    for c in ["true_mapped_label", "mapped_label", "pill_label"]:
        if c in pred_df.columns:
            true_col = c
            break

    for c in [
        "ensemble_pred_mapped_label",
        "pred_mapped_label",
        "pred_idx_final",
        "pred_idx_first",
    ]:
        if c in pred_df.columns:
            pred_col = c
            break

    if true_col is None or pred_col is None:
        print(f"[Warning] Cannot infer columns for {pred_path}")
        print("Columns:", pred_df.columns.tolist())
        return None

    y_true = pred_df[true_col].astype(int).tolist()
    y_pred = pred_df[pred_col].astype(int).tolist()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    labels_all = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
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

    metrics = {
        "prediction_file": str(pred_path),
        "true_col": true_col,
        "pred_col": pred_col,
        "rows": int(len(pred_df)),
        "accuracy": float(acc),
        "macro_precision_present": float(p),
        "macro_recall_present": float(r),
        "macro_f1_present": float(f1),
        "macro_precision_all": float(p_all),
        "macro_recall_all": float(r_all),
        "macro_f1_all": float(f1_all),
        "weighted_f1": float(f1_w),
    }

    save_json(metrics, output_dir / f"{output_prefix}_metrics.json")

    return metrics


def write_report(
    output_dir,
    split_summaries,
    path_checks,
    dist,
    bins,
    candidate_df,
    group_leaks,
    pill_path_leaks,
    pres_path_leaks,
    optional_metrics,
):
    output_dir = Path(output_dir)
    report_path = output_dir / "audit_report.md"

    lines = []

    lines.append("# PIKA Protocol Audit Report\n")

    lines.append("## 1. Split Summary\n")
    lines.append(pd.DataFrame(split_summaries).to_markdown(index=False))
    lines.append("\n")

    lines.append("## 2. Path Check\n")
    path_rows = []
    for split, obj in path_checks.items():
        row = {"split": split}
        row.update({k: v for k, v in obj.items() if not k.endswith("_examples")})
        path_rows.append(row)
    lines.append(pd.DataFrame(path_rows).to_markdown(index=False))
    lines.append("\n")

    lines.append("## 3. Leakage Check\n")
    lines.append(f"- Group leakage rows: {len(group_leaks)}")
    lines.append(f"- Pill crop path leakage rows: {len(pill_path_leaks)}")
    lines.append(f"- Prescription image path leakage rows: {len(pres_path_leaks)}")
    lines.append("\n")

    lines.append("## 4. Label Coverage\n")
    lines.append(f"- Total labels in train: {int(dist['in_train'].sum())}")
    lines.append(f"- Total labels in val: {int(dist['in_val'].sum())}")
    lines.append(f"- Total labels in test: {int(dist['in_test'].sum())}")
    lines.append(f"- Labels present in all splits: {int(dist['in_all_splits'].sum())}")
    lines.append(f"- Train labels missing in val: {int(dist['missing_in_val'].sum())}")
    lines.append(f"- Train labels missing in test: {int(dist['missing_in_test'].sum())}")
    lines.append(f"- Test labels not in train: {int(dist['test_not_in_train'].sum())}")
    lines.append("\n")

    lines.append("## 5. Support Bins\n")
    lines.append(bins.to_markdown(index=False))
    lines.append("\n")

    lines.append("## 6. Candidate Paper-Compatible Benchmarks\n")
    display_cols = [
        "candidate",
        "num_labels",
        "train_rows",
        "val_rows",
        "test_rows",
        "min_train_count",
        "max_train_count",
    ]
    lines.append(candidate_df[display_cols].to_markdown(index=False))
    lines.append("\n")

    lines.append("## 7. Most Frequent Labels\n")
    lines.append(
        dist.sort_values("train_count", ascending=False)
        .head(30)[["label", "train_count", "val_count", "test_count", "in_all_splits"]]
        .to_markdown(index=False)
    )
    lines.append("\n")

    lines.append("## 8. Riskiest Labels\n")
    risky = dist[
        (dist["train_count"] <= 30)
        | ((dist["test_count"] > 0) & (dist["train_count"] <= 100))
        | (dist["missing_in_val"])
        | (dist["missing_in_test"])
    ].copy()
    risky = risky.sort_values(["train_count", "test_count"], ascending=[True, False]).head(60)
    lines.append(
        risky[
            [
                "label",
                "train_count",
                "val_count",
                "test_count",
                "in_all_splits",
                "missing_in_val",
                "missing_in_test",
            ]
        ].to_markdown(index=False)
    )
    lines.append("\n")

    if optional_metrics:
        lines.append("## 9. Optional Prediction Metrics\n")
        lines.append(pd.DataFrame(optional_metrics).to_markdown(index=False))
        lines.append("\n")

    lines.append("## 10. Interpretation\n")
    lines.append("- If many labels are missing from validation/test, direct comparison with a paper using a different split is unsafe.")
    lines.append("- If leakage rows are zero, the benchmark is stricter but may be much harder than older validation results.")
    lines.append("- Candidate paper-compatible benchmarks show subsets that may be closer to paper-style evaluation.")
    lines.append("- For final reporting, keep two tracks: paper-compatible benchmark and clean no-leak benchmark.")
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved report:", report_path)


def main(args):
    ensure_dir(args.output_dir)

    output_dir = Path(args.output_dir)

    print("=== AUDIT PIKA PAPER PROTOCOL / CLEAN SPLIT ===")
    print("Train CSV:", args.train_csv)
    print("Val CSV:", args.val_csv)
    print("Test CSV:", args.test_csv)
    print("Output dir:", output_dir)

    train_df = load_csv(args.train_csv, "train")
    val_df = load_csv(args.val_csv, "val")
    test_df = load_csv(args.test_csv, "test")

    path_checks = {
        "train": check_paths(train_df, "train"),
        "val": check_paths(val_df, "val"),
        "test": check_paths(test_df, "test"),
    }

    split_summaries = [
        split_summary(train_df, "train"),
        split_summary(val_df, "val"),
        split_summary(test_df, "test"),
    ]

    dist = class_distribution(train_df, val_df, test_df)
    bins = support_bins(dist)
    candidate_df = candidate_benchmarks(dist, train_df, val_df, test_df)

    group_leaks, pill_path_leaks, pres_path_leaks = leakage_check(train_df, val_df, test_df)

    dist.to_csv(output_dir / "audit_class_distribution.csv", index=False)
    bins.to_csv(output_dir / "audit_support_bins.csv", index=False)
    candidate_df.to_csv(output_dir / "audit_candidate_benchmarks.csv", index=False)
    pd.DataFrame(split_summaries).to_csv(output_dir / "audit_split_summary.csv", index=False)

    group_leaks.to_csv(output_dir / "audit_group_leakage.csv", index=False)
    pill_path_leaks.to_csv(output_dir / "audit_pill_path_leakage.csv", index=False)
    pres_path_leaks.to_csv(output_dir / "audit_prescription_path_leakage.csv", index=False)

    save_json(path_checks, output_dir / "audit_path_checks.json")

    optional_metrics = []

    if args.m19_predictions:
        m = compare_prediction_metrics(
            args.m19_predictions,
            num_classes=int(dist["in_train"].sum()),
            output_prefix="m19",
            output_dir=output_dir,
        )
        if m:
            optional_metrics.append({"model": "m19", **m})

    if args.m22_predictions:
        m = compare_prediction_metrics(
            args.m22_predictions,
            num_classes=int(dist["in_train"].sum()),
            output_prefix="m22",
            output_dir=output_dir,
        )
        if m:
            optional_metrics.append({"model": "m22", **m})

    print("\n=== SPLIT SUMMARY ===")
    print(pd.DataFrame(split_summaries).to_string(index=False))

    print("\n=== PATH CHECK ===")
    print(pd.DataFrame([
        {"split": split, **{k: v for k, v in obj.items() if not k.endswith("_examples")}}
        for split, obj in path_checks.items()
    ]).to_string(index=False))

    print("\n=== LEAKAGE CHECK ===")
    print("Group leakage rows:", len(group_leaks))
    print("Pill path leakage rows:", len(pill_path_leaks))
    print("Prescription path leakage rows:", len(pres_path_leaks))

    print("\n=== LABEL COVERAGE ===")
    print("Train labels:", int(dist["in_train"].sum()))
    print("Val labels:", int(dist["in_val"].sum()))
    print("Test labels:", int(dist["in_test"].sum()))
    print("Labels in all splits:", int(dist["in_all_splits"].sum()))
    print("Train labels missing in val:", int(dist["missing_in_val"].sum()))
    print("Train labels missing in test:", int(dist["missing_in_test"].sum()))
    print("Test labels not in train:", int(dist["test_not_in_train"].sum()))

    print("\n=== SUPPORT BINS ===")
    print(bins.to_string(index=False))

    print("\n=== CANDIDATE BENCHMARKS ===")
    print(candidate_df[
        [
            "candidate",
            "num_labels",
            "train_rows",
            "val_rows",
            "test_rows",
            "min_train_count",
            "max_train_count",
        ]
    ].to_string(index=False))

    if optional_metrics:
        print("\n=== OPTIONAL PREDICTION METRICS ===")
        print(pd.DataFrame(optional_metrics)[
            [
                "model",
                "rows",
                "accuracy",
                "macro_f1_present",
                "macro_f1_all",
                "weighted_f1",
            ]
        ].to_string(index=False))

    write_report(
        output_dir=output_dir,
        split_summaries=split_summaries,
        path_checks=path_checks,
        dist=dist,
        bins=bins,
        candidate_df=candidate_df,
        group_leaks=group_leaks,
        pill_path_leaks=pill_path_leaks,
        pres_path_leaks=pres_path_leaks,
        optional_metrics=optional_metrics,
    )

    print("\n=== DONE ===")
    print("Main report:", output_dir / "audit_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit VAIPE clean split and paper-compatible protocol candidates."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--m19_predictions", type=str, default="")
    parser.add_argument("--m22_predictions", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/audit_pika_protocol_v1",
    )

    args = parser.parse_args()
    main(args)

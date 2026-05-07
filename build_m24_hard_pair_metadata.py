import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label_value(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except Exception:
        pass

    try:
        return int(s)
    except Exception:
        return s


def load_label_mapping(graph_artifacts_dir):
    graph_dir = Path(graph_artifacts_dir)

    label_to_idx_path = graph_dir / "label_to_idx.json"
    idx_to_label_path = graph_dir / "idx_to_label.json"

    if not label_to_idx_path.exists() or not idx_to_label_path.exists():
        print("[Warning] label_to_idx.json / idx_to_label.json not found.")
        print("Fallback: assume pill_label is already mapped label.")
        return None, None

    raw_label_to_idx = load_json(label_to_idx_path)
    raw_idx_to_label = load_json(idx_to_label_path)

    label_to_idx = {}
    for k, v in raw_label_to_idx.items():
        label_to_idx[normalize_label_value(k)] = int(v)

    idx_to_label = {}
    for k, v in raw_idx_to_label.items():
        idx_to_label[int(k)] = normalize_label_value(v)

    return label_to_idx, idx_to_label


def add_mapped_label(df, label_to_idx):
    df = df.copy()

    if "pill_label" not in df.columns:
        raise ValueError("CSV must contain pill_label column.")

    if label_to_idx is None:
        df["mapped_label"] = df["pill_label"].apply(normalize_label_value).astype(int)
        df["original_label_norm"] = df["pill_label"].apply(normalize_label_value)
        return df

    def map_label(x):
        x_norm = normalize_label_value(x)

        if x_norm not in label_to_idx:
            raise KeyError(f"Label {x_norm} not found in label_to_idx mapping.")

        return int(label_to_idx[x_norm])

    df["original_label_norm"] = df["pill_label"].apply(normalize_label_value)
    df["mapped_label"] = df["pill_label"].apply(map_label).astype(int)

    return df


def load_split_csv(path, split_name, label_to_idx):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} CSV: {path}")

    df = pd.read_csv(path)
    df = add_mapped_label(df, label_to_idx)
    df["split_name"] = split_name

    if "prescription_key" not in df.columns:
        if "prescription_image_path" in df.columns:
            df["prescription_key"] = df["prescription_image_path"].apply(lambda p: Path(str(p)).stem)
        else:
            df["prescription_key"] = ""

    return df


def load_m22_diagnosis(confusion_pairs_path, per_class_path):
    confusion_pairs_path = Path(confusion_pairs_path)
    per_class_path = Path(per_class_path)

    if not confusion_pairs_path.exists():
        raise FileNotFoundError(f"Missing confusion pairs file: {confusion_pairs_path}")

    if not per_class_path.exists():
        raise FileNotFoundError(f"Missing per-class metrics file: {per_class_path}")

    conf = pd.read_csv(confusion_pairs_path)
    per_class = pd.read_csv(per_class_path)

    # Expected M22 columns:
    # conf: true_label, pred_label, wrong_count, true_support, pred_support
    # per_class: label, support, precision, recall, f1, pred_count, train_count, val_count, test_count, error_count

    required_conf = ["true_label", "pred_label", "wrong_count"]
    missing_conf = [c for c in required_conf if c not in conf.columns]
    if missing_conf:
        raise ValueError(f"Confusion pairs missing columns: {missing_conf}")

    required_pc = ["label", "support", "precision", "recall", "f1"]
    missing_pc = [c for c in required_pc if c not in per_class.columns]
    if missing_pc:
        raise ValueError(f"Per-class metrics missing columns: {missing_pc}")

    conf = conf.copy()
    per_class = per_class.copy()

    conf["true_label"] = conf["true_label"].astype(int)
    conf["pred_label"] = conf["pred_label"].astype(int)
    conf["wrong_count"] = conf["wrong_count"].astype(int)

    per_class["label"] = per_class["label"].astype(int)

    return conf, per_class


def build_hard_pair_edges(conf, min_wrong_count, top_k_per_true):
    conf = conf.copy()
    conf = conf[conf["wrong_count"] >= min_wrong_count].copy()

    conf = conf.sort_values(
        ["true_label", "wrong_count"],
        ascending=[True, False],
    )

    selected_rows = []

    for true_label, group in conf.groupby("true_label"):
        selected_rows.append(group.head(top_k_per_true))

    if selected_rows:
        edges = pd.concat(selected_rows, ignore_index=True)
    else:
        edges = pd.DataFrame(columns=conf.columns)

    edges = edges.sort_values("wrong_count", ascending=False).reset_index(drop=True)

    return edges


def build_hard_negative_map(edges, make_symmetric=True):
    hard_negatives = defaultdict(set)

    for _, row in edges.iterrows():
        t = int(row["true_label"])
        p = int(row["pred_label"])

        if t == p:
            continue

        hard_negatives[t].add(p)

        if make_symmetric:
            hard_negatives[p].add(t)

    hard_negatives = {
        int(k): sorted([int(x) for x in v])
        for k, v in hard_negatives.items()
    }

    return hard_negatives


def build_hard_class_table(
    per_class,
    edges,
    hard_negative_map,
    f1_threshold,
    recall_threshold,
    min_support,
    min_error_count,
    hard_weight_boost,
):
    per_class = per_class.copy()

    # Fill optional columns
    for col in ["error_count", "train_count", "val_count", "test_count", "pred_count"]:
        if col not in per_class.columns:
            per_class[col] = 0

    per_class["support"] = per_class["support"].fillna(0).astype(int)
    per_class["error_count"] = per_class["error_count"].fillna(0).astype(float)
    per_class["f1"] = per_class["f1"].fillna(0.0).astype(float)
    per_class["recall"] = per_class["recall"].fillna(0.0).astype(float)
    per_class["precision"] = per_class["precision"].fillna(0.0).astype(float)

    edge_true_labels = set(edges["true_label"].astype(int).tolist()) if len(edges) else set()
    edge_pred_labels = set(edges["pred_label"].astype(int).tolist()) if len(edges) else set()
    edge_labels = edge_true_labels | edge_pred_labels

    hard_rows = []

    for _, row in per_class.iterrows():
        label = int(row["label"])
        support = int(row["support"])
        f1 = float(row["f1"])
        recall = float(row["recall"])
        error_count = float(row["error_count"])

        reasons = []

        if support >= min_support and f1 <= f1_threshold:
            reasons.append(f"low_f1<={f1_threshold}")

        if support >= min_support and recall <= recall_threshold:
            reasons.append(f"low_recall<={recall_threshold}")

        if error_count >= min_error_count:
            reasons.append(f"high_error_count>={min_error_count}")

        if label in edge_true_labels:
            reasons.append("appears_as_true_in_confusion")

        if label in edge_pred_labels:
            reasons.append("appears_as_pred_in_confusion")

        is_hard = len(reasons) > 0

        # Weight design:
        # - hard class gets boost
        # - lower f1 gets stronger boost
        # - labels that appear in hard-pair graph get extra boost
        difficulty = max(0.0, 1.0 - f1)

        pair_bonus = 0.25 if label in edge_labels else 0.0
        error_bonus = min(0.50, error_count / max(1.0, support) if support > 0 else 0.0)

        hard_weight = 1.0
        if is_hard:
            hard_weight += hard_weight_boost * difficulty
            hard_weight += pair_bonus
            hard_weight += error_bonus

        hard_negs = hard_negative_map.get(label, [])

        hard_rows.append({
            "mapped_label": label,
            "support": support,
            "precision": float(row["precision"]),
            "recall": recall,
            "f1": f1,
            "error_count": error_count,
            "train_count": int(row.get("train_count", 0)),
            "val_count": int(row.get("val_count", 0)),
            "test_count": int(row.get("test_count", 0)),
            "pred_count": int(row.get("pred_count", 0)),
            "is_hard_class": bool(is_hard),
            "hard_class_reason": "|".join(reasons),
            "hard_negative_labels": ",".join([str(x) for x in hard_negs]),
            "num_hard_negatives": int(len(hard_negs)),
            "hard_class_weight": float(hard_weight),
        })

    hard_table = pd.DataFrame(hard_rows)

    hard_table = hard_table.sort_values(
        ["is_hard_class", "hard_class_weight", "error_count", "support"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return hard_table


def attach_hard_metadata(df, hard_table, split_name):
    df = df.copy()

    attach_cols = [
        "mapped_label",
        "precision",
        "recall",
        "f1",
        "error_count",
        "support",
        "is_hard_class",
        "hard_class_reason",
        "hard_negative_labels",
        "num_hard_negatives",
        "hard_class_weight",
    ]

    rename_map = {
        "precision": "m22_precision",
        "recall": "m22_recall",
        "f1": "m22_f1",
        "error_count": "m22_error_count",
        "support": "m22_test_support",
    }

    sub = hard_table[attach_cols].rename(columns=rename_map)

    df = df.merge(
        sub,
        on="mapped_label",
        how="left",
    )

    df["is_hard_class"] = df["is_hard_class"].fillna(False).astype(bool)
    df["hard_class_reason"] = df["hard_class_reason"].fillna("")
    df["hard_negative_labels"] = df["hard_negative_labels"].fillna("")
    df["num_hard_negatives"] = df["num_hard_negatives"].fillna(0).astype(int)
    df["hard_class_weight"] = df["hard_class_weight"].fillna(1.0).astype(float)

    for c in ["m22_precision", "m22_recall", "m22_f1", "m22_error_count", "m22_test_support"]:
        df[c] = df[c].fillna(0.0)

    df["split_name"] = split_name

    return df


def summarize_split_metadata(df, split_name):
    return {
        "split": split_name,
        "rows": int(len(df)),
        "labels": int(df["mapped_label"].nunique()),
        "hard_rows": int(df["is_hard_class"].sum()),
        "hard_labels": int(df.loc[df["is_hard_class"], "mapped_label"].nunique()),
        "mean_hard_weight": float(df["hard_class_weight"].mean()),
        "max_hard_weight": float(df["hard_class_weight"].max()),
    }


def main(args):
    ensure_dir(args.output_dir)

    output_dir = Path(args.output_dir)

    print("=== BUILD M24 HARD PAIR METADATA ===")
    print("Train CSV:", args.train_csv)
    print("Val CSV:", args.val_csv)
    print("Test CSV:", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("M22 confusion pairs:", args.m22_confusion_pairs)
    print("M22 per-class metrics:", args.m22_per_class_metrics)
    print("Output dir:", output_dir)

    label_to_idx, idx_to_label = load_label_mapping(args.graph_artifacts_dir)

    train_df = load_split_csv(args.train_csv, "train", label_to_idx)
    val_df = load_split_csv(args.val_csv, "val", label_to_idx)
    test_df = load_split_csv(args.test_csv, "test", label_to_idx)

    conf, per_class = load_m22_diagnosis(
        confusion_pairs_path=args.m22_confusion_pairs,
        per_class_path=args.m22_per_class_metrics,
    )

    edges = build_hard_pair_edges(
        conf=conf,
        min_wrong_count=args.min_wrong_count,
        top_k_per_true=args.top_k_per_true,
    )

    hard_negative_map = build_hard_negative_map(
        edges=edges,
        make_symmetric=args.make_symmetric,
    )

    hard_table = build_hard_class_table(
        per_class=per_class,
        edges=edges,
        hard_negative_map=hard_negative_map,
        f1_threshold=args.f1_threshold,
        recall_threshold=args.recall_threshold,
        min_support=args.min_support,
        min_error_count=args.min_error_count,
        hard_weight_boost=args.hard_weight_boost,
    )

    train_meta = attach_hard_metadata(train_df, hard_table, "train")
    val_meta = attach_hard_metadata(val_df, hard_table, "val")
    test_meta = attach_hard_metadata(test_df, hard_table, "test")

    train_out = output_dir / "m24_train_hard_metadata.csv"
    val_out = output_dir / "m24_val_hard_metadata.csv"
    test_out = output_dir / "m24_test_hard_metadata.csv"

    train_meta.to_csv(train_out, index=False)
    val_meta.to_csv(val_out, index=False)
    test_meta.to_csv(test_out, index=False)

    edges_out = output_dir / "m24_hard_pair_edges.csv"
    hard_table_out = output_dir / "m24_hard_class_summary.csv"
    hard_neg_out = output_dir / "m24_hard_negative_map.json"

    edges.to_csv(edges_out, index=False)
    hard_table.to_csv(hard_table_out, index=False)
    save_json({str(k): v for k, v in hard_negative_map.items()}, hard_neg_out)

    split_summary = [
        summarize_split_metadata(train_meta, "train"),
        summarize_split_metadata(val_meta, "val"),
        summarize_split_metadata(test_meta, "test"),
    ]

    split_summary_df = pd.DataFrame(split_summary)
    split_summary_df.to_csv(output_dir / "m24_hard_metadata_split_summary.csv", index=False)

    config = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "graph_artifacts_dir": args.graph_artifacts_dir,
        "m22_confusion_pairs": args.m22_confusion_pairs,
        "m22_per_class_metrics": args.m22_per_class_metrics,
        "min_wrong_count": args.min_wrong_count,
        "top_k_per_true": args.top_k_per_true,
        "make_symmetric": args.make_symmetric,
        "f1_threshold": args.f1_threshold,
        "recall_threshold": args.recall_threshold,
        "min_support": args.min_support,
        "min_error_count": args.min_error_count,
        "hard_weight_boost": args.hard_weight_boost,
    }

    save_json(config, output_dir / "m24_hard_metadata_config.json")

    print("\n=== HARD PAIR EDGES TOP 30 ===")
    if len(edges) > 0:
        print(edges.head(30).to_string(index=False))
    else:
        print("No hard pair edges selected.")

    print("\n=== HARD CLASS SUMMARY TOP 40 ===")
    print(
        hard_table[
            [
                "mapped_label",
                "support",
                "precision",
                "recall",
                "f1",
                "error_count",
                "is_hard_class",
                "hard_class_reason",
                "hard_negative_labels",
                "num_hard_negatives",
                "hard_class_weight",
            ]
        ].head(40).to_string(index=False)
    )

    print("\n=== SPLIT SUMMARY ===")
    print(split_summary_df.to_string(index=False))

    print("\nSaved files:")
    print(train_out)
    print(val_out)
    print(test_out)
    print(edges_out)
    print(hard_table_out)
    print(hard_neg_out)
    print(output_dir / "m24_hard_metadata_split_summary.csv")
    print(output_dir / "m24_hard_metadata_config.json")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build M24 hard-pair metadata from M22 error diagnosis."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument("--m22_confusion_pairs", type=str, required=True)
    parser.add_argument("--m22_per_class_metrics", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M24_metric_learning/hard_pair_metadata",
    )

    parser.add_argument("--min_wrong_count", type=int, default=5)
    parser.add_argument("--top_k_per_true", type=int, default=3)
    parser.add_argument("--make_symmetric", action="store_true")

    parser.add_argument("--f1_threshold", type=float, default=0.45)
    parser.add_argument("--recall_threshold", type=float, default=0.35)
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--min_error_count", type=int, default=7)
    parser.add_argument("--hard_weight_boost", type=float, default=1.5)

    args = parser.parse_args()
    main(args)

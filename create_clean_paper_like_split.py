import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_with_source(path, source_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df["source_split"] = source_name
    return df


def parse_ratio_text(text):
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("split_ratios must have exactly 3 values: train,val,test")

    total = sum(parts)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")

    return {
        "train": parts[0] / total,
        "val": parts[1] / total,
        "test": parts[2] / total,
    }


def build_class_targets(total_counts, ratios):
    """
    Soft per-class targets.
    Main priority is still group-aware split and full train coverage.
    """
    targets = {"train": {}, "val": {}, "test": {}}

    for label, total in total_counts.items():
        total = int(total)

        if total <= 0:
            targets["train"][label] = 0
            targets["val"][label] = 0
            targets["test"][label] = 0

        elif total == 1:
            targets["train"][label] = 1
            targets["val"][label] = 0
            targets["test"][label] = 0

        elif total == 2:
            targets["train"][label] = 1
            targets["val"][label] = 1
            targets["test"][label] = 0

        else:
            val_target = max(1, int(round(total * ratios["val"])))
            test_target = max(1, int(round(total * ratios["test"])))

            if val_target + test_target >= total:
                val_target = 1
                test_target = 1

            train_target = total - val_target - test_target
            if train_target < 1:
                train_target = 1
                if val_target >= test_target and val_target > 1:
                    val_target -= 1
                elif test_target > 1:
                    test_target -= 1

            targets["train"][label] = int(train_target)
            targets["val"][label] = int(val_target)
            targets["test"][label] = int(test_target)

    return targets


def build_group_records(df, label_col, group_col, total_counts, seed=42):
    rng = random.Random(seed)
    group_records = []

    for group_id, g in df.groupby(group_col, sort=False):
        labels = g[label_col].astype(int).tolist()
        label_counts = Counter(labels)
        unique_labels = sorted(label_counts.keys())

        rarity_score = 0.0
        for lb, c in label_counts.items():
            rarity_score += float(c) / np.sqrt(max(1.0, float(total_counts[lb])))

        group_records.append({
            "group_id": group_id,
            "row_indices": g.index.tolist(),
            "num_rows": len(g),
            "labels": unique_labels,
            "label_counts": dict(label_counts),
            "num_unique_labels": len(unique_labels),
            "rarity_score": rarity_score,
            "rand": rng.random(),
        })

    return group_records


def summarize_split(df, label_col, split_col, group_col):
    rows = []
    all_labels = sorted(df[label_col].astype(int).unique().tolist())

    for split in ["train", "val", "test"]:
        part = df[df[split_col] == split]

        label_counts = part[label_col].astype(int).value_counts().sort_index()
        labels_present = set(label_counts.index.astype(int).tolist())
        missing = sorted(list(set(all_labels) - labels_present))

        rows.append({
            "split": split,
            "rows": int(len(part)),
            "groups": int(part[group_col].nunique()),
            "labels_present": int(len(labels_present)),
            "labels_missing": int(len(missing)),
            "missing_labels": missing,
            "min_class_count": int(label_counts.min()) if len(label_counts) > 0 else 0,
            "max_class_count": int(label_counts.max()) if len(label_counts) > 0 else 0,
        })

    return pd.DataFrame(rows)


def build_class_distribution(df, label_col, split_col):
    labels = sorted(df[label_col].astype(int).unique().tolist())
    rows = []

    for label in labels:
        row = {"pill_label": int(label)}
        for split in ["train", "val", "test"]:
            cnt = int(((df[split_col] == split) & (df[label_col].astype(int) == label)).sum())
            row[f"{split}_count"] = cnt
        row["total"] = row["train_count"] + row["val_count"] + row["test_count"]
        rows.append(row)

    return pd.DataFrame(rows)


def check_group_leakage(df, group_col, split_col):
    bad_groups = []

    for group_id, g in df.groupby(group_col):
        splits = sorted(g[split_col].dropna().unique().tolist())
        if len(splits) > 1:
            bad_groups.append({
                "group_id": group_id,
                "splits": ",".join(splits),
                "rows": len(g),
            })

    # Keep header even if empty
    return pd.DataFrame(bad_groups, columns=["group_id", "splits", "rows"])


def create_clean_split(df, label_col, group_col, ratios, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    total_counts = Counter(df[label_col].astype(int).tolist())
    class_targets = build_class_targets(total_counts, ratios)

    group_records = build_group_records(
        df=df,
        label_col=label_col,
        group_col=group_col,
        total_counts=total_counts,
        seed=seed,
    )

    group_records = sorted(
        group_records,
        key=lambda x: (-x["rarity_score"], -x["num_unique_labels"], x["num_rows"], x["rand"])
    )

    total_rows = len(df)
    row_targets = {
        "train": int(round(total_rows * ratios["train"])),
        "val": int(round(total_rows * ratios["val"])),
        "test": total_rows - int(round(total_rows * ratios["train"])) - int(round(total_rows * ratios["val"])),
    }

    assigned_group_to_split = {}
    split_rows = {"train": 0, "val": 0, "test": 0}
    split_class_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    label_to_groups = defaultdict(list)
    for rec in group_records:
        for lb in rec["labels"]:
            label_to_groups[lb].append(rec)

    def assign_group(rec, split):
        gid = rec["group_id"]
        if gid in assigned_group_to_split:
            return

        assigned_group_to_split[gid] = split
        split_rows[split] += rec["num_rows"]

        for lb, cnt in rec["label_counts"].items():
            split_class_counts[split][int(lb)] += int(cnt)

    # ------------------------------------------------------------------
    # Phase A: Force train to cover all classes if possible
    # ------------------------------------------------------------------
    labels_by_rarity = sorted(total_counts.keys(), key=lambda lb: (total_counts[lb], lb))

    for lb in labels_by_rarity:
        if split_class_counts["train"][lb] > 0:
            continue

        candidates = [rec for rec in label_to_groups[lb] if rec["group_id"] not in assigned_group_to_split]
        if not candidates:
            continue

        def anchor_train_score(rec):
            uncovered_bonus = 0.0
            for x in rec["labels"]:
                if split_class_counts["train"][x] == 0:
                    uncovered_bonus += 1.0 / np.sqrt(max(1.0, float(total_counts[x])))

            size_penalty = 0.002 * rec["num_rows"]
            over_penalty = max(
                0.0,
                (split_rows["train"] + rec["num_rows"] - row_targets["train"]) / max(1.0, row_targets["train"])
            )
            return uncovered_bonus - size_penalty - 0.50 * over_penalty

        best_rec = max(candidates, key=anchor_train_score)
        assign_group(best_rec, "train")

    # ------------------------------------------------------------------
    # Phase B: Greedy assign remaining groups
    # ------------------------------------------------------------------
    remaining_groups = [rec for rec in group_records if rec["group_id"] not in assigned_group_to_split]

    def score_group_for_split(rec, split):
        size = rec["num_rows"]
        new_rows = split_rows[split] + size
        target_rows = row_targets[split]

        # Reward useful class coverage according to soft class targets
        class_gain = 0.0
        for lb, cnt in rec["label_counts"].items():
            current = split_class_counts[split][lb]
            target = class_targets[split].get(lb, 0)
            need = max(0, target - current)
            useful = min(cnt, need)
            rarity = 1.0 / np.sqrt(max(1.0, float(total_counts[lb])))

            class_gain += useful * rarity

            if current == 0 and target > 0:
                class_gain += 0.20 * rarity

        # Strong preference to keep train row count healthy but not too oversized
        balance_penalty = abs(new_rows - target_rows) / max(1.0, float(target_rows))
        over_penalty = max(0.0, new_rows - target_rows) / max(1.0, float(target_rows))

        # Tiny stability bonus to train for very rare labels
        train_rare_bonus = 0.0
        if split == "train":
            for lb in rec["labels"]:
                if total_counts[lb] <= 2:
                    train_rare_bonus += 0.08

        score = (
            class_gain
            + train_rare_bonus
            - 0.35 * balance_penalty
            - 1.20 * over_penalty
        )

        return score

    for rec in remaining_groups:
        best_split = None
        best_score = None

        for split in ["train", "val", "test"]:
            sc = score_group_for_split(rec, split)
            if best_score is None or sc > best_score:
                best_score = sc
                best_split = split

        assign_group(rec, best_split)

    # ------------------------------------------------------------------
    # Build final dataframe
    # ------------------------------------------------------------------
    out_df = df.copy()
    out_df["clean_split"] = out_df[group_col].map(assigned_group_to_split)

    if out_df["clean_split"].isna().any():
        raise RuntimeError("Some groups were not assigned to a split.")

    return out_df, row_targets, class_targets


def save_outputs(df, output_dir, label_col, group_col):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    split_col = "clean_split"

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_path = output_dir / "train_clean.csv"
    val_path = output_dir / "val_clean.csv"
    test_path = output_dir / "test_clean.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary_df = summarize_split(df, label_col=label_col, split_col=split_col, group_col=group_col)
    summary_path = output_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    class_dist_df = build_class_distribution(df, label_col=label_col, split_col=split_col)
    class_dist_path = output_dir / "class_distribution_clean.csv"
    class_dist_df.to_csv(class_dist_path, index=False)

    leakage_df = check_group_leakage(df, group_col=group_col, split_col=split_col)
    leakage_path = output_dir / "group_leakage_check.csv"
    leakage_df.to_csv(leakage_path, index=False)

    full_meta_path = output_dir / "full_metadata_with_clean_split.csv"
    df.to_csv(full_meta_path, index=False)

    print("\nSaved files:")
    print("Train clean :", train_path)
    print("Val clean   :", val_path)
    print("Test clean  :", test_path)
    print("Summary     :", summary_path)
    print("Class dist  :", class_dist_path)
    print("Leakage chk :", leakage_path)
    print("Full meta   :", full_meta_path)


def main(args):
    ratios = parse_ratio_text(args.split_ratios)
    ensure_dir(args.output_dir)

    print("=== CREATE CLEAN PAPER-LIKE SPLIT V2 ===")
    print("Old train CSV:", args.old_train_csv)
    print("Old val CSV  :", args.old_val_csv)
    print("Old test CSV :", args.old_test_csv)
    print("Output dir   :", args.output_dir)
    print("Ratios       :", ratios)
    print("Label col    :", args.label_col)
    print("Group col    :", args.group_col)
    print("Seed         :", args.seed)
    print("Drop duplicate pill_crop_path:", args.drop_duplicate_pill_crop_path)

    train_df = read_csv_with_source(args.old_train_csv, "old_train")
    val_df = read_csv_with_source(args.old_val_csv, "old_val")
    test_df = read_csv_with_source(args.old_test_csv, "old_test")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if args.label_col not in full_df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in dataframe.")

    full_df[args.label_col] = full_df[args.label_col].astype(int)

    if args.group_col not in full_df.columns:
        print(f"Warning: group_col '{args.group_col}' not found. Fallback to row-level groups.")
        full_df[args.group_col] = [f"row_{i}" for i in range(len(full_df))]

    print("\nBefore duplicate handling:")
    print("Total rows:", len(full_df))
    print("Total groups:", full_df[args.group_col].nunique())
    print("Total labels:", full_df[args.label_col].nunique())

    if args.drop_duplicate_pill_crop_path:
        if "pill_crop_path" not in full_df.columns:
            raise ValueError("pill_crop_path not found but --drop_duplicate_pill_crop_path was used.")
        before = len(full_df)
        full_df = full_df.drop_duplicates(subset=["pill_crop_path"]).reset_index(drop=True)
        print(f"Dropped duplicate rows by pill_crop_path: {before - len(full_df)}")

    print("\nAfter duplicate handling:")
    print("Total rows:", len(full_df))
    print("Total groups:", full_df[args.group_col].nunique())
    print("Total labels:", full_df[args.label_col].nunique())

    out_df, row_targets, class_targets = create_clean_split(
        df=full_df,
        label_col=args.label_col,
        group_col=args.group_col,
        ratios=ratios,
        seed=args.seed,
    )

    print("\n=== CLEAN SPLIT SUMMARY ===")
    summary_df = summarize_split(
        out_df,
        label_col=args.label_col,
        split_col="clean_split",
        group_col=args.group_col,
    )
    print(summary_df.to_string(index=False))

    class_dist_df = build_class_distribution(
        out_df,
        label_col=args.label_col,
        split_col="clean_split",
    )

    print("\nRare classes after clean split:")
    print(class_dist_df.sort_values("total").head(30).to_string(index=False))

    print("\nMost frequent classes after clean split:")
    print(class_dist_df.sort_values("total", ascending=False).head(20).to_string(index=False))

    leakage_df = check_group_leakage(
        out_df,
        group_col=args.group_col,
        split_col="clean_split",
    )
    print("\nGroup leakage rows:", len(leakage_df))

    config = {
        "old_train_csv": args.old_train_csv,
        "old_val_csv": args.old_val_csv,
        "old_test_csv": args.old_test_csv,
        "output_dir": args.output_dir,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "split_ratios": args.split_ratios,
        "seed": args.seed,
        "drop_duplicate_pill_crop_path": args.drop_duplicate_pill_crop_path,
        "row_targets": row_targets,
        "note": "Train coverage is anchored first so train should include all classes that are coverable under group constraints.",
    }

    config_path = Path(args.output_dir) / "split_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    save_outputs(
        df=out_df,
        output_dir=args.output_dir,
        label_col=args.label_col,
        group_col=args.group_col,
    )

    print("Config      :", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create clean paper-like split v2 with full train coverage and group-aware splitting."
    )

    parser.add_argument(
        "--old_train_csv",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/best_pika_split_metadata/train.csv",
    )
    parser.add_argument(
        "--old_val_csv",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/best_pika_split_metadata/val.csv",
    )
    parser.add_argument(
        "--old_test_csv",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/best_pika_split_metadata/test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2",
    )
    parser.add_argument(
        "--split_ratios",
        type=str,
        default="0.70,0.15,0.15",
    )
    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--group_col", type=str, default="prescription_key")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--drop_duplicate_pill_crop_path",
        action="store_true",
        help="Optional. Default is OFF.",
    )

    args = parser.parse_args()
    main(args)

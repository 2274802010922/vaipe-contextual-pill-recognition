import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

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


def make_group_id(df, group_col):
    if group_col in df.columns:
        group = df[group_col].astype(str).fillna("missing_group")
    else:
        print(f"Warning: group_col '{group_col}' not found. Falling back to row-level groups.")
        group = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)

    return group


def build_group_table(df, label_col, group_col):
    group_rows = []

    for group_id, g in df.groupby(group_col):
        labels = g[label_col].astype(int).tolist()
        label_counts = Counter(labels)

        group_rows.append(
            {
                "group_id": group_id,
                "num_rows": len(g),
                "labels": sorted(label_counts.keys()),
                "label_counts": dict(label_counts),
                "num_unique_labels": len(label_counts),
            }
        )

    group_df = pd.DataFrame(group_rows)

    return group_df


def compute_class_targets(total_counts, ratios):
    """
    Build per-class target counts for train / val / test.

    Rules:
    - class with 1 sample: train only
    - class with 2 samples: train + val
    - class with >=3 samples: at least 1 in train, val, test if possible
    """
    targets = {
        "train": {},
        "val": {},
        "test": {},
    }

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


def label_rarity_score(label_counts, total_counts):
    score = 0.0

    for label, count in label_counts.items():
        total = max(1, total_counts[int(label)])
        score += float(count) / np.sqrt(float(total))

    return score


def choose_split_for_group(
    label_counts,
    group_size,
    split_counts,
    split_class_counts,
    split_targets,
    class_targets,
    total_counts,
    ratios,
    balance_weight=0.30,
    over_weight=1.50,
):
    best_split = None
    best_score = None

    for split in ["train", "val", "test"]:
        current_total = split_counts[split]
        target_total = split_targets[split]
        new_total = current_total + group_size

        class_benefit = 0.0

        for raw_label, raw_count in label_counts.items():
            label = int(raw_label)
            count = int(raw_count)

            target_for_class = class_targets[split].get(label, 0)
            current_for_class = split_class_counts[split].get(label, 0)

            remaining_need = max(0, target_for_class - current_for_class)
            useful = min(count, remaining_need)

            rarity = 1.0 / np.sqrt(max(1.0, float(total_counts[label])))
            class_benefit += useful * rarity

            # small bonus if this split has not seen this class yet and target wants it
            if current_for_class == 0 and target_for_class > 0:
                class_benefit += 0.25 * rarity

        balance_penalty = abs(new_total - target_total) / max(1.0, float(target_total))
        over_penalty = max(0, new_total - target_total) / max(1.0, float(target_total))

        # train split gets a tiny stability bonus for very rare labels
        rare_bonus = 0.0
        if split == "train":
            for raw_label in label_counts.keys():
                label = int(raw_label)
                if total_counts[label] <= 2:
                    rare_bonus += 0.10

        score = (
            class_benefit
            + rare_bonus
            - balance_weight * balance_penalty
            - over_weight * over_penalty
        )

        if best_score is None or score > best_score:
            best_score = score
            best_split = split

    return best_split


def greedy_group_stratified_split(df, label_col, group_col, ratios, seed):
    rng = np.random.default_rng(seed)

    total_rows = len(df)

    split_targets = {
        "train": int(round(total_rows * ratios["train"])),
        "val": int(round(total_rows * ratios["val"])),
        "test": total_rows - int(round(total_rows * ratios["train"])) - int(round(total_rows * ratios["val"])),
    }

    total_counts = Counter(df[label_col].astype(int).tolist())
    class_targets = compute_class_targets(total_counts, ratios)

    group_df = build_group_table(df, label_col=label_col, group_col=group_col)

    group_df["rarity_score"] = group_df["label_counts"].apply(
        lambda x: label_rarity_score(x, total_counts)
    )

    group_df["random_jitter"] = rng.random(len(group_df))

    # Process rare and multi-label groups first.
    group_df = group_df.sort_values(
        by=["rarity_score", "num_unique_labels", "num_rows", "random_jitter"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    split_class_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    group_to_split = {}

    for _, row in group_df.iterrows():
        group_id = row["group_id"]
        group_size = int(row["num_rows"])
        label_counts = row["label_counts"]

        chosen_split = choose_split_for_group(
            label_counts=label_counts,
            group_size=group_size,
            split_counts=split_counts,
            split_class_counts=split_class_counts,
            split_targets=split_targets,
            class_targets=class_targets,
            total_counts=total_counts,
            ratios=ratios,
        )

        group_to_split[group_id] = chosen_split
        split_counts[chosen_split] += group_size

        for label, count in label_counts.items():
            split_class_counts[chosen_split][int(label)] += int(count)

    out_df = df.copy()
    out_df["clean_split"] = out_df[group_col].map(group_to_split)

    return out_df, split_targets, class_targets


def summarize_split(df, label_col, split_col, group_col):
    rows = []

    all_labels = sorted(df[label_col].astype(int).unique().tolist())

    for split in ["train", "val", "test"]:
        part = df[df[split_col] == split]

        label_counts = part[label_col].astype(int).value_counts().sort_index()
        labels_present = set(label_counts.index.astype(int).tolist())

        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "groups": int(part[group_col].nunique()),
                "labels_present": int(len(labels_present)),
                "labels_missing": int(len(set(all_labels) - labels_present)),
                "missing_labels": sorted(list(set(all_labels) - labels_present)),
                "min_class_count": int(label_counts.min()) if len(label_counts) > 0 else 0,
                "max_class_count": int(label_counts.max()) if len(label_counts) > 0 else 0,
            }
        )

    return pd.DataFrame(rows)


def build_class_distribution(df, label_col, split_col):
    labels = sorted(df[label_col].astype(int).unique().tolist())

    rows = []

    for label in labels:
        row = {"pill_label": int(label)}

        for split in ["train", "val", "test"]:
            count = int(((df[split_col] == split) & (df[label_col].astype(int) == label)).sum())
            row[f"{split}_count"] = count

        row["total"] = row["train_count"] + row["val_count"] + row["test_count"]
        rows.append(row)

    return pd.DataFrame(rows)


def check_group_leakage(df, group_col, split_col):
    bad_groups = []

    for group_id, g in df.groupby(group_col):
        splits = sorted(g[split_col].dropna().unique().tolist())

        if len(splits) > 1:
            bad_groups.append(
                {
                    "group_id": group_id,
                    "splits": ",".join(splits),
                    "rows": len(g),
                }
            )

    return pd.DataFrame(bad_groups)


def save_outputs(df, output_dir, label_col, group_col, split_col, split_targets, class_targets, args):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    train_df = df[df[split_col] == "train"].drop(columns=[split_col], errors="ignore").copy()
    val_df = df[df[split_col] == "val"].drop(columns=[split_col], errors="ignore").copy()
    test_df = df[df[split_col] == "test"].drop(columns=[split_col], errors="ignore").copy()

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

    full_metadata_path = output_dir / "full_metadata_with_clean_split.csv"
    df.to_csv(full_metadata_path, index=False)

    config = {
        "old_train_csv": args.old_train_csv,
        "old_val_csv": args.old_val_csv,
        "old_test_csv": args.old_test_csv,
        "output_dir": str(output_dir),
        "label_col": label_col,
        "group_col": group_col,
        "split_col": split_col,
        "split_ratios": args.split_ratios,
        "seed": args.seed,
        "drop_duplicate_pill_crop_path": args.drop_duplicate_pill_crop_path,
        "split_targets": split_targets,
        "class_targets_note": "Per-class targets were used only as soft targets due to group-level splitting.",
        "num_rows_total": int(len(df)),
        "num_groups_total": int(df[group_col].nunique()),
    }

    config_path = output_dir / "split_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print("Train clean :", train_path)
    print("Val clean   :", val_path)
    print("Test clean  :", test_path)
    print("Summary     :", summary_path)
    print("Class dist  :", class_dist_path)
    print("Leakage chk :", leakage_path)
    print("Full metadata:", full_metadata_path)
    print("Config      :", config_path)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "summary_path": summary_path,
        "class_dist_path": class_dist_path,
        "leakage_path": leakage_path,
        "config_path": config_path,
    }


def main(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    ratios = parse_ratio_text(args.split_ratios)

    print("=== CREATE CLEAN PAPER-LIKE SPLIT ===")
    print("Old train CSV:", args.old_train_csv)
    print("Old val CSV  :", args.old_val_csv)
    print("Old test CSV :", args.old_test_csv)
    print("Output dir   :", output_dir)
    print("Ratios       :", ratios)
    print("Label col    :", args.label_col)
    print("Group col    :", args.group_col)
    print("Seed         :", args.seed)

    train_df = read_csv_with_source(args.old_train_csv, "old_train")
    val_df = read_csv_with_source(args.old_val_csv, "old_val")
    test_df = read_csv_with_source(args.old_test_csv, "old_test")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("\nBefore duplicate handling:")
    print("Total rows:", len(full_df))
    print("Columns:", full_df.columns.tolist())

    if args.label_col not in full_df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in metadata.")

    full_df[args.label_col] = full_df[args.label_col].astype(int)

    if args.drop_duplicate_pill_crop_path:
        if "pill_crop_path" in full_df.columns:
            before = len(full_df)
            full_df = full_df.drop_duplicates(subset=["pill_crop_path"]).reset_index(drop=True)
            after = len(full_df)
            print(f"\nDropped duplicate pill_crop_path rows: {before - after}")
        else:
            print("\nWarning: pill_crop_path column not found. Cannot drop duplicates by pill_crop_path.")

    full_df = full_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    internal_group_col = "__clean_group_id__"
    full_df[internal_group_col] = make_group_id(full_df, args.group_col)

    print("\nAfter duplicate handling:")
    print("Total rows:", len(full_df))
    print("Total groups:", full_df[internal_group_col].nunique())
    print("Total labels:", full_df[args.label_col].nunique())

    print("\nOriginal source split counts:")
    print(full_df["source_split"].value_counts())

    print("\nOriginal label distribution top 20:")
    print(full_df[args.label_col].value_counts().head(20))

    clean_df, split_targets, class_targets = greedy_group_stratified_split(
        df=full_df,
        label_col=args.label_col,
        group_col=internal_group_col,
        ratios=ratios,
        seed=args.seed,
    )

    clean_df = clean_df.rename(columns={"clean_split": "__clean_split__"})

    print("\n=== CLEAN SPLIT SUMMARY ===")
    summary_df = summarize_split(
        clean_df,
        label_col=args.label_col,
        split_col="__clean_split__",
        group_col=internal_group_col,
    )

    print(summary_df.to_string(index=False))

    leakage_df = check_group_leakage(
        clean_df,
        group_col=internal_group_col,
        split_col="__clean_split__",
    )

    print("\nGroup leakage rows:", len(leakage_df))

    class_dist_df = build_class_distribution(
        clean_df,
        label_col=args.label_col,
        split_col="__clean_split__",
    )

    print("\nRare classes after clean split:")
    print(class_dist_df.sort_values("total").head(30).to_string(index=False))

    print("\nMost frequent classes after clean split:")
    print(class_dist_df.sort_values("total", ascending=False).head(20).to_string(index=False))

    # Remove internal group column before saving split CSVs if user does not want it.
    if args.keep_internal_group_col:
        save_df = clean_df.copy()
    else:
        save_df = clean_df.drop(columns=[internal_group_col], errors="ignore").copy()

    save_outputs(
        df=save_df,
        output_dir=output_dir,
        label_col=args.label_col,
        group_col=args.group_col if args.group_col in save_df.columns else "prescription_key",
        split_col="__clean_split__",
        split_targets=split_targets,
        class_targets=class_targets,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a clean paper-like group-aware stratified split for VAIPE Best PIKA metadata."
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
        default="/content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v1",
    )

    parser.add_argument(
        "--split_ratios",
        type=str,
        default="0.70,0.15,0.15",
        help="Train,val,test ratios. Default is 70/15/15.",
    )

    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--group_col", type=str, default="prescription_key")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--drop_duplicate_pill_crop_path",
        action="store_true",
        help="Drop duplicate rows by pill_crop_path before splitting.",
    )

    parser.add_argument(
        "--keep_internal_group_col",
        action="store_true",
        help="Keep internal group id column in saved CSVs.",
    )

    args = parser.parse_args()
    main(args)

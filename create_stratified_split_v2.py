import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a stratified train/val split for VAIPE Best PIKA metadata."
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
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/vaipe_splits/best_pika_stratified_v2",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="pill_label",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def create_split(
    old_train_csv: str,
    old_val_csv: str,
    output_dir: str,
    label_col: str = "pill_label",
    val_ratio: float = 0.15,
    seed: int = 42,
):
    old_train_csv = Path(old_train_csv)
    old_val_csv = Path(old_val_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Old train CSV:", old_train_csv)
    print("Old val CSV  :", old_val_csv)
    print("Output dir   :", output_dir)
    print("Label column :", label_col)
    print("Val ratio    :", val_ratio)
    print("Seed         :", seed)

    if not old_train_csv.exists():
        raise FileNotFoundError(f"Old train CSV not found: {old_train_csv}")

    if not old_val_csv.exists():
        raise FileNotFoundError(f"Old val CSV not found: {old_val_csv}")

    old_train_df = pd.read_csv(old_train_csv)
    old_val_df = pd.read_csv(old_val_csv)

    print("\nOld train shape:", old_train_df.shape)
    print("Old val shape  :", old_val_df.shape)

    if label_col not in old_train_df.columns:
        raise ValueError(f"Column '{label_col}' not found in old train CSV.")

    if label_col not in old_val_df.columns:
        raise ValueError(f"Column '{label_col}' not found in old val CSV.")

    all_df = pd.concat([old_train_df, old_val_df], ignore_index=True)
    all_df = all_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    new_train_parts = []
    new_val_parts = []

    for label, group in all_df.groupby(label_col):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)

        if n == 1:
            # Cannot put this label into both train and validation.
            new_train_parts.append(group)
        elif n <= 5:
            # Very rare class: use exactly one sample for validation.
            new_val_parts.append(group.iloc[:1])
            new_train_parts.append(group.iloc[1:])
        else:
            # Normal class: use val_ratio, but keep at least one validation sample.
            val_n = max(1, int(round(n * val_ratio)))
            new_val_parts.append(group.iloc[:val_n])
            new_train_parts.append(group.iloc[val_n:])

    new_train_df = pd.concat(new_train_parts, ignore_index=True)
    new_val_df = pd.concat(new_val_parts, ignore_index=True)

    new_train_df = new_train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    new_val_df = new_val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    new_train_df["split"] = "train"
    new_val_df["split"] = "val"

    new_train_path = output_dir / "train.csv"
    new_val_path = output_dir / "val.csv"
    summary_path = output_dir / "class_distribution_summary.csv"

    new_train_df.to_csv(new_train_path, index=False)
    new_val_df.to_csv(new_val_path, index=False)

    train_counts = new_train_df[label_col].value_counts().sort_index()
    val_counts = new_val_df[label_col].value_counts().sort_index()

    summary_df = pd.DataFrame(
        {
            "train_count": train_counts,
            "val_count": val_counts,
        }
    ).fillna(0).astype(int)

    summary_df["total"] = summary_df["train_count"] + summary_df["val_count"]
    summary_df.to_csv(summary_path)

    train_labels = set(new_train_df[label_col].unique())
    val_labels = set(new_val_df[label_col].unique())

    missing_in_val = sorted(list(train_labels - val_labels))
    missing_in_train = sorted(list(val_labels - train_labels))

    print("\nSaved:")
    print("New train CSV:", new_train_path)
    print("New val CSV  :", new_val_path)
    print("Summary CSV  :", summary_path)

    print("\nNew train shape:", new_train_df.shape)
    print("New val shape  :", new_val_df.shape)

    print("\nNum train labels:", len(train_labels))
    print("Num val labels  :", len(val_labels))

    print("\nLabels in train but missing in val:", len(missing_in_val))
    print(missing_in_val)

    print("\nLabels in val but missing in train:", len(missing_in_train))
    print(missing_in_train)

    print("\nTrain count min/max:")
    print(summary_df["train_count"].min(), summary_df["train_count"].max())

    print("\nVal count min/max:")
    print(summary_df["val_count"].min(), summary_df["val_count"].max())

    print("\nRare classes after split:")
    print(summary_df.sort_values("train_count").head(30))

    print("\nMost frequent classes after split:")
    print(summary_df.sort_values("train_count", ascending=False).head(20))

    return new_train_path, new_val_path, summary_path


def main():
    args = parse_args()

    create_split(
        old_train_csv=args.old_train_csv,
        old_val_csv=args.old_val_csv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

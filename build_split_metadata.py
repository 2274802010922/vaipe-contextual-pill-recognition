import os
import argparse
import pandas as pd


def stem_no_ext(x: str) -> str:
    return os.path.splitext(os.path.basename(str(x)))[0]


def main(args):
    metadata_csv = args.metadata_csv
    split_map_csv = args.split_map_csv
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    split_df = pd.read_csv(split_map_csv)

    if "prescription_json" not in df.columns:
        raise RuntimeError(
            f"File {metadata_csv} không có cột 'prescription_json'. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    if "prescription_json" not in split_df.columns or "split" not in split_df.columns:
        raise RuntimeError(
            f"File {split_map_csv} phải có cột 'prescription_json' và 'split'. "
            f"Các cột hiện có: {split_df.columns.tolist()}"
        )

    df["prescription_key"] = df["prescription_json"].astype(str).apply(stem_no_ext)
    split_df["prescription_key"] = split_df["prescription_json"].astype(str).apply(stem_no_ext)

    key_to_split = dict(zip(split_df["prescription_key"], split_df["split"]))
    df["split"] = df["prescription_key"].map(key_to_split)

    missing = df["split"].isna().sum()
    print("Total rows:", len(df))
    print("Matched split rows:", len(df) - missing)
    print("Missing split rows:", missing)

    if missing > 0:
        print("\nSample missing prescription_key:")
        print(df[df["split"].isna()]["prescription_key"].drop_duplicates().head(20).tolist())

    df = df[df["split"].notna()].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    full_path = os.path.join(output_dir, "full_with_split.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    df.to_csv(full_path, index=False, encoding="utf-8")

    print("\nSaved:")
    print("-", train_path, "| rows:", len(train_df))
    print("-", val_path, "| rows:", len(val_df))
    print("-", test_path, "| rows:", len(test_df))
    print("-", full_path, "| rows:", len(df))

    print("\nSplit distribution:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split metadata CSV by prescription-level split map")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV gốc")
    parser.add_argument("--split_map_csv", type=str, required=True, help="prescription_split_map.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục output")
    args = parser.parse_args()
    main(args)

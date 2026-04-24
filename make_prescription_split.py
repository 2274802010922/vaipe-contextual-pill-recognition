import os
import json
import math
import random
import argparse
from collections import Counter
from typing import Dict, List

import pandas as pd


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stem_no_ext(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_public_train_map(train_root: str) -> List[Dict]:
    map_path = os.path.join(train_root, "pill_pres_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Không tìm thấy pill_pres_map.json tại: {map_path}")

    data = load_json(map_path)
    if not isinstance(data, list):
        raise RuntimeError("pill_pres_map.json của public_train không ở dạng list như mong đợi.")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pres = item.get("pres")
        pill_list = item.get("pill", [])

        if not pres or not isinstance(pill_list, list):
            continue

        cleaned.append(
            {
                "pres": str(pres),
                "pres_key": stem_no_ext(pres),
                "pill_list": [str(x) for x in pill_list],
                "num_pills": len(pill_list),
            }
        )

    if len(cleaned) == 0:
        raise RuntimeError("Không đọc được prescription entries nào từ pill_pres_map.json")

    return cleaned


def split_prescriptions(
    prescriptions: List[Dict],
    seed: int,
    test_ratio: float,
    val_ratio_within_trainval: float,
):
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio phải nằm trong (0, 1)")
    if not (0.0 < val_ratio_within_trainval < 1.0):
        raise ValueError("val_ratio_within_trainval phải nằm trong (0, 1)")

    rng = random.Random(seed)
    pres_copy = prescriptions.copy()
    rng.shuffle(pres_copy)

    n_total = len(pres_copy)
    n_test = max(1, round(n_total * test_ratio))
    n_trainval = n_total - n_test
    n_val = max(1, round(n_trainval * val_ratio_within_trainval))
    n_train = n_total - n_test - n_val

    # đảm bảo không âm
    if n_train <= 0:
        raise RuntimeError(
            f"Số prescriptions quá ít để chia. "
            f"n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    test_items = pres_copy[:n_test]
    val_items = pres_copy[n_test:n_test + n_val]
    train_items = pres_copy[n_test + n_val:]

    return train_items, val_items, test_items


def summarize_split(name: str, items: List[Dict]) -> Dict:
    return {
        "split": name,
        "num_prescriptions": len(items),
        "num_pills": int(sum(x["num_pills"] for x in items)),
        "avg_pills_per_prescription": round(
            float(sum(x["num_pills"] for x in items) / len(items)), 4
        ) if len(items) > 0 else 0.0,
    }


def save_txt(items: List[Dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x['pres_key']}\n")


def save_csv(items: List[Dict], split_name: str, out_path: str):
    rows = []
    for x in items:
        rows.append(
            {
                "split": split_name,
                "prescription_json": x["pres"],
                "prescription_key": x["pres_key"],
                "num_pills": x["num_pills"],
                "pill_json_list": "|".join(x["pill_list"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def main(args):
    train_root = args.train_root
    out_dir = args.output_dir

    ensure_dir(out_dir)

    prescriptions = load_public_train_map(train_root)

    print("Total prescriptions found:", len(prescriptions))
    print("Total pill references:", sum(x["num_pills"] for x in prescriptions))

    train_items, val_items, test_items = split_prescriptions(
        prescriptions=prescriptions,
        seed=args.seed,
        test_ratio=args.test_ratio,
        val_ratio_within_trainval=args.val_ratio_within_trainval,
    )

    # save txt
    save_txt(train_items, os.path.join(out_dir, "train_prescriptions.txt"))
    save_txt(val_items, os.path.join(out_dir, "val_prescriptions.txt"))
    save_txt(test_items, os.path.join(out_dir, "test_prescriptions.txt"))

    # save per-split csv
    save_csv(train_items, "train", os.path.join(out_dir, "train_prescriptions.csv"))
    save_csv(val_items, "val", os.path.join(out_dir, "val_prescriptions.csv"))
    save_csv(test_items, "test", os.path.join(out_dir, "test_prescriptions.csv"))

    # save combined split map
    all_rows = []
    for split_name, items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        for x in items:
            all_rows.append(
                {
                    "split": split_name,
                    "prescription_json": x["pres"],
                    "prescription_key": x["pres_key"],
                    "num_pills": x["num_pills"],
                }
            )

    split_map_df = pd.DataFrame(all_rows)
    split_map_path = os.path.join(out_dir, "prescription_split_map.csv")
    split_map_df.to_csv(split_map_path, index=False, encoding="utf-8")

    # save summary
    summary_rows = [
        summarize_split("train", train_items),
        summarize_split("val", val_items),
        summarize_split("test", test_items),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "split_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== SPLIT SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print("-", os.path.join(out_dir, "train_prescriptions.txt"))
    print("-", os.path.join(out_dir, "val_prescriptions.txt"))
    print("-", os.path.join(out_dir, "test_prescriptions.txt"))
    print("-", os.path.join(out_dir, "train_prescriptions.csv"))
    print("-", os.path.join(out_dir, "val_prescriptions.csv"))
    print("-", os.path.join(out_dir, "test_prescriptions.csv"))
    print("-", split_map_path)
    print("-", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create prescription-level train/val/test split from public_train"
    )
    parser.add_argument(
        "--train_root",
        type=str,
        required=True,
        help="Path tới public_train",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits_prescription_level",
        help="Thư mục lưu split files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=52 / 168,
        help="Tỉ lệ test theo prescription, mặc định gần bài báo PIKA",
    )
    parser.add_argument(
        "--val_ratio_within_trainval",
        type=float,
        default=0.203,
        help="Tỉ lệ val trong phần train+val còn lại",
    )
    args = parser.parse_args()
    main(args)

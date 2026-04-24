import os
import json
import random
import argparse
from typing import Dict, List, Set

import pandas as pd


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stem_no_ext(x: str) -> str:
    return os.path.splitext(os.path.basename(str(x)))[0]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_public_train_map(train_root: str) -> List[Dict]:
    map_path = os.path.join(train_root, "pill_pres_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Không tìm thấy: {map_path}")

    data = load_json(map_path)
    if not isinstance(data, list):
        raise RuntimeError("pill_pres_map.json phải ở dạng list")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue

        pres = item.get("pres")
        pill_list = item.get("pill", [])

        if pres is None or not isinstance(pill_list, list):
            continue

        rows.append({
            "pres": str(pres),
            "pres_key": stem_no_ext(pres),
            "pill_list": [str(x) for x in pill_list],
            "num_pills": len(pill_list),
        })

    if len(rows) == 0:
        raise RuntimeError("Không đọc được prescription nào từ pill_pres_map.json")

    return rows


def build_prescription_label_map(metadata_csv: str) -> Dict[str, Set[int]]:
    df = pd.read_csv(metadata_csv).copy()

    required_cols = ["prescription_json", "pill_label"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"{metadata_csv} thiếu cột bắt buộc '{c}'. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )

    df["prescription_key"] = df["prescription_json"].astype(str).apply(stem_no_ext)
    df["pill_label"] = df["pill_label"].astype(int)

    grouped = df.groupby("prescription_key")["pill_label"].apply(lambda x: set(x.tolist()))
    return grouped.to_dict()


def summarize_split(name: str, items: List[Dict], pres_to_labels: Dict[str, Set[int]]) -> Dict:
    label_union = set()
    for x in items:
        label_union.update(pres_to_labels.get(x["pres_key"], set()))

    return {
        "split": name,
        "num_prescriptions": len(items),
        "num_pills": int(sum(x["num_pills"] for x in items)),
        "num_classes": len(label_union),
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
        rows.append({
            "split": split_name,
            "prescription_json": x["pres"],
            "prescription_key": x["pres_key"],
            "num_pills": x["num_pills"],
            "pill_json_list": "|".join(x["pill_list"]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def try_one_split(
    prescriptions: List[Dict],
    pres_to_labels: Dict[str, Set[int]],
    seed: int,
    test_ratio: float,
    val_ratio_within_trainval: float,
):
    rng = random.Random(seed)
    items = prescriptions.copy()
    rng.shuffle(items)

    n_total = len(items)
    n_test = max(1, round(n_total * test_ratio))
    n_trainval = n_total - n_test
    n_val = max(1, round(n_trainval * val_ratio_within_trainval))
    n_train = n_total - n_test - n_val

    if n_train <= 0:
        raise RuntimeError(
            f"Split không hợp lệ: n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    test_items = items[:n_test]
    val_items = items[n_test:n_test + n_val]
    train_items = items[n_test + n_val:]

    train_labels = set()
    val_labels = set()
    test_labels = set()

    for x in train_items:
        train_labels.update(pres_to_labels.get(x["pres_key"], set()))
    for x in val_items:
        val_labels.update(pres_to_labels.get(x["pres_key"], set()))
    for x in test_items:
        test_labels.update(pres_to_labels.get(x["pres_key"], set()))

    missing_val = sorted(list(val_labels - train_labels))
    missing_test = sorted(list(test_labels - train_labels))

    ok = (len(missing_val) == 0 and len(missing_test) == 0)

    return {
        "ok": ok,
        "seed": seed,
        "train_items": train_items,
        "val_items": val_items,
        "test_items": test_items,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "missing_val": missing_val,
        "missing_test": missing_test,
    }


def main(args):
    ensure_dir(args.output_dir)

    prescriptions_all = load_public_train_map(args.train_root)
    pres_to_labels = build_prescription_label_map(args.metadata_csv)

    # chỉ giữ prescriptions có mặt trong metadata
    prescriptions = [x for x in prescriptions_all if x["pres_key"] in pres_to_labels]

    print("Total prescriptions in pill_pres_map:", len(prescriptions_all))
    print("Prescriptions found in metadata    :", len(prescriptions))
    print("Total pill refs in usable split    :", sum(x["num_pills"] for x in prescriptions))

    if len(prescriptions) == 0:
        raise RuntimeError("Không có prescription nào giao nhau giữa pill_pres_map và metadata.")

    best = None
    for seed in range(args.seed_start, args.seed_start + args.max_seed_search):
        result = try_one_split(
            prescriptions=prescriptions,
            pres_to_labels=pres_to_labels,
            seed=seed,
            test_ratio=args.test_ratio,
            val_ratio_within_trainval=args.val_ratio_within_trainval,
        )

        if result["ok"]:
            best = result
            break

    if best is None:
        raise RuntimeError(
            f"Không tìm thấy split hợp lệ trong khoảng seed "
            f"{args.seed_start} .. {args.seed_start + args.max_seed_search - 1}"
        )

    train_items = best["train_items"]
    val_items = best["val_items"]
    test_items = best["test_items"]

    print("\nFound valid seed:", best["seed"])
    print("Train classes:", len(best["train_labels"]))
    print("Val classes  :", len(best["val_labels"]))
    print("Test classes :", len(best["test_labels"]))
    print("Missing val labels in train :", best["missing_val"])
    print("Missing test labels in train:", best["missing_test"])

    save_txt(train_items, os.path.join(args.output_dir, "train_prescriptions.txt"))
    save_txt(val_items, os.path.join(args.output_dir, "val_prescriptions.txt"))
    save_txt(test_items, os.path.join(args.output_dir, "test_prescriptions.txt"))

    save_csv(train_items, "train", os.path.join(args.output_dir, "train_prescriptions.csv"))
    save_csv(val_items, "val", os.path.join(args.output_dir, "val_prescriptions.csv"))
    save_csv(test_items, "test", os.path.join(args.output_dir, "test_prescriptions.csv"))

    all_rows = []
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        for x in items:
            all_rows.append({
                "split": split_name,
                "prescription_json": x["pres"],
                "prescription_key": x["pres_key"],
                "num_pills": x["num_pills"],
            })

    split_map_df = pd.DataFrame(all_rows)
    split_map_df.to_csv(
        os.path.join(args.output_dir, "prescription_split_map.csv"),
        index=False,
        encoding="utf-8",
    )

    summary_df = pd.DataFrame([
        summarize_split("train", train_items, pres_to_labels),
        summarize_split("val", val_items, pres_to_labels),
        summarize_split("test", test_items, pres_to_labels),
    ])
    summary_df.to_csv(
        os.path.join(args.output_dir, "split_summary.csv"),
        index=False,
        encoding="utf-8",
    )

    print("\n=== SPLIT SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\nSaved files to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create prescription-level train/val/test split with label coverage check"
    )
    parser.add_argument("--train_root", type=str, required=True, help="Path tới public_train")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV có cột prescription_json và pill_label")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục lưu split")
    parser.add_argument("--seed_start", type=int, default=0, help="Seed bắt đầu tìm kiếm")
    parser.add_argument("--max_seed_search", type=int, default=5000, help="Số seed sẽ thử")
    parser.add_argument("--test_ratio", type=float, default=52/168, help="Tỉ lệ test theo prescriptions")
    parser.add_argument("--val_ratio_within_trainval", type=float, default=0.203, help="Tỉ lệ val trong phần train+val")
    args = parser.parse_args()
    main(args)

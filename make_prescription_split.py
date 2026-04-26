import os
import json
import random
import argparse
from typing import Dict, List, Set, Tuple

import numpy as np
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
        raise RuntimeError("pill_pres_map.json phải ở dạng list.")

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
        raise RuntimeError("Không đọc được prescription nào từ pill_pres_map.json.")

    return rows


def build_prescription_label_map(metadata_csv: str) -> Dict[str, Set[int]]:
    df = pd.read_csv(metadata_csv).copy()

    required_cols = ["prescription_json", "pill_label"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"{metadata_csv} thiếu cột '{c}'. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )

    df["prescription_key"] = df["prescription_json"].astype(str).apply(stem_no_ext)
    df["pill_label"] = df["pill_label"].astype(int)

    grouped = df.groupby("prescription_key")["pill_label"].apply(lambda x: set(x.tolist()))
    return grouped.to_dict()


def build_label_to_prescriptions(pres_to_labels: Dict[str, Set[int]]) -> Dict[int, Set[str]]:
    label_to_pres = {}
    for pres_key, labels in pres_to_labels.items():
        for label in labels:
            label_to_pres.setdefault(label, set()).add(pres_key)
    return label_to_pres


def multilabel_matrix(keys: List[str], pres_to_labels: Dict[str, Set[int]], label_list: List[int]):
    label_to_col = {label: i for i, label in enumerate(label_list)}
    y = np.zeros((len(keys), len(label_list)), dtype=int)

    for i, key in enumerate(keys):
        for label in pres_to_labels.get(key, set()):
            if label in label_to_col:
                y[i, label_to_col[label]] = 1

    return y


def iterative_split_indices(keys: List[str], y: np.ndarray, test_size: float):
    try:
        from skmultilearn.model_selection import iterative_train_test_split
    except ImportError as e:
        raise ImportError(
            "Thiếu thư viện scikit-multilearn. "
            "Cài trong Colab bằng: !pip install scikit-multilearn"
        ) from e

    if len(keys) == 0:
        return [], []

    if test_size <= 0:
        return keys, []

    if test_size >= 1:
        return [], keys

    x = np.arange(len(keys)).reshape(-1, 1)

    x_train, y_train, x_test, y_test = iterative_train_test_split(
        x,
        y,
        test_size=test_size,
    )

    train_indices = x_train.reshape(-1).astype(int).tolist()
    test_indices = x_test.reshape(-1).astype(int).tolist()

    train_keys = [keys[i] for i in train_indices]
    test_keys = [keys[i] for i in test_indices]

    return train_keys, test_keys


def label_union(keys: Set[str], pres_to_labels: Dict[str, Set[int]]) -> Set[int]:
    labels = set()
    for key in keys:
        labels.update(pres_to_labels.get(key, set()))
    return labels


def repair_unseen_labels(
    train_keys: Set[str],
    val_keys: Set[str],
    test_keys: Set[str],
    pres_to_labels: Dict[str, Set[int]],
):
    """
    Nếu val/test có nhãn mà train chưa có, dời prescription chứa nhãn đó sang train.
    Vẫn bảo đảm không chồng chéo vì prescription bị remove khỏi val/test trước khi thêm vào train.
    """
    changed = True

    while changed:
        changed = False

        train_labels = label_union(train_keys, pres_to_labels)
        val_labels = label_union(val_keys, pres_to_labels)
        test_labels = label_union(test_keys, pres_to_labels)

        missing_val = sorted(list(val_labels - train_labels))
        missing_test = sorted(list(test_labels - train_labels))

        for label in missing_val:
            candidates = [k for k in val_keys if label in pres_to_labels.get(k, set())]
            if len(candidates) > 0:
                move_key = sorted(candidates)[0]
                val_keys.remove(move_key)
                train_keys.add(move_key)
                changed = True

        train_labels = label_union(train_keys, pres_to_labels)

        for label in missing_test:
            candidates = [k for k in test_keys if label in pres_to_labels.get(k, set())]
            if len(candidates) > 0:
                move_key = sorted(candidates)[0]
                test_keys.remove(move_key)
                train_keys.add(move_key)
                changed = True

    return train_keys, val_keys, test_keys


def summarize_split(name: str, keys: Set[str], pres_lookup: Dict[str, Dict], pres_to_labels: Dict[str, Set[int]]) -> Dict:
    labels = label_union(keys, pres_to_labels)
    num_pills = sum(pres_lookup[k]["num_pills"] for k in keys)

    return {
        "split": name,
        "num_prescriptions": len(keys),
        "num_pills": int(num_pills),
        "num_classes": len(labels),
        "avg_pills_per_prescription": round(num_pills / len(keys), 4) if len(keys) > 0 else 0.0,
    }


def save_txt(keys: Set[str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for key in sorted(keys):
            f.write(f"{key}\n")


def save_csv(keys: Set[str], split_name: str, pres_lookup: Dict[str, Dict], out_path: str):
    rows = []
    for key in sorted(keys):
        x = pres_lookup[key]
        rows.append({
            "split": split_name,
            "prescription_json": x["pres"],
            "prescription_key": x["pres_key"],
            "num_pills": x["num_pills"],
            "pill_json_list": "|".join(x["pill_list"]),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def save_label_frequency(label_to_pres: Dict[int, Set[str]], out_path: str):
    rows = []
    for label, pres_set in sorted(label_to_pres.items()):
        rows.append({
            "pill_label": label,
            "num_prescriptions": len(pres_set),
            "prescription_keys": "|".join(sorted(pres_set)),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def main(args):
    ensure_dir(args.output_dir)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total_ratio
    val_ratio = args.val_ratio / total_ratio
    test_ratio = args.test_ratio / total_ratio

    print("Target ratios:")
    print(f"train={train_ratio:.4f}, val={val_ratio:.4f}, test={test_ratio:.4f}")

    prescriptions_all = load_public_train_map(args.train_root)
    pres_to_labels = build_prescription_label_map(args.metadata_csv)

    prescriptions = [x for x in prescriptions_all if x["pres_key"] in pres_to_labels]
    pres_lookup = {x["pres_key"]: x for x in prescriptions}
    all_keys = set(pres_lookup.keys())

    label_to_pres = build_label_to_prescriptions(pres_to_labels)
    label_list = sorted(label_to_pres.keys())

    print("\nDataset info:")
    print("Total prescriptions in pill_pres_map:", len(prescriptions_all))
    print("Prescriptions found in metadata    :", len(prescriptions))
    print("Total labels/classes               :", len(label_list))
    print("Total pill refs in usable split    :", sum(x["num_pills"] for x in prescriptions))

    save_label_frequency(
        label_to_pres,
        os.path.join(args.output_dir, "label_frequency_by_prescription.csv"),
    )

    # 1. Xử lý nhãn hiếm
    forced_train = set()
    preferred_test = set()
    rare_policy_rows = []

    # Nhãn chỉ có 1 prescription: bắt buộc vào train
    for label, pres_set in label_to_pres.items():
        if len(pres_set) == 1:
            key = list(pres_set)[0]
            forced_train.add(key)
            rare_policy_rows.append({
                "pill_label": label,
                "num_prescriptions": 1,
                "policy": "force_to_train",
                "train_prescription": key,
                "test_prescription": "",
            })

    # Nhãn có 2 prescription: cố gắng 1 train, 1 test
    for label, pres_set in label_to_pres.items():
        if len(pres_set) == 2:
            keys = sorted(list(pres_set))

            already_train = [k for k in keys if k in forced_train]

            if len(already_train) > 0:
                train_key = already_train[0]
                test_candidates = [k for k in keys if k != train_key]
                test_key = test_candidates[0] if len(test_candidates) > 0 else ""
            else:
                train_key = keys[0]
                test_key = keys[1]

            forced_train.add(train_key)

            if test_key and test_key not in forced_train:
                preferred_test.add(test_key)

            rare_policy_rows.append({
                "pill_label": label,
                "num_prescriptions": 2,
                "policy": "one_to_train_one_to_test_if_possible",
                "train_prescription": train_key,
                "test_prescription": test_key,
            })

    preferred_test = preferred_test - forced_train

    pd.DataFrame(rare_policy_rows).to_csv(
        os.path.join(args.output_dir, "rare_label_policy.csv"),
        index=False,
        encoding="utf-8",
    )

    print("\nRare label handling:")
    print("Forced train prescriptions:", len(forced_train))
    print("Preferred test prescriptions:", len(preferred_test))

    # 2. Chia phần còn lại bằng iterative stratification
    fixed_keys = forced_train | preferred_test
    remaining_keys = sorted(list(all_keys - fixed_keys))

    rng = random.Random(args.seed)
    rng.shuffle(remaining_keys)

    n_total = len(all_keys)
    target_train = round(n_total * train_ratio)
    target_val = round(n_total * val_ratio)
    target_test = n_total - target_train - target_val

    remaining_test_slots = max(0, target_test - len(preferred_test))
    remaining_val_slots = max(0, target_val)

    print("\nTarget split sizes:")
    print("target_train:", target_train)
    print("target_val  :", target_val)
    print("target_test :", target_test)

    print("\nRemaining split slots:")
    print("remaining keys       :", len(remaining_keys))
    print("remaining test slots :", remaining_test_slots)
    print("remaining val slots  :", remaining_val_slots)

    # split thêm test từ phần còn lại
    if len(remaining_keys) > 0 and remaining_test_slots > 0:
        y_remaining = multilabel_matrix(remaining_keys, pres_to_labels, label_list)
        test_size = min(0.95, remaining_test_slots / len(remaining_keys))
        pool_keys, add_test_keys = iterative_split_indices(remaining_keys, y_remaining, test_size)
    else:
        pool_keys = remaining_keys
        add_test_keys = []

    # split val từ pool
    if len(pool_keys) > 0 and remaining_val_slots > 0:
        y_pool = multilabel_matrix(pool_keys, pres_to_labels, label_list)
        val_size = min(0.95, remaining_val_slots / len(pool_keys))
        add_train_keys, val_keys_list = iterative_split_indices(pool_keys, y_pool, val_size)
    else:
        add_train_keys = pool_keys
        val_keys_list = []

    train_keys = set(forced_train) | set(add_train_keys)
    val_keys = set(val_keys_list)
    test_keys = set(preferred_test) | set(add_test_keys)

    # 3. Repair: nếu val/test có nhãn chưa có trong train thì dời prescription đó sang train
    train_keys, val_keys, test_keys = repair_unseen_labels(
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
        pres_to_labels=pres_to_labels,
    )

    # 4. Kiểm tra không chồng chéo
    overlap_train_val = train_keys & val_keys
    overlap_train_test = train_keys & test_keys
    overlap_val_test = val_keys & test_keys

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            "Split bị chồng chéo. "
            f"train∩val={len(overlap_train_val)}, "
            f"train∩test={len(overlap_train_test)}, "
            f"val∩test={len(overlap_val_test)}"
        )

    # 5. Kiểm tra label coverage
    train_labels = label_union(train_keys, pres_to_labels)
    val_labels = label_union(val_keys, pres_to_labels)
    test_labels = label_union(test_keys, pres_to_labels)

    unseen_val = sorted(list(val_labels - train_labels))
    unseen_test = sorted(list(test_labels - train_labels))

    print("\nFinal label coverage check:")
    print("Train classes:", len(train_labels))
    print("Val classes  :", len(val_labels))
    print("Test classes :", len(test_labels))
    print("Labels in val but not in train :", unseen_val)
    print("Labels in test but not in train:", unseen_test)

    if len(unseen_val) > 0 or len(unseen_test) > 0:
        raise RuntimeError("Vẫn còn nhãn ở val/test nhưng không có trong train.")

    print("\nOverlap check:")
    print("train∩val :", len(overlap_train_val))
    print("train∩test:", len(overlap_train_test))
    print("val∩test  :", len(overlap_val_test))

    # 6. Save files
    save_txt(train_keys, os.path.join(args.output_dir, "train_prescriptions.txt"))
    save_txt(val_keys, os.path.join(args.output_dir, "val_prescriptions.txt"))
    save_txt(test_keys, os.path.join(args.output_dir, "test_prescriptions.txt"))

    save_csv(train_keys, "train", pres_lookup, os.path.join(args.output_dir, "train_prescriptions.csv"))
    save_csv(val_keys, "val", pres_lookup, os.path.join(args.output_dir, "val_prescriptions.csv"))
    save_csv(test_keys, "test", pres_lookup, os.path.join(args.output_dir, "test_prescriptions.csv"))

    all_rows = []
    for split_name, keys in [
        ("train", train_keys),
        ("val", val_keys),
        ("test", test_keys),
    ]:
        for key in sorted(keys):
            x = pres_lookup[key]
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
        summarize_split("train", train_keys, pres_lookup, pres_to_labels),
        summarize_split("val", val_keys, pres_lookup, pres_to_labels),
        summarize_split("test", test_keys, pres_lookup, pres_to_labels),
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
        description="Prescription-level train/val/test split with rare-label handling and iterative stratification"
    )
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    args = parser.parse_args()
    main(args)

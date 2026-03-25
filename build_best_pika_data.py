import os
import json
import csv
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_ROOT = os.environ.get(
    "VAIPE_TRAIN_ROOT",
    "/root/.cache/kagglehub/datasets/tommyngx/vaipepill2022/versions/1/public_train"
)

OUTPUT_ROOT = os.environ.get(
    "PIKA_BEST_OUTPUT_ROOT",
    "/content/processed_pika_best"
)

METADATA_CSV = os.path.join(OUTPUT_ROOT, "best_pika_metadata.csv")
GRAPH_LABELS_JSON = os.path.join(OUTPUT_ROOT, "graph_labels.json")
GRAPH_PMI_NPY = os.path.join(OUTPUT_ROOT, "graph_pmi.npy")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp_bbox(x, y, w, h, img_w, img_h):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    return x1, y1, x2, y2


def load_pill_pres_map(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pill_to_pres = {}
    pres_to_pills = {}

    for item in data:
        pres_file = item["pres"]
        pill_files = item["pill"]
        pres_to_pills[pres_file] = pill_files
        for pill_file in pill_files:
            pill_to_pres[pill_file] = pres_file

    return pill_to_pres, pres_to_pills


def load_labels_from_pill_json(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    if not isinstance(anns, list) or len(anns) == 0:
        return []

    labels = []
    for ann in anns:
        if "label" in ann:
            labels.append(int(ann["label"]))

    return sorted(list(set(labels)))


def build_ppmi_matrix(pres_to_label_set, graph_labels):
    label_to_idx = {label: idx for idx, label in enumerate(graph_labels)}
    n = len(graph_labels)

    label_counts = np.zeros(n, dtype=np.float64)
    pair_counts = np.zeros((n, n), dtype=np.float64)

    prescriptions = list(pres_to_label_set.values())
    num_pres = len(prescriptions)

    for label_list in prescriptions:
        uniq = sorted(list(set(label_list)))
        for lab in uniq:
            label_counts[label_to_idx[lab]] += 1.0
        for a, b in combinations(uniq, 2):
            i = label_to_idx[a]
            j = label_to_idx[b]
            pair_counts[i, j] += 1.0
            pair_counts[j, i] += 1.0

    ppmi = np.zeros((n, n), dtype=np.float32)
    eps = 1e-8

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pair_counts[i, j] <= 0:
                continue
            p_ij = pair_counts[i, j] / num_pres
            p_i = label_counts[i] / num_pres
            p_j = label_counts[j] / num_pres
            val = np.log((p_ij / (p_i * p_j + eps)) + eps)
            ppmi[i, j] = max(0.0, val)

    return ppmi


def main():
    pill_image_dir = os.path.join(TRAIN_ROOT, "pill", "image")
    pill_label_dir = os.path.join(TRAIN_ROOT, "pill", "label")
    pres_image_dir = os.path.join(TRAIN_ROOT, "prescription", "image")
    map_path = os.path.join(TRAIN_ROOT, "pill_pres_map.json")

    ensure_dir(OUTPUT_ROOT)
    crops_dir = os.path.join(OUTPUT_ROOT, "pill_crops")
    ensure_dir(crops_dir)

    pill_to_pres, pres_to_pills = load_pill_pres_map(map_path)
    label_files = sorted([f for f in os.listdir(pill_label_dir) if f.endswith(".json")])

    print("Building pill label lookup...")
    pill_json_to_labels = {}
    all_labels = set()

    for label_file in tqdm(label_files):
        label_path = os.path.join(pill_label_dir, label_file)
        labels = load_labels_from_pill_json(label_path)
        pill_json_to_labels[label_file] = labels
        all_labels.update(labels)

    graph_labels = sorted(list(all_labels))

    print("Building prescription label sets...")
    pres_to_label_set = {}
    for pres_file, pill_files in tqdm(pres_to_pills.items()):
        label_set = set()
        for pill_json in pill_files:
            label_set.update(pill_json_to_labels.get(pill_json, []))
        pres_to_label_set[pres_file] = sorted(list(label_set))

    print("Building PPMI graph...")
    ppmi = build_ppmi_matrix(pres_to_label_set, graph_labels)

    with open(GRAPH_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_labels, f, indent=2)

    np.save(GRAPH_PMI_NPY, ppmi)

    rows = []
    skipped_files = 0
    skipped_boxes = 0
    total_instances = 0

    print("Building final metadata...")
    for label_file in tqdm(label_files):
        label_path = os.path.join(pill_label_dir, label_file)
        image_file = label_file.replace(".json", ".jpg")
        image_path = os.path.join(pill_image_dir, image_file)

        if not os.path.exists(image_path):
            skipped_files += 1
            continue

        pres_file = pill_to_pres.get(label_file, None)
        if pres_file is None:
            skipped_files += 1
            continue

        pres_image_file = pres_file.replace(".json", ".png")
        pres_image_path = os.path.join(pres_image_dir, pres_image_file)
        if not os.path.exists(pres_image_path):
            skipped_files += 1
            continue

        pres_label_set = pres_to_label_set.get(pres_file, [])

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            if not isinstance(annotations, list) or len(annotations) == 0:
                skipped_files += 1
                continue

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            for idx, ann in enumerate(annotations):
                if "label" not in ann:
                    skipped_boxes += 1
                    continue

                x = ann.get("x", 0)
                y = ann.get("y", 0)
                w = ann.get("w", 0)
                h = ann.get("h", 0)
                label = int(ann["label"])

                x1, y1, x2, y2 = clamp_bbox(x, y, w, h, img_w, img_h)
                if x2 <= x1 or y2 <= y1:
                    skipped_boxes += 1
                    continue

                context_labels = [lab for lab in pres_label_set if lab != label]

                crop = img.crop((x1, y1, x2, y2))
                crop_name = f"{Path(label_file).stem}_box{idx}.jpg"
                crop_path = os.path.join(crops_dir, crop_name)
                crop.save(crop_path)

                rows.append({
                    "pill_crop_path": crop_path,
                    "pill_label": label,
                    "pill_json": label_file,
                    "prescription_json": pres_file,
                    "prescription_image_path": pres_image_path,
                    "context_labels": json.dumps(sorted(list(set(context_labels))))
                })
                total_instances += 1

        except Exception as e:
            skipped_files += 1
            print(f"Error processing {label_file}: {e}")

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pill_crop_path",
                "pill_label",
                "pill_json",
                "prescription_json",
                "prescription_image_path",
                "context_labels"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Total valid instances: {total_instances}")
    print(f"Skipped files: {skipped_files}")
    print(f"Skipped boxes: {skipped_boxes}")
    print(f"Metadata saved to: {METADATA_CSV}")
    print(f"Graph labels saved to: {GRAPH_LABELS_JSON}")
    print(f"Graph PPMI saved to: {GRAPH_PMI_NPY}")
    print(f"Number of graph nodes: {len(graph_labels)}")


if __name__ == "__main__":
    main()

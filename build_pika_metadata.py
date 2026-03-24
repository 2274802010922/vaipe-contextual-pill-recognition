import os
import json
import csv
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


TRAIN_ROOT = "/kaggle/input/vaipepill2022/public_train"
OUTPUT_ROOT = "/content/processed_pika"
METADATA_CSV = "/content/processed_pika/pika_metadata.csv"


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
    for item in data:
        pres_file = item["pres"]
        for pill_file in item["pill"]:
            pill_to_pres[pill_file] = pres_file
    return pill_to_pres


def main():
    pill_image_dir = os.path.join(TRAIN_ROOT, "pill", "image")
    pill_label_dir = os.path.join(TRAIN_ROOT, "pill", "label")
    pres_image_dir = os.path.join(TRAIN_ROOT, "prescription", "image")
    map_path = os.path.join(TRAIN_ROOT, "pill_pres_map.json")

    ensure_dir(OUTPUT_ROOT)
    crops_dir = os.path.join(OUTPUT_ROOT, "pill_crops")
    ensure_dir(crops_dir)

    pill_to_pres = load_pill_pres_map(map_path)

    label_files = sorted([f for f in os.listdir(pill_label_dir) if f.endswith(".json")])

    rows = []
    skipped_files = 0
    skipped_boxes = 0
    total_instances = 0

    for label_file in tqdm(label_files, desc="Building PIKA metadata"):
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

                crop = img.crop((x1, y1, x2, y2))

                crop_name = f"{Path(label_file).stem}_box{idx}.jpg"
                crop_path = os.path.join(crops_dir, crop_name)
                crop.save(crop_path)

                rows.append({
                    "pill_crop_path": crop_path,
                    "pill_label": label,
                    "pill_json": label_file,
                    "pill_image": image_file,
                    "prescription_json": pres_file,
                    "prescription_image_path": pres_image_path
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
                "pill_image",
                "prescription_json",
                "prescription_image_path"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Total valid instances: {total_instances}")
    print(f"Skipped files: {skipped_files}")
    print(f"Skipped boxes: {skipped_boxes}")
    print(f"Metadata saved to: {METADATA_CSV}")


if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp_bbox(x, y, w, h, img_w, img_h):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    return x1, y1, x2, y2


def process_split(input_root: str, output_root: str, split_name: str = "train"):
    pill_image_dir = os.path.join(input_root, "pill", "image")
    pill_label_dir = os.path.join(input_root, "pill", "label")

    if not os.path.exists(pill_image_dir):
        raise FileNotFoundError(f"Image directory not found: {pill_image_dir}")
    if not os.path.exists(pill_label_dir):
        raise FileNotFoundError(f"Label directory not found: {pill_label_dir}")

    ensure_dir(output_root)

    label_files = sorted([f for f in os.listdir(pill_label_dir) if f.endswith(".json")])

    total_boxes = 0
    skipped_files = 0
    skipped_boxes = 0
    class_counter = {}

    for label_file in tqdm(label_files, desc=f"Processing {split_name}"):
        label_path = os.path.join(pill_label_dir, label_file)

        image_file = label_file.replace(".json", ".jpg")
        image_path = os.path.join(pill_image_dir, image_file)

        if not os.path.exists(image_path):
            skipped_files += 1
            continue

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            if not isinstance(annotations, list) or len(annotations) == 0:
                skipped_files += 1
                continue

            for idx, ann in enumerate(annotations):
                if "label" not in ann:
                    skipped_boxes += 1
                    continue

                x = ann.get("x", 0)
                y = ann.get("y", 0)
                w = ann.get("w", 0)
                h = ann.get("h", 0)
                cls = ann["label"]

                x1, y1, x2, y2 = clamp_bbox(x, y, w, h, img_w, img_h)

                if x2 <= x1 or y2 <= y1:
                    skipped_boxes += 1
                    continue

                crop = img.crop((x1, y1, x2, y2))

                class_dir = os.path.join(output_root, str(cls))
                ensure_dir(class_dir)

                crop_name = f"{Path(label_file).stem}_box{idx}.jpg"
                crop_path = os.path.join(class_dir, crop_name)

                crop.save(crop_path)

                total_boxes += 1
                class_counter[cls] = class_counter.get(cls, 0) + 1

        except Exception as e:
            skipped_files += 1
            print(f"Error processing {label_file}: {e}")

    print("\nDone.")
    print(f"Total cropped pill instances: {total_boxes}")
    print(f"Skipped files: {skipped_files}")
    print(f"Skipped boxes: {skipped_boxes}")
    print(f"Number of classes found: {len(class_counter)}")

    sorted_classes = sorted(class_counter.items(), key=lambda x: x[0])
    print("\nSample class distribution:")
    for cls, count in sorted_classes[:20]:
        print(f"Class {cls}: {count}")


if __name__ == "__main__":
    INPUT_ROOT = "/kaggle/input/vaipepill2022/public_train"
    OUTPUT_ROOT = "/content/processed_pill_cls"

    process_split(INPUT_ROOT, OUTPUT_ROOT, split_name="train")

import os
import csv
import argparse
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vaipe_detection_dataset import VaipeDetectionInferenceDataset


def build_model(num_classes: int = 2):
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception:
        model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint detector: {checkpoint_path}")

    model = build_model(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def clamp_box(box: Tuple[float, float, float, float], width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def add_padding(box: Tuple[int, int, int, int], width: int, height: int, pad_pixels: int):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad_pixels)
    y1 = max(0, y1 - pad_pixels)
    x2 = min(width, x2 + pad_pixels)
    y2 = min(height, y2 + pad_pixels)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_boxes(image: Image.Image, boxes: List[Tuple[int, int, int, int]], scores: List[float]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{i}:{score:.3f}"
        text_y = max(0, y1 - 14)
        draw.text((x1, text_y), text, fill="red")

    return annotated


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = VaipeDetectionInferenceDataset(args.test_root)
    model = load_detector(args.checkpoint, device)

    os.makedirs(args.out_dir, exist_ok=True)
    annotated_dir = os.path.join(args.out_dir, "annotated")
    crops_dir = os.path.join(args.out_dir, "crops")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "detections.csv")

    rows = []
    print("Num test images:", len(dataset))

    for idx in range(len(dataset)):
        image_tensor, image_path = dataset[idx]
        image_name = os.path.basename(image_path)
        image_stem = os.path.splitext(image_name)[0]

        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size

        with torch.no_grad():
            outputs = model([image_tensor.to(device)])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()

        kept_boxes = []
        kept_scores = []

        det_count = 0
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < args.score_thresh:
                continue
            if int(label) != 1:
                continue

            clamped = clamp_box((box[0], box[1], box[2], box[3]), width, height)
            if clamped is None:
                continue

            padded = add_padding(clamped, width, height, args.pad_pixels)
            if padded is None:
                continue

            kept_boxes.append(padded)
            kept_scores.append(float(score))
            det_count += 1

            if det_count >= args.max_detections:
                break

        # Save annotated image
        annotated = draw_boxes(pil_image, kept_boxes, kept_scores)
        annotated.save(os.path.join(annotated_dir, image_name))

        # Save crops and CSV rows
        for det_idx, (box, score) in enumerate(zip(kept_boxes, kept_scores), start=1):
            x1, y1, x2, y2 = box
            crop = pil_image.crop((x1, y1, x2, y2))

            crop_name = f"{image_stem}_det_{det_idx:03d}.png"
            crop_path = os.path.join(crops_dir, crop_name)
            crop.save(crop_path)

            rows.append({
                "image_name": image_name,
                "image_path": image_path,
                "crop_name": crop_name,
                "crop_path": crop_path,
                "det_idx": det_idx,
                "score": score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
            })

        if (idx + 1) % 20 == 0 or idx == len(dataset) - 1:
            print(f"Processed {idx + 1}/{len(dataset)} images")

    fieldnames = [
        "image_name",
        "image_path",
        "crop_name",
        "crop_path",
        "det_idx",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print("Annotated images:", annotated_dir)
    print("Crops:", crops_dir)
    print("Detections CSV:", csv_path)
    print("Total detections:", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Faster R-CNN detector on VAIPE public_test")

    parser.add_argument("--test_root", type=str, required=True, help="Path to public_test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fasterrcnn_vaipe_best.pth")
    parser.add_argument("--out_dir", type=str, default="outputs_public_test_detection")
    parser.add_argument("--score_thresh", type=float, default=0.60)
    parser.add_argument("--max_detections", type=int, default=50)
    parser.add_argument("--pad_pixels", type=int, default=4)

    args = parser.parse_args()
    run_inference(args)

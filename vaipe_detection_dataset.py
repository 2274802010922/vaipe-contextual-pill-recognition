import os
import glob
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def list_images(image_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(files)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_annotation_list(data: Any) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa dữ liệu annotation về dạng list[dict].

    Hỗ trợ các kiểu thường gặp:
    - list các object annotation
    - dict chứa 1 trong các key: annotations / objects / labels / pills / shapes
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ["annotations", "objects", "labels", "pills", "shapes"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    return []


def parse_bbox_from_annotation(ann: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Trả về bbox theo format (x1, y1, x2, y2).

    Ưu tiên hỗ trợ đúng format VAIPE thường gặp:
    - {"x": ..., "y": ..., "w": ..., "h": ...}

    Có hỗ trợ thêm một số format dự phòng:
    - {"bbox": [x, y, w, h]}
    - {"bbox": [x1, y1, x2, y2]}
    - {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    """
    try:
        # Format chính: x, y, w, h
        if all(k in ann for k in ["x", "y", "w", "h"]):
            x = float(ann["x"])
            y = float(ann["y"])
            w = float(ann["w"])
            h = float(ann["h"])
            return x, y, x + w, y + h

        # Dự phòng: bbox
        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
            a, b, c, d = ann["bbox"]
            a = float(a)
            b = float(b)
            c = float(c)
            d = float(d)

            # Nếu có vẻ là [x, y, w, h]
            if c > 0 and d > 0 and (c < 5000 and d < 5000):
                # Ta không thể chắc 100% là wh hay x2y2, nên ưu tiên cách phổ biến hơn: xywh
                # Nhưng nếu c <= a hoặc d <= b thì hiểu là x2y2
                if c <= a or d <= b:
                    return a, b, c, d
                return a, b, a + c, b + d

        # Dự phòng: x1,y1,x2,y2
        if all(k in ann for k in ["x1", "y1", "x2", "y2"]):
            return (
                float(ann["x1"]),
                float(ann["y1"]),
                float(ann["x2"]),
                float(ann["y2"]),
            )

    except Exception:
        return None

    return None


def clamp_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


class VaipeDetectionDataset(Dataset):
    """
    Dataset để train Faster R-CNN trên public_train.

    Kỳ vọng cấu trúc:
        root_dir/
            pill/
                image/
                    xxx.jpg
                label/
                    xxx.json

    Output:
        image: Tensor [C,H,W]
        target: dict gồm boxes, labels, image_id, area, iscrowd
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "pill", "image")
        self.label_dir = os.path.join(root_dir, "pill", "label")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục image: {self.image_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục label: {self.label_dir}")

        self.samples: List[Tuple[str, str]] = []
        for image_path in list_images(self.image_dir):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(self.label_dir, stem + ".json")
            if os.path.exists(json_path):
                self.samples.append((image_path, json_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Không tìm thấy cặp (ảnh, json) hợp lệ trong {root_dir}. "
                f"Kiểm tra lại cấu trúc pill/image và pill/label."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, json_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        raw_data = load_json(json_path)
        annotations = normalize_annotation_list(raw_data)

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        area_list: List[float] = []

        for ann in annotations:
            parsed = parse_bbox_from_annotation(ann)
            if parsed is None:
                continue

            clamped = clamp_box(parsed, width, height)
            if clamped is None:
                continue

            x1, y1, x2, y2 = clamped
            boxes_list.append([x1, y1, x2, y2])

            # Detector chỉ có 1 class foreground là "pill"
            labels_list.append(1)
            area_list.append((x2 - x1) * (y2 - y1))

        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            area = torch.tensor(area_list, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        image_tensor = F.to_tensor(image)
        return image_tensor, target


class VaipeDetectionInferenceDataset(Dataset):
    """
    Dataset chỉ dùng để suy luận detector trên public_test.

    Kỳ vọng cấu trúc:
        root_dir/
            pill/
                image/
                    *.jpg / *.png ...

    Output:
        image_tensor, image_path
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "pill", "image")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục image: {self.image_dir}")

        self.image_paths = list_images(self.image_dir)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)
        return image_tensor, image_path


def collate_fn(batch):
    return tuple(zip(*batch))

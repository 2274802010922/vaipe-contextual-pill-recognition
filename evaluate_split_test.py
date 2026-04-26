import os
import json
import argparse
from typing import Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import timm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_image_column(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col is not None and user_col != "":
        if user_col not in df.columns:
            raise RuntimeError(
                f"Không tìm thấy image column '{user_col}'. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )
        return user_col

    candidates = [
        "pill_crop_path",
        "crop_path",
        "pill_image_path",
        "image_path",
        "pill_image",
        "image",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Không tự tìm được cột ảnh. "
        f"Các cột hiện có: {df.columns.tolist()}\n"
        "Hãy truyền thêm --image_col tên_cột_ảnh."
    )


def resolve_image_path(path_value: str, image_root: Optional[str] = None) -> str:
    path_value = str(path_value)

    if os.path.isabs(path_value):
        return path_value

    if image_root is not None and image_root != "":
        return os.path.join(image_root, path_value)

    return path_value


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def normalize_label_to_idx(raw_mapping: Dict) -> Dict[int, int]:
    fixed = {}
    for k, v in raw_mapping.items():
        fixed[int(k)] = int(v)
    return fixed


def normalize_idx_to_label(raw_mapping: Dict) -> Dict[int, int]:
    fixed = {}
    for k, v in raw_mapping.items():
        fixed[int(k)] = int(v)
    return fixed


class SplitTestDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_to_idx: Dict[int, int],
        transform,
        image_col: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path).copy()
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.image_root = image_root

        if "pill_label" not in self.df.columns:
            raise RuntimeError(
                f"{csv_path} thiếu cột 'pill_label'. "
                f"Các cột hiện có: {self.df.columns.tolist()}"
            )

        self.image_col = find_image_column(self.df, image_col)

        self.df["pill_label"] = self.df["pill_label"].astype(int)
        self.df["image_path_resolved"] = self.df[self.image_col].astype(str).apply(
            lambda x: resolve_image_path(x, self.image_root)
        )

        before = len(self.df)
        self.df = self.df[self.df["image_path_resolved"].apply(os.path.exists)].copy()
        after = len(self.df)

        if after < before:
            print(f"[Warning] Bỏ {before - after} dòng vì image path không tồn tại.")

        self.df["target_idx"] = self.df["pill_label"].map(self.label_to_idx)

        missing = self.df["target_idx"].isna().sum()
        if missing > 0:
            missing_labels = sorted(self.df[self.df["target_idx"].isna()]["pill_label"].unique().tolist())
            print("[Warning] Có label không nằm trong label_to_idx:", missing_labels)
            self.df = self.df[self.df["target_idx"].notna()].copy()

        self.df["target_idx"] = self.df["target_idx"].astype(int)
        self.df.reset_index(drop=True, inplace=True)

        if len(self.df) == 0:
            raise RuntimeError(f"Dataset rỗng sau khi lọc: {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path_resolved"]).convert("RGB")
        img = self.transform(img)

        target_idx = int(row["target_idx"])
        true_label = int(row["pill_label"])
        image_path = row["image_path_resolved"]

        return img, target_idx, true_label, image_path


@torch.no_grad()
def predict(model, loader, device, idx_to_label: Dict[int, int]):
    model.eval()

    rows = []
    all_true_idx = []
    all_pred_idx = []

    for images, targets, true_labels, image_paths in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        preds_cpu = preds.detach().cpu().numpy().tolist()
        targets_cpu = targets.detach().cpu().numpy().tolist()
        confs_cpu = confs.detach().cpu().numpy().tolist()

        true_labels_cpu = true_labels.detach().cpu().numpy().tolist()

        for i in range(len(preds_cpu)):
            pred_idx = int(preds_cpu[i])
            true_idx = int(targets_cpu[i])

            pred_label = int(idx_to_label[pred_idx])
            true_label = int(true_labels_cpu[i])

            rows.append({
                "image_path": image_paths[i],
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": float(confs_cpu[i]),
                "correct": int(pred_idx == true_idx),
            })

            all_true_idx.append(true_idx)
            all_pred_idx.append(pred_idx)

    return pd.DataFrame(rows), all_true_idx, all_pred_idx


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Checkpoint:", args.checkpoint)
    print("Test CSV  :", args.test_csv)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if "label_to_idx" not in ckpt or "idx_to_label" not in ckpt:
        raise RuntimeError("Checkpoint thiếu label_to_idx hoặc idx_to_label.")

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt.get("num_classes", len(idx_to_label)))
    model_name = ckpt.get("model_name", args.model_name)

    print("Model name :", model_name)
    print("Num classes:", num_classes)

    dataset = SplitTestDataset(
        csv_path=args.test_csv,
        label_to_idx=label_to_idx,
        transform=get_eval_transform(args.image_size),
        image_col=args.image_col,
        image_root=args.image_root,
    )

    print("Test samples:", len(dataset))
    print("Image column:", dataset.image_col)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    pred_df, y_true, y_pred = predict(
        model=model,
        loader=loader,
        device=device,
        idx_to_label=idx_to_label,
    )

    acc = accuracy_score(y_true, y_pred)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "num_test_samples": len(y_true),
        "num_classes": num_classes,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    summary_df = pd.DataFrame([summary])

    predictions_path = os.path.join(args.output_dir, args.predictions_name)
    summary_path = os.path.join(args.output_dir, args.summary_name)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")

    pred_df.to_csv(predictions_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    np.save(cm_path, cm)

    print("\n=== TEST METRICS ===")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro Precision   : {precision:.4f}")
    print(f"Macro Recall      : {recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall   : {weighted_recall:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    print("\nSaved:")
    print("-", predictions_path)
    print("-", summary_path)
    print("-", report_path)
    print("-", cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on split test set")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--image_col", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--predictions_name", type=str, default="test_predictions.csv")
    parser.add_argument("--summary_name", type=str, default="test_metrics_summary.csv")

    args = parser.parse_args()
    main(args)

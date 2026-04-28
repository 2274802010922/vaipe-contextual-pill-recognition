import os
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from PIL import ImageFile

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from train_pika_baseline import (
    DualEncoderPIKA,
    PIKADataset,
    add_mapped_columns,
    check_image_paths,
    build_transforms,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_label_to_idx(raw_mapping: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in raw_mapping.items()}


def normalize_idx_to_label(raw_mapping: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in raw_mapping.items()}


@torch.no_grad()
def predict_m2(model, loader, device, idx_to_label):
    model.eval()

    rows = []
    all_true_idx = []
    all_pred_idx = []

    for pill_imgs, pres_imgs, labels in loader:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        labels = labels.to(device)

        outputs = model(pill_imgs, pres_imgs)

        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        preds_cpu = preds.detach().cpu().numpy().tolist()
        labels_cpu = labels.detach().cpu().numpy().tolist()
        confs_cpu = confs.detach().cpu().numpy().tolist()

        for i in range(len(preds_cpu)):
            true_idx = int(labels_cpu[i])
            pred_idx = int(preds_cpu[i])

            rows.append({
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_label": int(idx_to_label[true_idx]),
                "pred_label": int(idx_to_label[pred_idx]),
                "confidence": float(confs_cpu[i]),
                "correct": int(true_idx == pred_idx),
            })

            all_true_idx.append(true_idx)
            all_pred_idx.append(pred_idx)

    return pd.DataFrame(rows), all_true_idx, all_pred_idx


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test CSV  :", args.test_csv)

    ckpt = safe_torch_load(args.checkpoint, device)

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt.get("num_classes", len(idx_to_label)))
    pill_model_name = ckpt.get("pill_model_name", args.pill_model_name)
    pres_model_name = ckpt.get("pres_model_name", args.pres_model_name)

    print("Model type        :", ckpt.get("model_type", "DualEncoderPIKA"))
    print("Pill model        :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Num classes       :", num_classes)

    test_df = pd.read_csv(args.test_csv)
    test_df = add_mapped_columns(test_df, label_to_idx)
    test_df = check_image_paths(test_df, "Test")

    _, val_tfms = build_transforms(args.image_size)

    test_dataset = PIKADataset(
        test_df,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    print("Test samples:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DualEncoderPIKA(
        num_classes=num_classes,
        pill_model_name=pill_model_name,
        pres_model_name=pres_model_name,
        pretrained=False,
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    pred_df, y_true, y_pred = predict_m2(
        model=model,
        loader=test_loader,
        device=device,
        idx_to_label=idx_to_label,
    )

    acc = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
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
        "model": "M2_split_pika_baseline",
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "num_test_samples": len(y_true),
        "num_classes": num_classes,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    predictions_path = os.path.join(args.output_dir, args.predictions_name)
    summary_path = os.path.join(args.output_dir, args.summary_name)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")

    pred_df.to_csv(predictions_path, index=False, encoding="utf-8")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8")

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
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
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
    parser = argparse.ArgumentParser(description="Evaluate M2 DualEncoderPIKA checkpoint on split test set")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--predictions_name", type=str, default="test_predictions.csv")
    parser.add_argument("--summary_name", type=str, default="test_metrics_summary.csv")

    args = parser.parse_args()
    main(args)

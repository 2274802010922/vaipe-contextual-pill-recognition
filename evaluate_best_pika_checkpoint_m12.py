import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    BestPIKAModel,
    BestPIKADataset,
    add_mapped_columns,
    check_image_paths,
    build_graph_matrix,
    build_transforms,
)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_label_to_idx(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def normalize_idx_to_label(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def set_dropout_p(model, dropout_p: float):
    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            changed += 1
    print(f"Updated Dropout modules: {changed} | dropout_p={dropout_p}")


def build_model_from_checkpoint(ckpt, adj_matrix, device, dropout_p=-1):
    num_classes = int(ckpt["num_classes"])
    pill_model_name = ckpt.get("pill_model_name", "tf_efficientnetv2_s.in21k_ft_in1k")
    pres_model_name = ckpt.get("pres_model_name", "resnet18.a1_in1k")
    hidden_dim = int(ckpt.get("hidden_dim", 256))

    print("Model type        :", ckpt.get("model_type", "BestPIKAModel"))
    print("Pill model        :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Hidden dim        :", hidden_dim)
    print("Num classes       :", num_classes)

    try:
        model = BestPIKAModel(
            num_classes=num_classes,
            adj_matrix=adj_matrix,
            pill_model_name=pill_model_name,
            pres_model_name=pres_model_name,
            hidden_dim=hidden_dim,
            pretrained=False,
        )
    except TypeError:
        model = BestPIKAModel(
            num_classes=num_classes,
            adj_matrix=adj_matrix,
            pill_model_name=pill_model_name,
            pres_model_name=pres_model_name,
            hidden_dim=hidden_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    if dropout_p >= 0:
        set_dropout_p(model, dropout_p)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    all_labels = []
    all_preds = []
    all_conf = []

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(loader, desc="Evaluating"):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(pill_imgs, pres_imgs, context_indices, context_mask)
            probs = torch.softmax(logits, dim=1)

        conf, preds = probs.max(dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)

    # Macro F1 over only classes that appear in test.
    macro_f1_present = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    # Macro F1 over all 108 classes.
    # This is stricter if some classes do not appear in the test CSV.
    macro_f1_all = f1_score(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    precision_present, recall_present, _, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    precision_all, recall_all, _, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    return {
        "labels": all_labels,
        "preds": all_preds,
        "confidence": all_conf,
        "accuracy": acc,
        "macro_f1_present_classes": macro_f1_present,
        "macro_precision_present_classes": precision_present,
        "macro_recall_present_classes": recall_present,
        "macro_f1_all_classes": macro_f1_all,
        "macro_precision_all_classes": precision_all,
        "macro_recall_all_classes": recall_all,
    }


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test CSV  :", args.test_csv)
    print("Data root :", args.data_root)

    ckpt = safe_torch_load(args.checkpoint, device)

    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not contain model_state_dict.")

    if "label_to_idx" not in ckpt or "idx_to_label" not in ckpt:
        raise RuntimeError("Checkpoint must contain label_to_idx and idx_to_label.")

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt.get("num_classes", len(label_to_idx)))
    max_context_len = int(ckpt.get("max_context_len", args.max_context_len))

    graph_labels_json = args.graph_labels_json
    graph_pmi_npy = args.graph_pmi_npy

    if graph_labels_json == "":
        graph_labels_json = os.path.join(args.data_root, "graph_labels.json")

    if graph_pmi_npy == "":
        graph_pmi_npy = os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

    test_df = pd.read_csv(args.test_csv)
    print("Raw test rows:", len(test_df))
    print("Raw test columns:", test_df.columns.tolist())

    test_df = add_mapped_columns(test_df, label_to_idx)
    test_df = check_image_paths(test_df, "Test")

    print("Test rows after image check:", len(test_df))
    print("Test labels present:", test_df["mapped_label"].nunique())
    print("Test mapped label min:", int(test_df["mapped_label"].min()))
    print("Test mapped label max:", int(test_df["mapped_label"].max()))

    sub_pmi = build_graph_matrix(
        graph_labels_json=graph_labels_json,
        graph_pmi_npy=graph_pmi_npy,
        idx_to_label=idx_to_label,
        device=device,
    )

    print("Graph PMI shape:", tuple(sub_pmi.shape))
    print("Max context length used:", max_context_len)

    _, val_tfms = build_transforms(args.image_size)

    test_dataset = BestPIKADataset(
        test_df,
        max_context_len=max_context_len,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model_from_checkpoint(
        ckpt=ckpt,
        adj_matrix=sub_pmi,
        device=device,
        dropout_p=args.dropout_p,
    )

    result = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    y_true = result["labels"]
    y_pred = result["preds"]
    conf = result["confidence"]

    print("\n=== TEST RESULTS ===")
    print(f"Test Accuracy                 : {result['accuracy']:.6f}")
    print(f"Test Macro Precision Present  : {result['macro_precision_present_classes']:.6f}")
    print(f"Test Macro Recall Present     : {result['macro_recall_present_classes']:.6f}")
    print(f"Test Macro F1 Present Classes : {result['macro_f1_present_classes']:.6f}")
    print(f"Test Macro Precision All      : {result['macro_precision_all_classes']:.6f}")
    print(f"Test Macro Recall All         : {result['macro_recall_all_classes']:.6f}")
    print(f"Test Macro F1 All Classes     : {result['macro_f1_all_classes']:.6f}")

    # Save metrics.
    metrics = {
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "num_test_rows": len(test_df),
        "num_classes": num_classes,
        "num_test_labels_present": int(test_df["mapped_label"].nunique()),
        "accuracy": result["accuracy"],
        "macro_precision_present_classes": result["macro_precision_present_classes"],
        "macro_recall_present_classes": result["macro_recall_present_classes"],
        "macro_f1_present_classes": result["macro_f1_present_classes"],
        "macro_precision_all_classes": result["macro_precision_all_classes"],
        "macro_recall_all_classes": result["macro_recall_all_classes"],
        "macro_f1_all_classes": result["macro_f1_all_classes"],
    }

    metrics_path = Path(args.output_dir) / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save predictions.
    pred_df = test_df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = conf
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]

    predictions_path = Path(args.output_dir) / "test_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    # Per-class report.
    labels_all = list(range(num_classes))
    target_names = [str(idx_to_label[i]) for i in labels_all]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T
    report_path = Path(args.output_dir) / "classification_report.csv"
    report_df.to_csv(report_path)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )

    per_class_df = pd.DataFrame({
        "mapped_label": labels_all,
        "original_label": [idx_to_label[i] for i in labels_all],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    per_class_path = Path(args.output_dir) / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    cm_path = Path(args.output_dir) / "confusion_matrix.csv"
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)

    print("\nSaved files:")
    print("Metrics              :", metrics_path)
    print("Predictions          :", predictions_path)
    print("Classification report:", report_path)
    print("Per-class metrics    :", per_class_path)
    print("Confusion matrix     :", cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Best PIKA checkpoint on an independent test CSV."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M12_stratified_balanced_focal/test_eval",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--dropout_p", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

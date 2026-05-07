import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm import tqdm

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    add_mapped_columns,
    build_transforms,
)

from train_m23_stage1_visual_classifier import (
    check_pill_paths,
    PillCropDataset,
    M23VisualClassifier,
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_metrics(y_true, y_pred, num_classes):
    labels_all = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)

    p_present, r_present, f1_present, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    p_all, r_all, f1_all, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        average="macro",
        zero_division=0,
    )

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_precision_present": float(p_present),
        "macro_recall_present": float(r_present),
        "macro_f1_present": float(f1_present),
        "macro_precision_all": float(p_all),
        "macro_recall_all": float(r_all),
        "macro_f1_all": float(f1_all),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
    }


def prepare_dataframe(csv_path, label_to_idx, split_name):
    df = pd.read_csv(csv_path)
    df = add_mapped_columns(df, label_to_idx)
    df = check_pill_paths(df, split_name)
    return df


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_conf = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating M23 visual"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)

        conf, preds = probs.max(dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())
        all_probs.append(probs.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    return {
        "labels": np.array(all_labels, dtype=np.int64),
        "preds": np.array(all_preds, dtype=np.int64),
        "confidence": np.array(all_conf, dtype=np.float32),
        "probs": all_probs.astype(np.float32),
    }


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== EVALUATE M23 STAGE1 VISUAL CLASSIFIER ===")
    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Eval CSV:", args.eval_csv)
    print("Output dir:", args.output_dir)

    ckpt = safe_torch_load(args.checkpoint, device)

    label_to_idx = normalize_label_to_idx(ckpt["label_to_idx"])
    idx_to_label = normalize_idx_to_label(ckpt["idx_to_label"])

    num_classes = int(ckpt["num_classes"])
    backbone_name = ckpt.get("backbone_name", "convnext_tiny.in12k_ft_in1k")
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout_p = float(ckpt.get("dropout_p", 0.55))
    image_size = int(ckpt.get("image_size", args.image_size))

    print("Model type:", ckpt.get("model_type"))
    print("Stage:", ckpt.get("stage"))
    print("Epoch:", ckpt.get("epoch"))
    print("Checkpoint Val Macro F1 Present:", ckpt.get("val_macro_f1_present"))
    print("Num classes:", num_classes)
    print("Backbone:", backbone_name)
    print("Hidden dim:", hidden_dim)
    print("Dropout:", dropout_p)
    print("Image size:", image_size)

    df = prepare_dataframe(args.eval_csv, label_to_idx, "Eval")

    print("\nEval rows:", len(df))
    print("Eval labels:", df["mapped_label"].nunique())
    print("Mapped label min:", int(df["mapped_label"].min()))
    print("Mapped label max:", int(df["mapped_label"].max()))

    _, val_tfms = build_transforms(image_size)

    dataset = PillCropDataset(df, transform=val_tfms)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = M23VisualClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    result = evaluate(model, loader, device)

    y_true = result["labels"]
    y_pred = result["preds"]
    conf = result["confidence"]
    probs = result["probs"]

    metrics = compute_metrics(y_true, y_pred, num_classes)

    metrics.update({
        "checkpoint": args.checkpoint,
        "eval_csv": args.eval_csv,
        "num_rows": int(len(df)),
        "num_classes": int(num_classes),
        "num_present_labels": int(df["mapped_label"].nunique()),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_val_macro_f1_present": float(ckpt.get("val_macro_f1_present", -1)),
        "stage": ckpt.get("stage"),
        "model_type": ckpt.get("model_type"),
        "backbone_name": backbone_name,
    })

    output_dir = Path(args.output_dir)

    print("\n=== M23 VISUAL EVAL RESULTS ===")
    print(f"Accuracy                 : {metrics['accuracy']:.6f}")
    print(f"Macro Precision Present  : {metrics['macro_precision_present']:.6f}")
    print(f"Macro Recall Present     : {metrics['macro_recall_present']:.6f}")
    print(f"Macro F1 Present Classes : {metrics['macro_f1_present']:.6f}")
    print(f"Macro F1 All Classes     : {metrics['macro_f1_all']:.6f}")
    print(f"Weighted F1              : {metrics['weighted_f1']:.6f}")

    save_json(metrics, output_dir / "m23_visual_eval_metrics.json")

    pred_df = df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = conf
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]

    pred_df.to_csv(output_dir / "m23_visual_eval_predictions.csv", index=False)
    np.save(output_dir / "m23_visual_eval_probs.npy", probs)
    np.save(output_dir / "m23_visual_eval_y_true.npy", y_true)
    np.save(output_dir / "m23_visual_eval_y_pred.npy", y_pred)

    labels_all = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "mapped_label": labels_all,
        "original_label": [idx_to_label[i] for i in labels_all],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    pred_count = pd.Series(y_pred).value_counts().rename("pred_count")
    per_class = per_class.merge(
        pred_count,
        left_on="mapped_label",
        right_index=True,
        how="left",
    )
    per_class["pred_count"] = per_class["pred_count"].fillna(0).astype(int)

    per_class.to_csv(output_dir / "m23_visual_per_class_metrics.csv", index=False)

    wrong = pred_df[pred_df["true_mapped_label"] != pred_df["pred_mapped_label"]].copy()

    conf_pairs = (
        wrong.groupby(["true_mapped_label", "pred_mapped_label"])
        .size()
        .reset_index(name="wrong_count")
        .sort_values("wrong_count", ascending=False)
    )

    true_support = pred_df["true_mapped_label"].value_counts().rename("true_support")
    pred_support = pred_df["pred_mapped_label"].value_counts().rename("pred_support")

    conf_pairs = conf_pairs.merge(
        true_support,
        left_on="true_mapped_label",
        right_index=True,
        how="left",
    )
    conf_pairs = conf_pairs.merge(
        pred_support,
        left_on="pred_mapped_label",
        right_index=True,
        how="left",
    )

    conf_pairs.to_csv(output_dir / "m23_visual_confusion_pairs.csv", index=False)

    print("\nSaved files:")
    print(output_dir / "m23_visual_eval_metrics.json")
    print(output_dir / "m23_visual_eval_predictions.csv")
    print(output_dir / "m23_visual_eval_probs.npy")
    print(output_dir / "m23_visual_per_class_metrics.csv")
    print(output_dir / "m23_visual_confusion_pairs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate M23 visual-only classifier."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

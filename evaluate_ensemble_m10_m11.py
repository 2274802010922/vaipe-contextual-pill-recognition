import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def check_mapping_compatible(name_a, map_a, name_b, map_b):
    if map_a != map_b:
        raise RuntimeError(
            f"Label mapping mismatch between {name_a} and {name_b}. "
            "Cannot safely ensemble checkpoints with different label mappings."
        )


def build_model_from_checkpoint(ckpt, adj_matrix, device, model_name):
    num_classes = int(ckpt["num_classes"])
    pill_model_name = ckpt.get("pill_model_name", "tf_efficientnetv2_s.in21k_ft_in1k")
    pres_model_name = ckpt.get("pres_model_name", "resnet18.a1_in1k")
    hidden_dim = int(ckpt.get("hidden_dim", 256))

    print(f"\n=== Build {model_name} ===")
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
    model.eval()

    print(f"Loaded {model_name} checkpoint successfully.")
    return model


@torch.no_grad()
def collect_probabilities(model_a, model_b, loader, device):
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = str(device).startswith("cuda")

    y_true_all = []
    probs_a_all = []
    probs_b_all = []

    model_a.eval()
    model_b.eval()

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(
        loader, desc="Collecting probabilities"
    ):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type, enabled=amp_enabled):
            logits_a = model_a(pill_imgs, pres_imgs, context_indices, context_mask)
            logits_b = model_b(pill_imgs, pres_imgs, context_indices, context_mask)

            probs_a = torch.softmax(logits_a, dim=1)
            probs_b = torch.softmax(logits_b, dim=1)

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        probs_a_all.append(probs_a.detach().cpu().numpy())
        probs_b_all.append(probs_b.detach().cpu().numpy())

    y_true_all = np.array(y_true_all, dtype=np.int64)
    probs_a_all = np.concatenate(probs_a_all, axis=0)
    probs_b_all = np.concatenate(probs_b_all, axis=0)

    return y_true_all, probs_a_all, probs_b_all


def compute_metrics(y_true, y_pred, num_classes):
    labels_all = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)

    macro_precision_present, macro_recall_present, macro_f1_present, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    macro_precision_all, macro_recall_all, macro_f1_all, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_all,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(acc),
        "macro_precision_present_classes": float(macro_precision_present),
        "macro_recall_present_classes": float(macro_recall_present),
        "macro_f1_present_classes": float(macro_f1_present),
        "macro_precision_all_classes": float(macro_precision_all),
        "macro_recall_all_classes": float(macro_recall_all),
        "macro_f1_all_classes": float(macro_f1_all),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def save_best_outputs(
    output_dir,
    test_df,
    y_true,
    y_pred,
    confidence,
    idx_to_label,
    num_classes,
    best_weight_m11,
    best_metrics,
    checkpoint_m10,
    checkpoint_m11,
    test_csv,
):
    output_dir = Path(output_dir)
    labels_all = list(range(num_classes))
    target_names = [str(idx_to_label[i]) for i in labels_all]

    best_config = {
        "checkpoint_m10": checkpoint_m10,
        "checkpoint_m11": checkpoint_m11,
        "test_csv": test_csv,
        "best_weight_m11": float(best_weight_m11),
        "best_weight_m10": float(1.0 - best_weight_m11),
        "num_classes": int(num_classes),
        "num_test_rows": int(len(test_df)),
        "num_test_labels_present": int(test_df["mapped_label"].nunique()),
        **best_metrics,
    }

    with open(output_dir / "best_ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    pred_df = test_df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = confidence
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]
    pred_df["ensemble_weight_m11"] = float(best_weight_m11)
    pred_df["ensemble_weight_m10"] = float(1.0 - best_weight_m11)

    pred_df.to_csv(output_dir / "best_ensemble_predictions.csv", index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(output_dir / "best_ensemble_classification_report.csv")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )

    per_class_df = pd.DataFrame(
        {
            "mapped_label": labels_all,
            "original_label": [idx_to_label[i] for i in labels_all],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    per_class_df.to_csv(output_dir / "best_ensemble_per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(output_dir / "best_ensemble_confusion_matrix.csv")

    print("\nSaved best ensemble files:")
    print("Metrics              :", output_dir / "best_ensemble_metrics.json")
    print("Predictions          :", output_dir / "best_ensemble_predictions.csv")
    print("Classification report:", output_dir / "best_ensemble_classification_report.csv")
    print("Per-class metrics    :", output_dir / "best_ensemble_per_class_metrics.csv")
    print("Confusion matrix     :", output_dir / "best_ensemble_confusion_matrix.csv")


def parse_weight_list(weight_text):
    weights = []
    for item in weight_text.split(","):
        item = item.strip()
        if item == "":
            continue
        weights.append(float(item))

    for w in weights:
        if w < 0 or w > 1:
            raise ValueError("All ensemble weights must be in [0, 1].")

    return weights


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("Using device:", device)
    print("M10 checkpoint:", args.checkpoint_m10)
    print("M11 checkpoint:", args.checkpoint_m11)
    print("Test CSV      :", args.test_csv)
    print("Data root     :", args.data_root)
    print("Output dir    :", args.output_dir)

    ckpt_m10 = safe_torch_load(args.checkpoint_m10, device)
    ckpt_m11 = safe_torch_load(args.checkpoint_m11, device)

    for name, ckpt in [("M10", ckpt_m10), ("M11", ckpt_m11)]:
        if "model_state_dict" not in ckpt:
            raise RuntimeError(f"{name} checkpoint does not contain model_state_dict.")
        if "label_to_idx" not in ckpt or "idx_to_label" not in ckpt:
            raise RuntimeError(f"{name} checkpoint must contain label_to_idx and idx_to_label.")

    label_to_idx_m10 = normalize_label_to_idx(ckpt_m10["label_to_idx"])
    idx_to_label_m10 = normalize_idx_to_label(ckpt_m10["idx_to_label"])

    label_to_idx_m11 = normalize_label_to_idx(ckpt_m11["label_to_idx"])
    idx_to_label_m11 = normalize_idx_to_label(ckpt_m11["idx_to_label"])

    check_mapping_compatible("M10 label_to_idx", label_to_idx_m10, "M11 label_to_idx", label_to_idx_m11)
    check_mapping_compatible("M10 idx_to_label", idx_to_label_m10, "M11 idx_to_label", idx_to_label_m11)

    label_to_idx = label_to_idx_m10
    idx_to_label = idx_to_label_m10

    num_classes_m10 = int(ckpt_m10.get("num_classes", len(label_to_idx)))
    num_classes_m11 = int(ckpt_m11.get("num_classes", len(label_to_idx)))

    if num_classes_m10 != num_classes_m11:
        raise RuntimeError(f"num_classes mismatch: M10={num_classes_m10}, M11={num_classes_m11}")

    num_classes = num_classes_m10

    max_context_len_m10 = int(ckpt_m10.get("max_context_len", args.max_context_len))
    max_context_len_m11 = int(ckpt_m11.get("max_context_len", args.max_context_len))

    if max_context_len_m10 != max_context_len_m11:
        raise RuntimeError(
            f"max_context_len mismatch: M10={max_context_len_m10}, M11={max_context_len_m11}"
        )

    max_context_len = max_context_len_m10

    graph_labels_json = args.graph_labels_json
    graph_pmi_npy = args.graph_pmi_npy

    if graph_labels_json == "":
        graph_labels_json = os.path.join(args.data_root, "graph_labels.json")

    if graph_pmi_npy == "":
        graph_pmi_npy = os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

    test_df = pd.read_csv(args.test_csv)
    print("\nRaw test rows:", len(test_df))
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

    model_m10 = build_model_from_checkpoint(
        ckpt=ckpt_m10,
        adj_matrix=sub_pmi,
        device=device,
        model_name="M10",
    )

    model_m11 = build_model_from_checkpoint(
        ckpt=ckpt_m11,
        adj_matrix=sub_pmi,
        device=device,
        model_name="M11",
    )

    y_true, probs_m10, probs_m11 = collect_probabilities(
        model_a=model_m10,
        model_b=model_m11,
        loader=test_loader,
        device=device,
    )

    print("\nCollected probabilities:")
    print("y_true    :", y_true.shape)
    print("probs_m10 :", probs_m10.shape)
    print("probs_m11 :", probs_m11.shape)

    weight_m11_list = parse_weight_list(args.weight_m11_list)

    rows = []
    best = None

    print("\n=== ENSEMBLE RESULTS ===")
    for weight_m11 in weight_m11_list:
        weight_m10 = 1.0 - weight_m11

        probs_ens = weight_m10 * probs_m10 + weight_m11 * probs_m11
        y_pred = probs_ens.argmax(axis=1)
        confidence = probs_ens.max(axis=1)

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes,
        )

        row = {
            "weight_m10": float(weight_m10),
            "weight_m11": float(weight_m11),
            **metrics,
        }

        rows.append(row)

        print(
            f"M10 {weight_m10:.2f} + M11 {weight_m11:.2f} | "
            f"Acc: {metrics['accuracy']:.6f} | "
            f"Macro F1 Present: {metrics['macro_f1_present_classes']:.6f} | "
            f"Macro F1 All: {metrics['macro_f1_all_classes']:.6f}"
        )

        if best is None or metrics["macro_f1_present_classes"] > best["metrics"]["macro_f1_present_classes"]:
            best = {
                "weight_m11": weight_m11,
                "weight_m10": weight_m10,
                "metrics": metrics,
                "y_pred": y_pred,
                "confidence": confidence,
            }

    summary_df = pd.DataFrame(rows).sort_values("macro_f1_present_classes", ascending=False)
    summary_path = Path(args.output_dir) / "ensemble_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== BEST ENSEMBLE ===")
    print(summary_df.head(1).to_string(index=False))
    print("\nSaved summary:", summary_path)

    save_best_outputs(
        output_dir=args.output_dir,
        test_df=test_df,
        y_true=y_true,
        y_pred=best["y_pred"],
        confidence=best["confidence"],
        idx_to_label=idx_to_label,
        num_classes=num_classes,
        best_weight_m11=best["weight_m11"],
        best_metrics=best["metrics"],
        checkpoint_m10=args.checkpoint_m10,
        checkpoint_m11=args.checkpoint_m11,
        test_csv=args.test_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate probability ensemble of M10 and M11 Best PIKA checkpoints."
    )

    parser.add_argument("--checkpoint_m10", type=str, required=True)
    parser.add_argument("--checkpoint_m11", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M13_ensemble_m10_m11/test_eval",
    )

    parser.add_argument(
        "--weight_m11_list",
        type=str,
        default="0.0,0.25,0.4,0.5,0.6,0.75,1.0",
        help="Comma-separated list of M11 weights. M10 weight = 1 - M11 weight.",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

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
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from train_best_pika_model import (
    seed_everything,
    ensure_dir,
    get_device,
    BestPIKADataset,
    add_mapped_columns,
    check_image_paths,
    build_transforms,
)

from train_m19_arch_pika_v1 import M19ArchitecturePIKA


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_label_to_idx(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def normalize_idx_to_label(raw_mapping):
    return {int(k): int(v) for k, v in raw_mapping.items()}


def load_graph_embeddings(checkpoint, graph_artifacts_dir):
    candidate_paths = []

    if checkpoint.get("graph_embeddings_path", ""):
        candidate_paths.append(Path(checkpoint["graph_embeddings_path"]))

    if graph_artifacts_dir:
        candidate_paths.append(Path(graph_artifacts_dir) / "graph_embeddings.npy")

    if checkpoint.get("graph_artifacts_dir", ""):
        candidate_paths.append(Path(checkpoint["graph_artifacts_dir"]) / "graph_embeddings.npy")

    for p in candidate_paths:
        if p.exists():
            print("Using graph embeddings:", p)
            return np.load(p).astype(np.float32), str(p)

    raise FileNotFoundError("Cannot find graph_embeddings.npy.")


def build_model_from_checkpoint(checkpoint, graph_embeddings, device):
    config = checkpoint.get("config", {})

    num_classes = int(checkpoint["num_classes"])
    pill_model_name = checkpoint.get(
        "pill_model_name",
        config.get("pill_model_name", "tf_efficientnetv2_s.in21k_ft_in1k"),
    )
    pres_model_name = checkpoint.get(
        "pres_model_name",
        config.get("pres_model_name", "resnet18.a1_in1k"),
    )
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("hidden_dim", 256)))
    graph_dim = int(checkpoint.get("graph_dim", graph_embeddings.shape[1]))
    dropout_p = float(config.get("dropout_p", 0.0))
    context_dropout_p = float(config.get("context_dropout_p", 0.0))
    train_graph_embeddings = bool(config.get("train_graph_embeddings", False))

    print("Model type        :", checkpoint.get("model_type", "M19ArchitecturePIKA"))
    print("Pill model        :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Hidden dim        :", hidden_dim)
    print("Graph dim         :", graph_dim)
    print("Num classes       :", num_classes)
    print("Dropout p         :", dropout_p)
    print("Context dropout p :", context_dropout_p)
    print("Train graph emb   :", train_graph_embeddings)

    model = M19ArchitecturePIKA(
        num_classes=num_classes,
        graph_embeddings=graph_embeddings,
        pill_model_name=pill_model_name,
        pres_model_name=pres_model_name,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        context_dropout_p=context_dropout_p,
        pretrained=False,
        train_graph_embeddings=train_graph_embeddings,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    all_labels = []
    all_preds = []
    all_conf = []
    all_pseudo_preds = []

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(
        loader, desc="Evaluating M19"
    ):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
            logits = outputs["main_logits"]
            pseudo_logits = outputs["pseudo_logits"]
            probs = torch.softmax(logits, dim=1)

        conf, preds = probs.max(dim=1)
        pseudo_preds = pseudo_logits.argmax(dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())
        all_pseudo_preds.extend(pseudo_preds.detach().cpu().numpy().tolist())

    labels_all = list(range(num_classes))

    accuracy = accuracy_score(all_labels, all_preds)

    macro_precision_present, macro_recall_present, macro_f1_present, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    macro_precision_all, macro_recall_all, macro_f1_all, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=labels_all,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "labels": all_labels,
        "preds": all_preds,
        "confidence": all_conf,
        "pseudo_preds": all_pseudo_preds,
        "accuracy": float(accuracy),
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


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== EVALUATE M19 ARCHITECTURE PIKA V1 ===")
    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test CSV  :", args.test_csv)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    checkpoint = safe_torch_load(args.checkpoint, device)

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain model_state_dict.")

    label_to_idx = normalize_label_to_idx(checkpoint["label_to_idx"])
    idx_to_label = normalize_idx_to_label(checkpoint["idx_to_label"])

    num_classes = int(checkpoint["num_classes"])
    max_context_len = int(checkpoint.get("max_context_len", args.max_context_len))

    graph_embeddings, graph_embeddings_path = load_graph_embeddings(
        checkpoint=checkpoint,
        graph_artifacts_dir=args.graph_artifacts_dir,
    )

    print("Num classes:", num_classes)
    print("Max context len:", max_context_len)
    print("Graph embeddings shape:", graph_embeddings.shape)

    test_df = pd.read_csv(args.test_csv)

    print("\nRaw test rows:", len(test_df))
    print("Raw test labels:", test_df["pill_label"].nunique())
    print("Raw test columns:", test_df.columns.tolist())

    test_df = add_mapped_columns(test_df, label_to_idx)
    test_df = check_image_paths(test_df, "Test")

    print("\nTest rows after image check:", len(test_df))
    print("Test labels present:", test_df["mapped_label"].nunique())
    print("Test mapped label min:", int(test_df["mapped_label"].min()))
    print("Test mapped label max:", int(test_df["mapped_label"].max()))

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
        checkpoint=checkpoint,
        graph_embeddings=graph_embeddings,
        device=device,
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
    pseudo_pred = result["pseudo_preds"]

    print("\n=== M19 TEST RESULTS ===")
    print(f"Test Accuracy                 : {result['accuracy']:.6f}")
    print(f"Test Macro Precision Present  : {result['macro_precision_present_classes']:.6f}")
    print(f"Test Macro Recall Present     : {result['macro_recall_present_classes']:.6f}")
    print(f"Test Macro F1 Present Classes : {result['macro_f1_present_classes']:.6f}")
    print(f"Test Macro Precision All      : {result['macro_precision_all_classes']:.6f}")
    print(f"Test Macro Recall All         : {result['macro_recall_all_classes']:.6f}")
    print(f"Test Macro F1 All Classes     : {result['macro_f1_all_classes']:.6f}")
    print(f"Test Weighted F1              : {result['weighted_f1']:.6f}")

    metrics = {
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "graph_embeddings_path": graph_embeddings_path,
        "num_test_rows": int(len(test_df)),
        "num_classes": int(num_classes),
        "num_test_labels_present": int(test_df["mapped_label"].nunique()),
        "checkpoint_val_macro_f1": float(checkpoint.get("val_macro_f1", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "accuracy": result["accuracy"],
        "macro_precision_present_classes": result["macro_precision_present_classes"],
        "macro_recall_present_classes": result["macro_recall_present_classes"],
        "macro_f1_present_classes": result["macro_f1_present_classes"],
        "macro_precision_all_classes": result["macro_precision_all_classes"],
        "macro_recall_all_classes": result["macro_recall_all_classes"],
        "macro_f1_all_classes": result["macro_f1_all_classes"],
        "weighted_precision": result["weighted_precision"],
        "weighted_recall": result["weighted_recall"],
        "weighted_f1": result["weighted_f1"],
    }

    output_dir = Path(args.output_dir)

    with open(output_dir / "m19_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = test_df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["pseudo_pred_mapped_label"] = pseudo_pred
    pred_df["confidence"] = conf
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["pseudo_pred_original_label"] = [idx_to_label[int(x)] for x in pseudo_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]

    pred_df.to_csv(output_dir / "m19_test_predictions.csv", index=False)

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

    pd.DataFrame(report_dict).T.to_csv(output_dir / "m19_classification_report.csv")

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

    per_class_df.to_csv(output_dir / "m19_per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        output_dir / "m19_confusion_matrix.csv"
    )

    print("\nSaved files:")
    print(output_dir / "m19_test_metrics.json")
    print(output_dir / "m19_test_predictions.csv")
    print(output_dir / "m19_classification_report.csv")
    print(output_dir / "m19_per_class_metrics.csv")
    print(output_dir / "m19_confusion_matrix.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate M19 architecture PIKA v1 checkpoint on clean test split."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M19_arch_pika_v1/test_eval",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

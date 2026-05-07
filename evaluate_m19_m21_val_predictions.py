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
from train_m21_strong_visual_pika import M21StrongVisualPIKA


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def compute_metrics(y_true, y_pred, num_classes):
    labels_all = list(range(num_classes))

    accuracy = accuracy_score(y_true, y_pred)

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

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision_present": float(p_present),
        "macro_recall_present": float(r_present),
        "macro_f1_present": float(f1_present),
        "macro_precision_all": float(p_all),
        "macro_recall_all": float(r_all),
        "macro_f1_all": float(f1_all),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
    }


def build_m19_from_checkpoint(checkpoint, graph_embeddings, device):
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

    print("\n=== Build M19 ===")
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


def build_m21_from_checkpoint(checkpoint, graph_embeddings, device):
    config = checkpoint.get("config", {})

    num_classes = int(checkpoint["num_classes"])
    pill_model_name = checkpoint.get(
        "pill_model_name",
        config.get("pill_model_name", "convnext_tiny.in12k_ft_in1k"),
    )
    pres_model_name = checkpoint.get(
        "pres_model_name",
        config.get("pres_model_name", "resnet18.a1_in1k"),
    )
    common_dim = int(checkpoint.get("common_dim", config.get("common_dim", 128)))
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("hidden_dim", 384)))
    dropout_p = float(config.get("dropout_p", 0.0))
    context_dropout_p = float(config.get("context_dropout_p", 0.0))
    train_graph_embeddings = bool(config.get("train_graph_embeddings", False))

    print("\n=== Build M21 ===")
    print("Model type        :", checkpoint.get("model_type", "M21StrongVisualPIKA"))
    print("Pill model        :", pill_model_name)
    print("Prescription model:", pres_model_name)
    print("Common dim        :", common_dim)
    print("Hidden dim        :", hidden_dim)
    print("Num classes       :", num_classes)
    print("Dropout p         :", dropout_p)
    print("Context dropout p :", context_dropout_p)
    print("Train graph emb   :", train_graph_embeddings)

    model = M21StrongVisualPIKA(
        num_classes=num_classes,
        graph_embeddings=graph_embeddings,
        pill_model_name=pill_model_name,
        pres_model_name=pres_model_name,
        common_dim=common_dim,
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
def collect_probabilities(model, loader, device, desc):
    all_true = []
    all_pred = []
    all_conf = []
    all_probs = []

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(loader, desc=desc):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(pill_imgs, pres_imgs, context_indices, context_mask)
            logits = outputs["main_logits"]
            probs = torch.softmax(logits, dim=1)

        conf, pred = probs.max(dim=1)

        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_conf.extend(conf.detach().cpu().numpy().tolist())
        all_probs.append(probs.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    return {
        "y_true": np.array(all_true, dtype=np.int64),
        "y_pred": np.array(all_pred, dtype=np.int64),
        "confidence": np.array(all_conf, dtype=np.float32),
        "probs": all_probs.astype(np.float32),
    }


def prepare_eval_dataframe(eval_csv, label_to_idx):
    df = pd.read_csv(eval_csv)

    print("\nRaw eval rows:", len(df))
    print("Raw eval labels:", df["pill_label"].nunique())
    print("Raw eval columns:", df.columns.tolist())

    df = add_mapped_columns(df, label_to_idx)
    df = check_image_paths(df, "Eval")

    print("\nEval rows after image check:", len(df))
    print("Eval labels present:", df["mapped_label"].nunique())
    print("Eval mapped label min:", int(df["mapped_label"].min()))
    print("Eval mapped label max:", int(df["mapped_label"].max()))

    return df


def save_prediction_outputs(
    output_dir,
    prefix,
    eval_df,
    idx_to_label,
    result,
    num_classes,
    checkpoint_path,
    checkpoint,
    graph_embeddings_path,
    eval_csv,
):
    output_dir = Path(output_dir)

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    confidence = result["confidence"]
    probs = result["probs"]

    metrics = compute_metrics(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        num_classes=num_classes,
    )

    metrics.update({
        "prefix": prefix,
        "checkpoint": checkpoint_path,
        "eval_csv": eval_csv,
        "num_rows": int(len(eval_df)),
        "num_classes": int(num_classes),
        "num_present_labels": int(eval_df["mapped_label"].nunique()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_macro_f1": float(checkpoint.get("val_macro_f1", -1)),
        "graph_embeddings_path": graph_embeddings_path,
    })

    pred_df = eval_df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = confidence
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]

    pred_csv = output_dir / f"{prefix}_val_predictions.csv"
    probs_npy = output_dir / f"{prefix}_val_probs.npy"
    y_true_npy = output_dir / f"{prefix}_val_y_true.npy"
    y_pred_npy = output_dir / f"{prefix}_val_y_pred.npy"
    metrics_json = output_dir / f"{prefix}_val_metrics.json"

    pred_df.to_csv(pred_csv, index=False)
    np.save(probs_npy, probs)
    np.save(y_true_npy, y_true)
    np.save(y_pred_npy, y_pred)
    save_json(metrics, metrics_json)

    print(f"\n=== {prefix.upper()} VAL METRICS ===")
    print(f"Accuracy           : {metrics['accuracy']:.6f}")
    print(f"Macro F1 Present   : {metrics['macro_f1_present']:.6f}")
    print(f"Macro F1 All       : {metrics['macro_f1_all']:.6f}")
    print(f"Weighted F1        : {metrics['weighted_f1']:.6f}")

    print("\nSaved files:")
    print(pred_csv)
    print(probs_npy)
    print(y_true_npy)
    print(y_pred_npy)
    print(metrics_json)

    return metrics


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("=== EVALUATE M19 + M21 ON VALIDATION ===")
    print("Using device:", device)
    print("Eval CSV:", args.eval_csv)
    print("M19 checkpoint:", args.m19_checkpoint)
    print("M21 checkpoint:", args.m21_checkpoint)
    print("Graph artifacts dir:", args.graph_artifacts_dir)
    print("Output dir:", args.output_dir)

    m19_ckpt = safe_torch_load(args.m19_checkpoint, device)
    m21_ckpt = safe_torch_load(args.m21_checkpoint, device)

    label_to_idx_m19 = normalize_label_to_idx(m19_ckpt["label_to_idx"])
    label_to_idx_m21 = normalize_label_to_idx(m21_ckpt["label_to_idx"])

    idx_to_label_m19 = normalize_idx_to_label(m19_ckpt["idx_to_label"])
    idx_to_label_m21 = normalize_idx_to_label(m21_ckpt["idx_to_label"])

    if label_to_idx_m19 != label_to_idx_m21:
        raise RuntimeError("M19 and M21 label_to_idx mappings are different. Cannot ensemble safely.")

    if idx_to_label_m19 != idx_to_label_m21:
        raise RuntimeError("M19 and M21 idx_to_label mappings are different. Cannot ensemble safely.")

    label_to_idx = label_to_idx_m19
    idx_to_label = idx_to_label_m19

    num_classes = int(m19_ckpt["num_classes"])

    if int(m21_ckpt["num_classes"]) != num_classes:
        raise RuntimeError("M19 and M21 have different num_classes.")

    graph_embeddings_m19, graph_path_m19 = load_graph_embeddings(
        checkpoint=m19_ckpt,
        graph_artifacts_dir=args.graph_artifacts_dir,
    )

    graph_embeddings_m21, graph_path_m21 = load_graph_embeddings(
        checkpoint=m21_ckpt,
        graph_artifacts_dir=args.graph_artifacts_dir,
    )

    print("\nGraph M19 shape:", graph_embeddings_m19.shape)
    print("Graph M21 shape:", graph_embeddings_m21.shape)

    eval_df = prepare_eval_dataframe(args.eval_csv, label_to_idx)

    _, val_tfms = build_transforms(args.image_size)

    max_context_len_m19 = int(m19_ckpt.get("max_context_len", 5))
    max_context_len_m21 = int(m21_ckpt.get("max_context_len", 8))

    print("\nM19 max_context_len:", max_context_len_m19)
    print("M21 max_context_len:", max_context_len_m21)

    dataset_m19 = BestPIKADataset(
        eval_df,
        max_context_len=max_context_len_m19,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    dataset_m21 = BestPIKADataset(
        eval_df,
        max_context_len=max_context_len_m21,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    loader_m19 = DataLoader(
        dataset_m19,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    loader_m21 = DataLoader(
        dataset_m21,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_m19 = build_m19_from_checkpoint(
        checkpoint=m19_ckpt,
        graph_embeddings=graph_embeddings_m19,
        device=device,
    )

    model_m21 = build_m21_from_checkpoint(
        checkpoint=m21_ckpt,
        graph_embeddings=graph_embeddings_m21,
        device=device,
    )

    print("\nCollecting M19 validation probabilities...")
    result_m19 = collect_probabilities(
        model=model_m19,
        loader=loader_m19,
        device=device,
        desc="M19 val",
    )

    print("\nCollecting M21 validation probabilities...")
    result_m21 = collect_probabilities(
        model=model_m21,
        loader=loader_m21,
        device=device,
        desc="M21 val",
    )

    if not np.array_equal(result_m19["y_true"], result_m21["y_true"]):
        raise RuntimeError("M19 and M21 y_true arrays are not aligned.")

    metrics_m19 = save_prediction_outputs(
        output_dir=args.output_dir,
        prefix="m19",
        eval_df=eval_df,
        idx_to_label=idx_to_label,
        result=result_m19,
        num_classes=num_classes,
        checkpoint_path=args.m19_checkpoint,
        checkpoint=m19_ckpt,
        graph_embeddings_path=graph_path_m19,
        eval_csv=args.eval_csv,
    )

    metrics_m21 = save_prediction_outputs(
        output_dir=args.output_dir,
        prefix="m21",
        eval_df=eval_df,
        idx_to_label=idx_to_label,
        result=result_m21,
        num_classes=num_classes,
        checkpoint_path=args.m21_checkpoint,
        checkpoint=m21_ckpt,
        graph_embeddings_path=graph_path_m21,
        eval_csv=args.eval_csv,
    )

    summary = pd.DataFrame([
        {"model": "m19", **metrics_m19},
        {"model": "m21", **metrics_m21},
    ])

    summary_path = Path(args.output_dir) / "val_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY ===")
    print(summary[[
        "model",
        "accuracy",
        "macro_f1_present",
        "macro_f1_all",
        "weighted_f1",
        "checkpoint_epoch",
        "checkpoint_val_macro_f1",
    ]].to_string(index=False))

    print("\nSaved summary:", summary_path)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect M19 and M21 validation predictions/probabilities for M22 ensemble."
    )

    parser.add_argument("--eval_csv", type=str, required=True)
    parser.add_argument("--m19_checkpoint", type=str, required=True)
    parser.add_argument("--m21_checkpoint", type=str, required=True)
    parser.add_argument("--graph_artifacts_dir", type=str, required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M22_selective_ensemble/val_predictions",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

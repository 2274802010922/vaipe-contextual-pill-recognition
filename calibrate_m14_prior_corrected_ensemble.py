import os
import json
import argparse
from pathlib import Path
from collections import Counter

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
def collect_probabilities(model_m10, model_m11, loader, device):
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = str(device).startswith("cuda")

    y_true_all = []
    probs_m10_all = []
    probs_m11_all = []

    model_m10.eval()
    model_m11.eval()

    for pill_imgs, pres_imgs, context_indices, context_mask, labels in tqdm(
        loader, desc="Collecting probabilities"
    ):
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        context_indices = context_indices.to(device)
        context_mask = context_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type, enabled=amp_enabled):
            logits_m10 = model_m10(pill_imgs, pres_imgs, context_indices, context_mask)
            logits_m11 = model_m11(pill_imgs, pres_imgs, context_indices, context_mask)

            probs_m10 = torch.softmax(logits_m10, dim=1)
            probs_m11 = torch.softmax(logits_m11, dim=1)

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        probs_m10_all.append(probs_m10.detach().cpu().numpy())
        probs_m11_all.append(probs_m11.detach().cpu().numpy())

    y_true_all = np.array(y_true_all, dtype=np.int64)
    probs_m10_all = np.concatenate(probs_m10_all, axis=0)
    probs_m11_all = np.concatenate(probs_m11_all, axis=0)

    return y_true_all, probs_m10_all, probs_m11_all


def build_loader(csv_path, label_to_idx, max_context_len, image_size, batch_size, num_workers, split_name):
    df = pd.read_csv(csv_path)

    print(f"\n=== Load {split_name} CSV ===")
    print("CSV:", csv_path)
    print("Raw rows:", len(df))
    print("Columns:", df.columns.tolist())

    df = add_mapped_columns(df, label_to_idx)
    df = check_image_paths(df, split_name)

    print(f"{split_name} rows after image check:", len(df))
    print(f"{split_name} labels present:", df["mapped_label"].nunique())

    _, val_tfms = build_transforms(image_size)

    dataset = BestPIKADataset(
        df,
        max_context_len=max_context_len,
        pill_transform=val_tfms,
        pres_transform=val_tfms,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return df, loader


def parse_float_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def compute_train_prior(train_csv, label_to_idx, num_classes, label_col="pill_label", smoothing=1.0):
    train_df = pd.read_csv(train_csv)

    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in train CSV.")

    train_df = add_mapped_columns(train_df, label_to_idx)

    counts = np.zeros(num_classes, dtype=np.float64)

    for y in train_df["mapped_label"].tolist():
        counts[int(y)] += 1.0

    counts = counts + smoothing
    prior = counts / counts.sum()

    print("\n=== Train prior ===")
    print("Train CSV:", train_csv)
    print("Prior smoothing:", smoothing)
    print("Prior min:", float(prior.min()))
    print("Prior max:", float(prior.max()))
    print("Most frequent class:", int(prior.argmax()), "prior:", float(prior.max()))

    return prior


def predict_with_prior_correction(probs_m10, probs_m11, weight_m11, prior, tau, eps=1e-12):
    weight_m10 = 1.0 - weight_m11

    probs = weight_m10 * probs_m10 + weight_m11 * probs_m11

    log_scores = np.log(probs + eps) - tau * np.log(prior.reshape(1, -1) + eps)

    y_pred = log_scores.argmax(axis=1)

    exp_scores = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
    adjusted_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    confidence = adjusted_probs.max(axis=1)

    return y_pred, confidence


def compute_metrics(y_true, y_pred, num_classes):
    labels_all = list(range(num_classes))

    accuracy = accuracy_score(y_true, y_pred)

    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    mp_all, mr_all, mf1_all, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        average="macro",
        zero_division=0,
    )

    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision_present_classes": float(mp),
        "macro_recall_present_classes": float(mr),
        "macro_f1_present_classes": float(mf1),
        "macro_precision_all_classes": float(mp_all),
        "macro_recall_all_classes": float(mr_all),
        "macro_f1_all_classes": float(mf1_all),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf1),
    }


def tune_on_calibration(y_cal, probs10_cal, probs11_cal, prior, num_classes, weight_list, tau_list):
    rows = []
    best = None

    print("\n=== CALIBRATION GRID SEARCH ===")

    for weight_m11 in weight_list:
        for tau in tau_list:
            y_pred, conf = predict_with_prior_correction(
                probs_m10=probs10_cal,
                probs_m11=probs11_cal,
                weight_m11=weight_m11,
                prior=prior,
                tau=tau,
            )

            metrics = compute_metrics(y_cal, y_pred, num_classes)

            row = {
                "weight_m10": float(1.0 - weight_m11),
                "weight_m11": float(weight_m11),
                "tau": float(tau),
                **metrics,
            }

            rows.append(row)

            if best is None or metrics["macro_f1_present_classes"] > best["metrics"]["macro_f1_present_classes"]:
                best = {
                    "weight_m11": float(weight_m11),
                    "weight_m10": float(1.0 - weight_m11),
                    "tau": float(tau),
                    "metrics": metrics,
                }

            print(
                f"w11={weight_m11:.3f} tau={tau:.3f} | "
                f"Acc={metrics['accuracy']:.6f} | "
                f"MacroF1={metrics['macro_f1_present_classes']:.6f}"
            )

    summary_df = pd.DataFrame(rows).sort_values("macro_f1_present_classes", ascending=False)

    print("\n=== BEST CALIBRATION CONFIG ===")
    print(summary_df.head(1).to_string(index=False))

    return best, summary_df


def save_test_outputs(
    output_dir,
    test_df,
    y_true,
    y_pred,
    confidence,
    idx_to_label,
    num_classes,
    config,
    metrics,
):
    output_dir = Path(output_dir)
    labels_all = list(range(num_classes))
    target_names = [str(idx_to_label[i]) for i in labels_all]

    metrics_out = {
        **config,
        **metrics,
        "num_test_rows": int(len(test_df)),
        "num_test_labels_present": int(test_df["mapped_label"].nunique()),
    }

    with open(output_dir / "m14_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    pred_df = test_df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = confidence
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]
    pred_df["m14_weight_m10"] = config["weight_m10"]
    pred_df["m14_weight_m11"] = config["weight_m11"]
    pred_df["m14_tau"] = config["tau"]

    pred_df.to_csv(output_dir / "m14_test_predictions.csv", index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).T.to_csv(output_dir / "m14_classification_report.csv")

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

    per_class_df.to_csv(output_dir / "m14_per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        output_dir / "m14_confusion_matrix.csv"
    )

    print("\nSaved M14 test files:")
    print(output_dir / "m14_test_metrics.json")
    print(output_dir / "m14_test_predictions.csv")
    print(output_dir / "m14_classification_report.csv")
    print(output_dir / "m14_per_class_metrics.csv")
    print(output_dir / "m14_confusion_matrix.csv")


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("Using device:", device)
    print("M10 checkpoint:", args.checkpoint_m10)
    print("M11 checkpoint:", args.checkpoint_m11)
    print("Train CSV     :", args.train_csv)
    print("Calibration CSV:", args.calibration_csv)
    print("Test CSV      :", args.test_csv)
    print("Output dir    :", args.output_dir)

    ckpt_m10 = safe_torch_load(args.checkpoint_m10, device)
    ckpt_m11 = safe_torch_load(args.checkpoint_m11, device)

    label_to_idx_m10 = normalize_label_to_idx(ckpt_m10["label_to_idx"])
    idx_to_label_m10 = normalize_idx_to_label(ckpt_m10["idx_to_label"])

    label_to_idx_m11 = normalize_label_to_idx(ckpt_m11["label_to_idx"])
    idx_to_label_m11 = normalize_idx_to_label(ckpt_m11["idx_to_label"])

    check_mapping_compatible("M10 label_to_idx", label_to_idx_m10, "M11 label_to_idx", label_to_idx_m11)
    check_mapping_compatible("M10 idx_to_label", idx_to_label_m10, "M11 idx_to_label", idx_to_label_m11)

    label_to_idx = label_to_idx_m10
    idx_to_label = idx_to_label_m10

    num_classes = int(ckpt_m10.get("num_classes", len(label_to_idx)))
    max_context_len = int(ckpt_m10.get("max_context_len", args.max_context_len))

    graph_labels_json = args.graph_labels_json or os.path.join(args.data_root, "graph_labels.json")
    graph_pmi_npy = args.graph_pmi_npy or os.path.join(args.data_root, "graph_pmi.npy")

    print("Graph labels:", graph_labels_json)
    print("Graph PMI   :", graph_pmi_npy)

    sub_pmi = build_graph_matrix(
        graph_labels_json=graph_labels_json,
        graph_pmi_npy=graph_pmi_npy,
        idx_to_label=idx_to_label,
        device=device,
    )

    print("Graph PMI shape:", tuple(sub_pmi.shape))
    print("Max context length used:", max_context_len)

    model_m10 = build_model_from_checkpoint(ckpt_m10, sub_pmi, device, "M10")
    model_m11 = build_model_from_checkpoint(ckpt_m11, sub_pmi, device, "M11")

    prior = compute_train_prior(
        train_csv=args.train_csv,
        label_to_idx=label_to_idx,
        num_classes=num_classes,
        label_col=args.label_col,
        smoothing=args.prior_smoothing,
    )

    cal_df, cal_loader = build_loader(
        csv_path=args.calibration_csv,
        label_to_idx=label_to_idx,
        max_context_len=max_context_len,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_name="Calibration",
    )

    test_df, test_loader = build_loader(
        csv_path=args.test_csv,
        label_to_idx=label_to_idx,
        max_context_len=max_context_len,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_name="Test",
    )

    print("\nCollect calibration probabilities")
    y_cal, probs10_cal, probs11_cal = collect_probabilities(
        model_m10, model_m11, cal_loader, device
    )

    print("\nCollect test probabilities")
    y_test, probs10_test, probs11_test = collect_probabilities(
        model_m10, model_m11, test_loader, device
    )

    weight_list = parse_float_list(args.weight_m11_list)
    tau_list = parse_float_list(args.tau_list)

    best, cal_summary_df = tune_on_calibration(
        y_cal=y_cal,
        probs10_cal=probs10_cal,
        probs11_cal=probs11_cal,
        prior=prior,
        num_classes=num_classes,
        weight_list=weight_list,
        tau_list=tau_list,
    )

    cal_summary_path = Path(args.output_dir) / "m14_calibration_grid_summary.csv"
    cal_summary_df.to_csv(cal_summary_path, index=False)

    best_config = {
        "weight_m10": best["weight_m10"],
        "weight_m11": best["weight_m11"],
        "tau": best["tau"],
        "selected_by": "calibration_macro_f1_present_classes",
        "calibration_csv": args.calibration_csv,
        "test_csv": args.test_csv,
        "checkpoint_m10": args.checkpoint_m10,
        "checkpoint_m11": args.checkpoint_m11,
    }

    with open(Path(args.output_dir) / "m14_best_calibration_config.json", "w", encoding="utf-8") as f:
        json.dump({**best_config, **best["metrics"]}, f, ensure_ascii=False, indent=2)

    print("\n=== APPLY BEST CONFIG TO TEST ===")
    print(json.dumps(best_config, indent=2, ensure_ascii=False))

    y_pred_test, conf_test = predict_with_prior_correction(
        probs_m10=probs10_test,
        probs_m11=probs11_test,
        weight_m11=best["weight_m11"],
        prior=prior,
        tau=best["tau"],
    )

    test_metrics = compute_metrics(y_test, y_pred_test, num_classes)

    print("\n=== M14 TEST RESULTS ===")
    print(f"Test Accuracy                 : {test_metrics['accuracy']:.6f}")
    print(f"Test Macro Precision Present  : {test_metrics['macro_precision_present_classes']:.6f}")
    print(f"Test Macro Recall Present     : {test_metrics['macro_recall_present_classes']:.6f}")
    print(f"Test Macro F1 Present Classes : {test_metrics['macro_f1_present_classes']:.6f}")
    print(f"Test Macro Precision All      : {test_metrics['macro_precision_all_classes']:.6f}")
    print(f"Test Macro Recall All         : {test_metrics['macro_recall_all_classes']:.6f}")
    print(f"Test Macro F1 All Classes     : {test_metrics['macro_f1_all_classes']:.6f}")

    save_test_outputs(
        output_dir=args.output_dir,
        test_df=test_df,
        y_true=y_test,
        y_pred=y_pred_test,
        confidence=conf_test,
        idx_to_label=idx_to_label,
        num_classes=num_classes,
        config=best_config,
        metrics=test_metrics,
    )

    print("\nSaved calibration summary:", cal_summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M14 prior-corrected ensemble calibrated on validation and evaluated on test."
    )

    parser.add_argument("--checkpoint_m10", type=str, required=True)
    parser.add_argument("--checkpoint_m11", type=str, required=True)

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--calibration_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M14_prior_corrected_ensemble/test_eval",
    )

    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--prior_smoothing", type=float, default=1.0)

    parser.add_argument(
        "--weight_m11_list",
        type=str,
        default="0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58",
    )

    parser.add_argument(
        "--tau_list",
        type=str,
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.70,1.00",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

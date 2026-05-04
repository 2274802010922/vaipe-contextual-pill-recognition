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


def parse_int_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_directed_pairs(text, label_to_idx):
    """
    Format:
        24>89,7>46,19>104

    Meaning:
        True class 24 is often predicted as 89.
        If model predicts 89 and class 24 is close in top-k, switch 89 -> 24.
    """
    pairs = []

    for raw_item in text.split(","):
        item = raw_item.strip()

        if not item:
            continue

        if ">" not in item:
            raise ValueError(
                f"Invalid pair '{item}'. Use format like 24>89, meaning true 24 is predicted as 89."
            )

        true_original, pred_original = item.split(">")
        true_original = int(true_original.strip())
        pred_original = int(pred_original.strip())

        if true_original not in label_to_idx:
            print(f"Skip pair {item}: true label not found in mapping.")
            continue

        if pred_original not in label_to_idx:
            print(f"Skip pair {item}: predicted label not found in mapping.")
            continue

        pairs.append(
            {
                "true_original": true_original,
                "pred_original": pred_original,
                "true_mapped": int(label_to_idx[true_original]),
                "pred_mapped": int(label_to_idx[pred_original]),
            }
        )

    return pairs


def ensemble_probs(probs_m10, probs_m11, weight_m11):
    weight_m10 = 1.0 - weight_m11
    return weight_m10 * probs_m10 + weight_m11 * probs_m11


def compute_rank_matrix(probs):
    sorted_idx = np.argsort(-probs, axis=1)
    rank = np.empty_like(sorted_idx)
    rank[np.arange(probs.shape[0])[:, None], sorted_idx] = np.arange(probs.shape[1])[None, :]
    return rank


def apply_rules(probs, rules):
    """
    Apply selected top-k correction rules in order.

    Rule condition:
        current prediction == pred_mapped
        true_mapped class rank <= top_k
        prob[pred_mapped] - prob[true_mapped] <= margin
        prob[pred_mapped] <= max_pred_prob
        prob[true_mapped] >= min_alt_prob
    """
    y_pred = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float64)
    applied_rule = np.array([""] * len(y_pred), dtype=object)

    rank = compute_rank_matrix(probs)

    for rule in rules:
        true_mapped = int(rule["true_mapped"])
        pred_mapped = int(rule["pred_mapped"])
        top_k = int(rule["top_k"])
        margin = float(rule["margin"])
        max_pred_prob = float(rule["max_pred_prob"])
        min_alt_prob = float(rule["min_alt_prob"])

        current_pred_prob = probs[:, pred_mapped]
        alt_prob = probs[:, true_mapped]

        mask = (
            (y_pred == pred_mapped)
            & (rank[:, true_mapped] < top_k)
            & ((current_pred_prob - alt_prob) <= margin)
            & (current_pred_prob <= max_pred_prob)
            & (alt_prob >= min_alt_prob)
        )

        y_pred[mask] = true_mapped
        confidence[mask] = alt_prob[mask]
        applied_rule[mask] = (
            f"{rule['true_original']}>{rule['pred_original']}"
            f"|k={top_k}|m={margin}|maxp={max_pred_prob}|min_alt={min_alt_prob}"
        )

    return y_pred, confidence, applied_rule


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


def make_candidate_rules(base_pairs, top_k_values, margin_values, max_pred_prob_values, min_alt_prob_values):
    candidates = []

    for pair in base_pairs:
        for top_k in top_k_values:
            for margin in margin_values:
                for max_pred_prob in max_pred_prob_values:
                    for min_alt_prob in min_alt_prob_values:
                        rule = dict(pair)
                        rule["top_k"] = int(top_k)
                        rule["margin"] = float(margin)
                        rule["max_pred_prob"] = float(max_pred_prob)
                        rule["min_alt_prob"] = float(min_alt_prob)
                        candidates.append(rule)

    return candidates


def greedy_select_rules(
    probs_cal,
    y_cal,
    base_pairs,
    num_classes,
    top_k_values,
    margin_values,
    max_pred_prob_values,
    min_alt_prob_values,
    max_rules,
    min_improvement,
):
    selected_rules = []

    base_pred = probs_cal.argmax(axis=1)
    best_metrics = compute_metrics(y_cal, base_pred, num_classes)
    best_f1 = best_metrics["macro_f1_present_classes"]

    print("\n=== M15 GREEDY RULE SELECTION ON CALIBRATION ===")
    print(f"Base calibration Macro F1: {best_f1:.6f}")

    remaining_pairs = list(base_pairs)
    selection_rows = []

    for step in range(1, max_rules + 1):
        candidates = make_candidate_rules(
            base_pairs=remaining_pairs,
            top_k_values=top_k_values,
            margin_values=margin_values,
            max_pred_prob_values=max_pred_prob_values,
            min_alt_prob_values=min_alt_prob_values,
        )

        best_candidate = None

        for candidate in candidates:
            trial_rules = selected_rules + [candidate]
            y_pred, _, _ = apply_rules(probs_cal, trial_rules)
            metrics = compute_metrics(y_cal, y_pred, num_classes)
            f1 = metrics["macro_f1_present_classes"]

            if best_candidate is None or f1 > best_candidate["f1"]:
                best_candidate = {
                    "rule": candidate,
                    "metrics": metrics,
                    "f1": f1,
                }

        if best_candidate is None:
            print("No candidate found.")
            break

        improvement = best_candidate["f1"] - best_f1

        if improvement < min_improvement:
            print(
                f"Stop at step {step}: best improvement {improvement:.8f} "
                f"< min_improvement {min_improvement}"
            )
            break

        selected_rule = best_candidate["rule"]
        selected_rules.append(selected_rule)
        best_metrics = best_candidate["metrics"]
        best_f1 = best_candidate["f1"]

        # Remove same directed pair from future selection to avoid duplicate rule.
        remaining_pairs = [
            p for p in remaining_pairs
            if not (
                p["true_mapped"] == selected_rule["true_mapped"]
                and p["pred_mapped"] == selected_rule["pred_mapped"]
            )
        ]

        row = {
            "step": step,
            "true_original": selected_rule["true_original"],
            "pred_original": selected_rule["pred_original"],
            "true_mapped": selected_rule["true_mapped"],
            "pred_mapped": selected_rule["pred_mapped"],
            "top_k": selected_rule["top_k"],
            "margin": selected_rule["margin"],
            "max_pred_prob": selected_rule["max_pred_prob"],
            "min_alt_prob": selected_rule["min_alt_prob"],
            "calibration_macro_f1": best_f1,
            "improvement": improvement,
            **best_metrics,
        }

        selection_rows.append(row)

        print(
            f"Step {step}: add rule {selected_rule['true_original']}>{selected_rule['pred_original']} "
            f"k={selected_rule['top_k']} margin={selected_rule['margin']} "
            f"maxp={selected_rule['max_pred_prob']} min_alt={selected_rule['min_alt_prob']} | "
            f"Cal Macro F1={best_f1:.6f} | improvement={improvement:.6f}"
        )

        if len(remaining_pairs) == 0:
            print("No remaining pairs.")
            break

    return selected_rules, best_metrics, pd.DataFrame(selection_rows)


def save_outputs(
    output_dir,
    split_name,
    df,
    y_true,
    y_pred,
    confidence,
    applied_rule,
    idx_to_label,
    num_classes,
    config,
    metrics,
):
    output_dir = Path(output_dir)
    labels_all = list(range(num_classes))
    target_names = [str(idx_to_label[i]) for i in labels_all]

    metrics_path = output_dir / f"{split_name}_metrics.json"
    predictions_path = output_dir / f"{split_name}_predictions.csv"
    report_path = output_dir / f"{split_name}_classification_report.csv"
    per_class_path = output_dir / f"{split_name}_per_class_metrics.csv"
    cm_path = output_dir / f"{split_name}_confusion_matrix.csv"

    out_metrics = {
        **config,
        **metrics,
        "num_rows": int(len(df)),
        "num_labels_present": int(df["mapped_label"].nunique()),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)

    pred_df = df.copy()
    pred_df["true_mapped_label"] = y_true
    pred_df["pred_mapped_label"] = y_pred
    pred_df["confidence"] = confidence
    pred_df["true_original_label"] = [idx_to_label[int(x)] for x in y_true]
    pred_df["pred_original_label"] = [idx_to_label[int(x)] for x in y_pred]
    pred_df["is_correct"] = pred_df["true_mapped_label"] == pred_df["pred_mapped_label"]
    pred_df["applied_m15_rule"] = applied_rule

    pred_df.to_csv(predictions_path, index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).T.to_csv(report_path)

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

    per_class_df.to_csv(per_class_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)

    print(f"\nSaved {split_name} outputs:")
    print(metrics_path)
    print(predictions_path)
    print(report_path)
    print(per_class_path)
    print(cm_path)


def main(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    print("Using device:", device)
    print("M10 checkpoint:", args.checkpoint_m10)
    print("M11 checkpoint:", args.checkpoint_m11)
    print("Calibration CSV:", args.calibration_csv)
    print("Test CSV:", args.test_csv)
    print("Output dir:", args.output_dir)

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

    weight_m11 = float(args.weight_m11)
    probs_cal = ensemble_probs(probs10_cal, probs11_cal, weight_m11)
    probs_test = ensemble_probs(probs10_test, probs11_test, weight_m11)

    base_pairs = parse_directed_pairs(args.directed_pairs, label_to_idx)

    print("\n=== Candidate directed pairs ===")
    for p in base_pairs:
        print(
            f"{p['true_original']}>{p['pred_original']} "
            f"(mapped {p['true_mapped']}>{p['pred_mapped']})"
        )

    top_k_values = parse_int_list(args.top_k_values)
    margin_values = parse_float_list(args.margin_values)
    max_pred_prob_values = parse_float_list(args.max_pred_prob_values)
    min_alt_prob_values = parse_float_list(args.min_alt_prob_values)

    # Base metrics without correction.
    base_cal_pred = probs_cal.argmax(axis=1)
    base_test_pred = probs_test.argmax(axis=1)

    base_cal_metrics = compute_metrics(y_cal, base_cal_pred, num_classes)
    base_test_metrics = compute_metrics(y_test, base_test_pred, num_classes)

    print("\n=== BASE ENSEMBLE BEFORE M15 ===")
    print(f"Calibration Macro F1: {base_cal_metrics['macro_f1_present_classes']:.6f}")
    print(f"Test Macro F1       : {base_test_metrics['macro_f1_present_classes']:.6f}")
    print(f"Test Accuracy       : {base_test_metrics['accuracy']:.6f}")

    selected_rules, selected_cal_metrics, selected_df = greedy_select_rules(
        probs_cal=probs_cal,
        y_cal=y_cal,
        base_pairs=base_pairs,
        num_classes=num_classes,
        top_k_values=top_k_values,
        margin_values=margin_values,
        max_pred_prob_values=max_pred_prob_values,
        min_alt_prob_values=min_alt_prob_values,
        max_rules=args.max_rules,
        min_improvement=args.min_improvement,
    )

    selected_rules_path = Path(args.output_dir) / "m15_selected_rules.json"
    with open(selected_rules_path, "w", encoding="utf-8") as f:
        json.dump(selected_rules, f, ensure_ascii=False, indent=2)

    selected_df_path = Path(args.output_dir) / "m15_rule_selection_history.csv"
    selected_df.to_csv(selected_df_path, index=False)

    print("\nSelected rules:")
    print(json.dumps(selected_rules, indent=2, ensure_ascii=False))

    y_cal_m15, conf_cal_m15, applied_cal = apply_rules(probs_cal, selected_rules)
    y_test_m15, conf_test_m15, applied_test = apply_rules(probs_test, selected_rules)

    cal_metrics = compute_metrics(y_cal, y_cal_m15, num_classes)
    test_metrics = compute_metrics(y_test, y_test_m15, num_classes)

    config = {
        "checkpoint_m10": args.checkpoint_m10,
        "checkpoint_m11": args.checkpoint_m11,
        "calibration_csv": args.calibration_csv,
        "test_csv": args.test_csv,
        "weight_m10": float(1.0 - weight_m11),
        "weight_m11": weight_m11,
        "selected_rules": selected_rules,
        "base_calibration_metrics": base_cal_metrics,
        "base_test_metrics": base_test_metrics,
    }

    print("\n=== M15 CALIBRATION RESULTS ===")
    print(f"Calibration Accuracy                 : {cal_metrics['accuracy']:.6f}")
    print(f"Calibration Macro F1 Present Classes : {cal_metrics['macro_f1_present_classes']:.6f}")

    print("\n=== M15 TEST RESULTS ===")
    print(f"Test Accuracy                 : {test_metrics['accuracy']:.6f}")
    print(f"Test Macro Precision Present  : {test_metrics['macro_precision_present_classes']:.6f}")
    print(f"Test Macro Recall Present     : {test_metrics['macro_recall_present_classes']:.6f}")
    print(f"Test Macro F1 Present Classes : {test_metrics['macro_f1_present_classes']:.6f}")
    print(f"Test Macro Precision All      : {test_metrics['macro_precision_all_classes']:.6f}")
    print(f"Test Macro Recall All         : {test_metrics['macro_recall_all_classes']:.6f}")
    print(f"Test Macro F1 All Classes     : {test_metrics['macro_f1_all_classes']:.6f}")

    save_outputs(
        output_dir=args.output_dir,
        split_name="m15_calibration",
        df=cal_df,
        y_true=y_cal,
        y_pred=y_cal_m15,
        confidence=conf_cal_m15,
        applied_rule=applied_cal,
        idx_to_label=idx_to_label,
        num_classes=num_classes,
        config=config,
        metrics=cal_metrics,
    )

    save_outputs(
        output_dir=args.output_dir,
        split_name="m15_test",
        df=test_df,
        y_true=y_test,
        y_pred=y_test_m15,
        confidence=conf_test_m15,
        applied_rule=applied_test,
        idx_to_label=idx_to_label,
        num_classes=num_classes,
        config=config,
        metrics=test_metrics,
    )

    print("\nSaved rule file:", selected_rules_path)
    print("Saved rule selection history:", selected_df_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M15 pairwise top-k correction for M10+M11 ensemble."
    )

    parser.add_argument("--checkpoint_m10", type=str, required=True)
    parser.add_argument("--checkpoint_m11", type=str, required=True)

    parser.add_argument("--calibration_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M15_pairwise_topk_correction/test_eval",
    )

    parser.add_argument("--weight_m11", type=float, default=0.46)

    parser.add_argument(
        "--directed_pairs",
        type=str,
        default="24>89,7>46,19>104,36>51,60>1,4>73,3>89,44>8,105>34,23>107",
    )

    parser.add_argument("--top_k_values", type=str, default="2,3,5")
    parser.add_argument("--margin_values", type=str, default="0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20")
    parser.add_argument("--max_pred_prob_values", type=str, default="0.40,0.50,0.60,0.70,0.80,1.00")
    parser.add_argument("--min_alt_prob_values", type=str, default="0.00,0.02,0.05,0.10")

    parser.add_argument("--max_rules", type=int, default=5)
    parser.add_argument("--min_improvement", type=float, default=0.0001)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_context_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

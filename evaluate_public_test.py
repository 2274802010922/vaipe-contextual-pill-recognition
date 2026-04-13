import os
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def stem_no_ext(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def normalize_map_key(name: str) -> List[str]:
    name = str(name)
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]

    keys = [name, base, stem]

    if "_det_" in stem:
        original_stem = stem.split("_det_")[0]
        keys.extend(
            [
                original_stem,
                original_stem + ".jpg",
                original_stem + ".png",
                original_stem + ".json",
            ]
        )

    return dedupe_keep_order(keys)


def guess_pill_label_dirs(test_root: str) -> List[str]:
    candidates = [
        os.path.join(test_root, "pill", "label"),
        os.path.join(test_root, "pill", "labels"),
        os.path.join(test_root, "pill", "json"),
        os.path.join(test_root, "pill", "annotation"),
        os.path.join(test_root, "pill", "annotations"),
        os.path.join(test_root, "label"),
        os.path.join(test_root, "labels"),
        os.path.join(test_root, "json"),
        os.path.join(test_root, "annotation"),
        os.path.join(test_root, "annotations"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


def build_pill_json_index(test_root: str) -> Dict[str, str]:
    label_dirs = guess_pill_label_dirs(test_root)
    index: Dict[str, str] = {}

    for root_dir in label_dirs:
        for fn in os.listdir(root_dir):
            full_path = os.path.join(root_dir, fn)
            if not os.path.isfile(full_path):
                continue
            for key in normalize_map_key(fn):
                index[key] = full_path

    return index


def build_pill_to_prescription_map(test_root: str) -> Dict[str, str]:
    candidates = [
        os.path.join(test_root, "pill_pres_map.json"),
        os.path.join(os.path.dirname(test_root), "pill_pres_map.json"),
    ]
    map_path = find_first_existing(candidates)
    if map_path is None:
        raise FileNotFoundError("Không tìm thấy pill_pres_map.json")

    data = load_json(map_path)
    out: Dict[str, str] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            if v is None:
                continue
            for kk in normalize_map_key(str(k)):
                out[kk] = str(v)

    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            # Format chuẩn VAIPE:
            # {"pres": "VAIPE_P_TEST_0.json", "pill": ["VAIPE_P_0_0.json", ...]}
            if "pres" in item and "pill" in item:
                pres_name = item.get("pres")
                pill_list = item.get("pill")

                if pres_name and isinstance(pill_list, list):
                    for pill_name in pill_list:
                        for kk in normalize_map_key(str(pill_name)):
                            out[kk] = str(pres_name)
                    continue

                if pres_name and pill_list and not isinstance(pill_list, list):
                    for kk in normalize_map_key(str(pill_list)):
                        out[kk] = str(pres_name)
                    continue

            pill_name = (
                item.get("pill_name")
                or item.get("pill_image")
                or item.get("pill")
                or item.get("image_name")
            )
            pres_name = (
                item.get("prescription_name")
                or item.get("prescription_image")
                or item.get("prescription")
                or item.get("pres_name")
                or item.get("pres")
            )

            if pres_name and isinstance(pill_name, list):
                for p in pill_name:
                    for kk in normalize_map_key(str(p)):
                        out[kk] = str(pres_name)
            elif pill_name and pres_name:
                for kk in normalize_map_key(str(pill_name)):
                    out[kk] = str(pres_name)
    else:
        raise RuntimeError("pill_pres_map.json có format lạ")

    return out


def extract_row_identifier_candidates(row: pd.Series) -> List[str]:
    candidates: List[str] = []
    for col in ["image_name", "crop_name", "image_path", "crop_path"]:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                candidates.append(value)
                candidates.append(os.path.basename(value))
    return dedupe_keep_order(candidates)


def build_group_key_for_row(row: pd.Series, pill_to_pres: Dict[str, str]) -> str:
    for candidate in extract_row_identifier_candidates(row):
        for key in normalize_map_key(candidate):
            pres_name = pill_to_pres.get(key)
            if pres_name:
                return stem_no_ext(pres_name)

    if "image_name" in row and pd.notna(row["image_name"]):
        return stem_no_ext(str(row["image_name"]))

    if "crop_name" in row and pd.notna(row["crop_name"]):
        return stem_no_ext(str(row["crop_name"]))

    return "unknown_group"


def safe_int(value):
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def parse_gt_instances_from_json(json_path: str) -> List[Dict]:
    data = load_json(json_path)
    instances: List[Dict] = []

    def add_instance(obj, fallback_name: str):
        if not isinstance(obj, dict):
            return

        label = obj.get("label", obj.get("class_id", obj.get("category_id", obj.get("pill_label"))))
        label = safe_int(label)

        # common box formats
        if all(k in obj for k in ["x", "y", "w", "h"]):
            x = float(obj["x"])
            y = float(obj["y"])
            w = float(obj["w"])
            h = float(obj["h"])
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif all(k in obj for k in ["x1", "y1", "x2", "y2"]):
            x1 = float(obj["x1"])
            y1 = float(obj["y1"])
            x2 = float(obj["x2"])
            y2 = float(obj["y2"])
        elif "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) == 4:
            x, y, w, h = obj["bbox"]
            x1, y1, x2, y2 = float(x), float(y), float(x) + float(w), float(y) + float(h)
        else:
            return

        if label is None:
            return

        if x2 <= x1 or y2 <= y1:
            return

        instances.append(
            {
                "gt_json_name": fallback_name,
                "gt_pill_stem": stem_no_ext(fallback_name),
                "gt_label": int(label),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    if isinstance(data, list):
        for obj in data:
            add_instance(obj, os.path.basename(json_path))
    elif isinstance(data, dict):
        # direct dict
        add_instance(data, os.path.basename(json_path))

        # nested common keys
        for key in ["annotations", "objects", "shapes", "boxes"]:
            if key in data and isinstance(data[key], list):
                for obj in data[key]:
                    add_instance(obj, os.path.basename(json_path))
    else:
        pass

    return instances


def build_gt_by_group(test_root: str) -> Dict[str, List[Dict]]:
    map_path = os.path.join(test_root, "pill_pres_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Không thấy pill_pres_map.json tại {map_path}")

    data = load_json(map_path)
    pill_json_index = build_pill_json_index(test_root)
    gt_by_group: Dict[str, List[Dict]] = {}

    if not isinstance(data, list):
        raise RuntimeError("Hiện script này kỳ vọng pill_pres_map.json của public_test là list format VAIPE.")

    for item in data:
        if not isinstance(item, dict):
            continue

        pres_name = item.get("pres")
        pill_list = item.get("pill", [])

        if not pres_name or not isinstance(pill_list, list):
            continue

        group_key = stem_no_ext(pres_name)
        gt_by_group.setdefault(group_key, [])

        for pill_json_name in pill_list:
            json_path = None
            for key in normalize_map_key(pill_json_name):
                if key in pill_json_index:
                    json_path = pill_json_index[key]
                    break

            if json_path is None:
                continue

            instances = parse_gt_instances_from_json(json_path)
            gt_by_group[group_key].extend(instances)

    return gt_by_group


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match_predictions_to_gt(
    pred_group: pd.DataFrame,
    gt_group: List[Dict],
    iou_thresh: float,
) -> Tuple[List[Dict], int, int, int]:
    """
    Returns:
      matched_rows, num_matched_det, num_unmatched_preds, num_unmatched_gts
    """
    matched_rows: List[Dict] = []
    used_gt = set()

    pred_group = pred_group.sort_values(by="detector_score", ascending=False).reset_index(drop=True)

    num_unmatched_preds = 0

    for _, pred in pred_group.iterrows():
        pred_box = (
            float(pred["x1"]),
            float(pred["y1"]),
            float(pred["x2"]),
            float(pred["y2"]),
        )

        best_gt_idx = None
        best_iou = -1.0

        for gt_idx, gt in enumerate(gt_group):
            if gt_idx in used_gt:
                continue
            gt_box = (gt["x1"], gt["y1"], gt["x2"], gt["y2"])
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is None:
            num_unmatched_preds += 1
            continue

        used_gt.add(best_gt_idx)
        gt = gt_group[best_gt_idx]

        matched_rows.append(
            {
                "group_key": pred["group_key"],
                "image_name": pred.get("image_name", ""),
                "crop_name": pred.get("crop_name", ""),
                "crop_path": pred.get("crop_path", ""),
                "detector_score": float(pred["detector_score"]),
                "x1": float(pred["x1"]),
                "y1": float(pred["y1"]),
                "x2": float(pred["x2"]),
                "y2": float(pred["y2"]),
                "pred_label": safe_int(pred["pred_label_final"]),
                "pred_idx_final": safe_int(pred["pred_idx_final"]),
                "conf_final": float(pred["conf_final"]),
                "gt_label": int(gt["gt_label"]),
                "gt_pill_stem": gt["gt_pill_stem"],
                "gt_json_name": gt["gt_json_name"],
                "iou": float(best_iou),
                "is_correct_label": int(safe_int(pred["pred_label_final"]) == int(gt["gt_label"])),
            }
        )

    num_matched_det = len(matched_rows)
    num_unmatched_gts = max(0, len(gt_group) - len(used_gt))
    return matched_rows, num_matched_det, num_unmatched_preds, num_unmatched_gts


def evaluate_one_model(
    model_key: str,
    pred_df: pd.DataFrame,
    gt_by_group: Dict[str, List[Dict]],
    iou_thresh: float,
) -> Tuple[Dict, pd.DataFrame]:
    matched_all: List[Dict] = []
    total_gt = 0
    total_preds = 0
    total_matched = 0
    total_unmatched_preds = 0
    total_unmatched_gts = 0

    group_keys = sorted(set(pred_df["group_key"].tolist()) | set(gt_by_group.keys()))

    for group_key in group_keys:
        pred_group = pred_df[pred_df["group_key"] == group_key].copy()
        gt_group = gt_by_group.get(group_key, [])

        total_gt += len(gt_group)
        total_preds += len(pred_group)

        if len(pred_group) == 0:
            total_unmatched_gts += len(gt_group)
            continue

        matched_rows, num_matched_det, num_unmatched_preds, num_unmatched_gts = greedy_match_predictions_to_gt(
            pred_group=pred_group,
            gt_group=gt_group,
            iou_thresh=iou_thresh,
        )

        matched_all.extend(matched_rows)
        total_matched += num_matched_det
        total_unmatched_preds += num_unmatched_preds
        total_unmatched_gts += num_unmatched_gts

    matched_df = pd.DataFrame(matched_all)

    if len(matched_df) == 0:
        summary = {
            "model_key": model_key,
            "num_gt": int(total_gt),
            "num_predictions": int(total_preds),
            "num_matched": 0,
            "num_unmatched_predictions": int(total_unmatched_preds),
            "num_unmatched_gt": int(total_unmatched_gts),
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "matched_accuracy": 0.0,
            "matched_macro_precision": 0.0,
            "matched_macro_recall": 0.0,
            "matched_macro_f1": 0.0,
            "strict_correct_over_gt": 0.0,
        }
        return summary, matched_df

    y_true = matched_df["gt_label"].astype(int).tolist()
    y_pred = matched_df["pred_label"].astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    num_correct = int((matched_df["is_correct_label"] == 1).sum())

    summary = {
        "model_key": model_key,
        "num_gt": int(total_gt),
        "num_predictions": int(total_preds),
        "num_matched": int(total_matched),
        "num_unmatched_predictions": int(total_unmatched_preds),
        "num_unmatched_gt": int(total_unmatched_gts),
        "detection_precision": round(float(total_matched / total_preds), 6) if total_preds > 0 else 0.0,
        "detection_recall": round(float(total_matched / total_gt), 6) if total_gt > 0 else 0.0,
        "matched_accuracy": round(float(acc), 6),
        "matched_macro_precision": round(float(macro_p), 6),
        "matched_macro_recall": round(float(macro_r), 6),
        "matched_macro_f1": round(float(macro_f1), 6),
        "strict_correct_over_gt": round(float(num_correct / total_gt), 6) if total_gt > 0 else 0.0,
    }
    return summary, matched_df


def main(args):
    if not os.path.exists(args.predictions_csv):
        raise FileNotFoundError(f"Không thấy predictions_csv: {args.predictions_csv}")
    if not os.path.exists(args.detections_csv):
        raise FileNotFoundError(f"Không thấy detections_csv: {args.detections_csv}")

    print("Loading predictions...")
    pred_df = pd.read_csv(args.predictions_csv)
    print("Pred rows:", len(pred_df))

    print("Loading detections...")
    det_df = pd.read_csv(args.detections_csv)
    print("Detection rows:", len(det_df))

    required_det_cols = {"crop_name", "x1", "y1", "x2", "y2", "score"}
    missing_det = required_det_cols - set(det_df.columns)
    if missing_det:
        raise RuntimeError(f"detections.csv thiếu cột: {sorted(missing_det)}")

    required_pred_cols = {"model_key", "crop_name", "pred_idx_final", "pred_label_final", "conf_final"}
    missing_pred = required_pred_cols - set(pred_df.columns)
    if missing_pred:
        raise RuntimeError(f"predictions.csv thiếu cột: {sorted(missing_pred)}")

    merge_cols = ["crop_name", "image_name", "image_path", "crop_path", "score", "x1", "y1", "x2", "y2", "width", "height"]
    merge_cols = [c for c in merge_cols if c in det_df.columns]

    eval_df = pred_df.merge(
        det_df[merge_cols].drop_duplicates(subset=["crop_name"]),
        on="crop_name",
        how="left",
    )

    eval_df = eval_df.rename(columns={"score": "detector_score"})

    print("Building pill->prescription map...")
    pill_to_pres = build_pill_to_prescription_map(args.test_root)

    print("Building ground truth from public_test...")
    gt_by_group = build_gt_by_group(args.test_root)
    print("Num GT groups:", len(gt_by_group))
    print("Total GT pills:", sum(len(v) for v in gt_by_group.values()))

    eval_df["group_key"] = eval_df.apply(
        lambda row: build_group_key_for_row(row, pill_to_pres),
        axis=1,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    summaries = []
    all_matched_parts = []

    model_keys = sorted(eval_df["model_key"].dropna().unique().tolist())
    print("Models:", model_keys)

    for model_key in model_keys:
        model_pred_df = eval_df[eval_df["model_key"] == model_key].copy()

        summary, matched_df = evaluate_one_model(
            model_key=model_key,
            pred_df=model_pred_df,
            gt_by_group=gt_by_group,
            iou_thresh=args.iou_thresh,
        )

        summaries.append(summary)

        if len(matched_df) > 0:
            matched_df["model_key"] = model_key
            all_matched_parts.append(matched_df)

        print(
            f"[{model_key}] "
            f"matched_macro_f1={summary['matched_macro_f1']:.4f} | "
            f"matched_accuracy={summary['matched_accuracy']:.4f} | "
            f"detection_recall={summary['detection_recall']:.4f} | "
            f"strict_correct_over_gt={summary['strict_correct_over_gt']:.4f}"
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["matched_macro_f1", "matched_accuracy", "strict_correct_over_gt"],
        ascending=False,
    )
    summary_path = os.path.join(args.output_dir, "public_test_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nSaved summary to:", summary_path)

    if len(all_matched_parts) > 0:
        matched_all_df = pd.concat(all_matched_parts, ignore_index=True)
    else:
        matched_all_df = pd.DataFrame()

    matched_path = os.path.join(args.output_dir, "public_test_matched_pairs.csv")
    matched_all_df.to_csv(matched_path, index=False, encoding="utf-8")
    print("Saved matched pairs to:", matched_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate public_test using ground truth + detections + M1-M9 predictions"
    )
    parser.add_argument(
        "--test_root",
        type=str,
        required=True,
        help="Path tới public_test",
    )
    parser.add_argument(
        "--detections_csv",
        type=str,
        required=True,
        help="detections.csv từ detect_public_test_faster_rcnn.py",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="m1_m9_predictions.csv từ run_m1_m9_on_detected_crops.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_public_test_eval",
        help="Thư mục lưu file tổng hợp đánh giá",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="Ngưỡng IoU để match detection với ground truth",
    )
    args = parser.parse_args()
    main(args)

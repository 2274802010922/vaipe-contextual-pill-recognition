import os
import json
import math
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm

from model_registry_m1_m9 import build_model_registry

from train_pika_v1_1_improved import DualEncoderPIKA
from train_pika_v2_context_labels import PIKAV2Model
from train_pika_v3_triple_context import TripleContextPIKA
from train_pika_graph import PIKAGraphModel
from train_best_pika_model import BestPIKAModel


IMAGE_SIZE = 224
PILL_MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
PRES_MODEL_NAME = "resnet18.a1_in1k"
GRAPH_HIDDEN_DIM = 256
MAX_CONTEXT_LEN = 8


def get_val_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_state_dict_flexible(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith("classifier.weight") and value.ndim == 2:
            return int(value.shape[0])
    raise RuntimeError("Không suy ra được num_classes từ state_dict (không thấy classifier.weight).")


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = adj.sum(axis=1)
    degree = np.where(degree == 0, 1.0, degree)
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


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


def guess_artifact_dirs(model_key: str, artifact_root: Optional[str]) -> List[str]:
    dirs = []

    if artifact_root:
        primary_mapping = {
            "M1": "M1_baseline",
            "M2": "M2_pika_like_v1",
            "M3": "M3_pika_v2",
            "M4": "M4_pika_v3",
            "M5": "M5_pika_graph_v1",
            "M6": "M6_best_pika_pre_ft",
            "M7": "M7_finetune_v1",
            "M8": "M8_finetune_v2",
            "M9": "M9_finetune_v3",
        }
        alias_mapping = {
            "M5": ["M5_pika_graph", "processed_pika_graph", "M6_best_pika_pre_ft"],
            "M6": ["processed_pika_best"],
            "M7": ["M6_best_pika_pre_ft", "processed_pika_best"],
            "M8": ["M6_best_pika_pre_ft", "processed_pika_best"],
            "M9": ["M6_best_pika_pre_ft", "processed_pika_best"],
        }

        dirs.append(os.path.join(artifact_root, primary_mapping[model_key]))
        for alias in alias_mapping.get(model_key, []):
            dirs.append(os.path.join(artifact_root, alias))

    fallback = {
        "M1": [],
        "M2": ["/content/processed_pika"],
        "M3": ["/content/processed_pika_v2"],
        "M4": ["/content/processed_pika_v3"],
        "M5": ["/content/processed_pika_graph", "/content/processed_pika_best"],
        "M6": ["/content/processed_pika_best"],
        "M7": ["/content/processed_pika_best"],
        "M8": ["/content/processed_pika_best"],
        "M9": ["/content/processed_pika_best"],
    }
    dirs.extend(fallback[model_key])

    return dedupe_keep_order(dirs)


def build_idx_to_label_map_from_csv(csv_path: str) -> Optional[Dict[int, str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {"mapped_label", "pill_label"}
    if not required.issubset(df.columns):
        return None

    pairs = df[["mapped_label", "pill_label"]].drop_duplicates()
    idx_to_label = {}
    for _, row in pairs.iterrows():
        idx_to_label[int(row["mapped_label"])] = str(int(row["pill_label"]))
    return idx_to_label


def load_idx_to_label_map(model_key: str, artifact_root: Optional[str], num_classes: int) -> Dict[int, str]:
    candidate_dirs = guess_artifact_dirs(model_key, artifact_root)

    # 1) ưu tiên idx_to_label.json
    for d in candidate_dirs:
        p = os.path.join(d, "idx_to_label.json")
        if os.path.exists(p):
            data = load_json(p)
            if isinstance(data, dict):
                return {int(k): str(v) for k, v in data.items()}
            if isinstance(data, list):
                return {i: str(v) for i, v in enumerate(data)}

    # 2) fallback từ metadata csv
    csv_names = [
        "pika_metadata.csv",
        "pika_context_metadata.csv",
        "pika_v3_metadata.csv",
        "pika_graph_metadata.csv",
        "best_pika_metadata.csv",
    ]
    for d in candidate_dirs:
        for name in csv_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                mapping = build_idx_to_label_map_from_csv(p)
                if mapping is not None and len(mapping) > 0:
                    return mapping

    # 3) fallback identity
    warnings.warn(
        f"[{model_key}] Không tìm thấy idx_to_label mapping. "
        f"Sẽ dùng mapped class index làm nhãn tạm thời."
    )
    return {i: str(i) for i in range(num_classes)}


def load_graph_artifact(model_key: str, artifact_root: Optional[str], num_classes: int, device: torch.device) -> torch.Tensor:
    candidate_dirs = guess_artifact_dirs(model_key, artifact_root)

    graph_json_candidates = []
    graph_npy_candidates = []

    for d in candidate_dirs:
        graph_json_candidates.extend([
            os.path.join(d, "graph_labels.json"),
        ])
        graph_npy_candidates.extend([
            os.path.join(d, "graph_pmi.npy"),
            os.path.join(d, "graph_cooccur.npy"),
        ])

    graph_json_path = find_first_existing(graph_json_candidates)
    graph_npy_path = find_first_existing(graph_npy_candidates)

    if graph_npy_path is None:
        raise FileNotFoundError(
            f"[{model_key}] Không tìm thấy graph artifact (.npy). "
            f"Hãy kiểm tra processed_pika_graph hoặc processed_pika_best."
        )

    graph_matrix = np.load(graph_npy_path)

    # Trường hợp đơn giản nhất: graph đã đúng kích thước num_classes
    if graph_matrix.shape[0] == num_classes and graph_matrix.shape[1] == num_classes:
        sub_graph = graph_matrix
    else:
        # Nếu có graph_labels.json và idx_to_label map thì cố subselect
        idx_to_label = load_idx_to_label_map(model_key, artifact_root, num_classes)
        original_labels = [int(idx_to_label[i]) for i in range(num_classes)]

        if graph_json_path is not None:
            graph_labels = load_json(graph_json_path)
            graph_label_to_idx = {int(lbl): idx for idx, lbl in enumerate(graph_labels)}

            try:
                indices = [graph_label_to_idx[int(lbl)] for lbl in original_labels]
                sub_graph = graph_matrix[np.ix_(indices, indices)]
            except Exception:
                warnings.warn(
                    f"[{model_key}] Không subselect graph chính xác được. "
                    f"Sẽ cắt top-left {num_classes}x{num_classes}."
                )
                sub_graph = graph_matrix[:num_classes, :num_classes]
        else:
            warnings.warn(
                f"[{model_key}] Không có graph_labels.json. "
                f"Sẽ cắt top-left {num_classes}x{num_classes}."
            )
            sub_graph = graph_matrix[:num_classes, :num_classes]

    sub_graph = normalize_adjacency(sub_graph)
    return torch.tensor(sub_graph, dtype=torch.float32, device=device)


def build_model_from_spec(model_key: str, checkpoint_path: str, artifact_root: Optional[str], device: torch.device):
    state_dict = load_state_dict_flexible(checkpoint_path, device)
    num_classes = infer_num_classes_from_state_dict(state_dict)

    if model_key == "M1":
        model = timm.create_model(
            PILL_MODEL_NAME,
            pretrained=False,
            num_classes=num_classes,
        )

    elif model_key == "M2":
        model = DualEncoderPIKA(
            num_classes=num_classes,
            pill_model_name=PILL_MODEL_NAME,
            pres_model_name=PRES_MODEL_NAME,
        )

    elif model_key == "M3":
        model = PIKAV2Model(
            num_classes=num_classes,
            pill_model_name=PILL_MODEL_NAME,
        )

    elif model_key == "M4":
        model = TripleContextPIKA(
            num_classes=num_classes,
            pill_model_name=PILL_MODEL_NAME,
            pres_model_name=PRES_MODEL_NAME,
        )

    elif model_key == "M5":
        adj = load_graph_artifact(model_key, artifact_root, num_classes, device)
        model = PIKAGraphModel(
            num_classes=num_classes,
            adj_matrix=adj,
            pill_model_name=PILL_MODEL_NAME,
            hidden_dim=GRAPH_HIDDEN_DIM,
        )

    elif model_key in {"M6", "M7", "M8", "M9"}:
        adj = load_graph_artifact(model_key, artifact_root, num_classes, device)
        model = BestPIKAModel(
            num_classes=num_classes,
            adj_matrix=adj,
            pill_model_name=PILL_MODEL_NAME,
            pres_model_name=PRES_MODEL_NAME,
            hidden_dim=GRAPH_HIDDEN_DIM,
        )

    else:
        raise ValueError(f"Model key không hợp lệ: {model_key}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    idx_to_label = load_idx_to_label_map(model_key, artifact_root, num_classes)
    return model, num_classes, idx_to_label


def normalize_map_key(name: str) -> List[str]:
    name = str(name)
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    keys = [name, base, stem]

    if "_det_" in stem:
        original_stem = stem.split("_det_")[0]
        keys.append(original_stem)
        keys.append(original_stem + ".jpg")
        keys.append(original_stem + ".png")

    return dedupe_keep_order(keys)


def extract_row_identifier_candidates(row: pd.Series) -> List[str]:
    candidates = []

    for col in ["image_name", "crop_name", "image_path", "crop_path"]:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                candidates.append(value)
                candidates.append(os.path.basename(value))

    return dedupe_keep_order(candidates)


def lookup_prescription_name_for_row(row: pd.Series, pill_to_pres: Dict[str, str]) -> Optional[str]:
    for candidate in extract_row_identifier_candidates(row):
        for key in normalize_map_key(candidate):
            pres_name = pill_to_pres.get(key)
            if pres_name:
                return str(pres_name)
    return None


def build_prescription_file_index(test_root: str) -> Dict[str, str]:
    candidate_dirs = [
        os.path.join(test_root, "prescription", "image"),
        os.path.join(test_root, "prescription"),
        os.path.join(test_root, "pres", "image"),
        os.path.join(test_root, "pres"),
    ]

    index = {}
    for root_dir in candidate_dirs:
        if not os.path.isdir(root_dir):
            continue
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
        warnings.warn("Không tìm thấy pill_pres_map.json. Các model cần prescription có thể không chạy được.")
        return {}

    data = load_json(map_path)
    mapping = {}

    def add_entry(pill_name, pres_name):
        if pill_name is None or pres_name is None:
            return

        if isinstance(pill_name, (list, tuple, set)):
            for item in pill_name:
                add_entry(item, pres_name)
            return

        for key in normalize_map_key(str(pill_name)):
            mapping[key] = str(pres_name)

    if isinstance(data, dict):
        simple = True
        for k, v in data.items():
            if isinstance(v, str):
                add_entry(k, v)
            else:
                simple = False

        if not simple:
            for _, item in data.items():
                if not isinstance(item, dict):
                    continue
                pill_name = (
                    item.get("pill")
                    or item.get("pill_image")
                    or item.get("pill_image_name")
                    or item.get("pill_name")
                )
                pres_name = (
                    item.get("pres")
                    or item.get("prescription")
                    or item.get("prescription_image")
                    or item.get("prescription_image_name")
                    or item.get("pres_name")
                )
                add_entry(pill_name, pres_name)

    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pill_name = (
                item.get("pill")
                or item.get("pill_image")
                or item.get("pill_image_name")
                or item.get("pill_name")
            )
            pres_name = (
                item.get("pres")
                or item.get("prescription")
                or item.get("prescription_image")
                or item.get("prescription_image_name")
                or item.get("pres_name")
            )
            add_entry(pill_name, pres_name)

    return mapping


def resolve_prescription_path(
    test_root: str,
    row: pd.Series,
    pill_to_pres: Dict[str, str],
    prescription_index: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    pres_name = lookup_prescription_name_for_row(row, pill_to_pres)
    if pres_name is None:
        return None

    if prescription_index is None:
        prescription_index = build_prescription_file_index(test_root)

    direct_candidates = [pres_name, os.path.basename(pres_name)]
    for candidate in direct_candidates:
        for key in normalize_map_key(candidate):
            found = prescription_index.get(key)
            if found and os.path.exists(found):
                return found

    return None


def load_image_tensor(image_path: str, tfm, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)
    return tensor


def build_context_vector(context_labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    vec = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
    for lbl in sorted(set(context_labels)):
        if 0 <= int(lbl) < num_classes:
            vec[0, int(lbl)] = 1.0
    return vec


def build_context_indices_and_mask(context_labels: List[int], max_context_len: int, device: torch.device):
    context_labels = sorted(set(int(x) for x in context_labels if int(x) >= 0))
    context_labels = context_labels[:max_context_len]

    padded = [-1] * max_context_len
    mask = [0] * max_context_len

    for i, val in enumerate(context_labels):
        padded[i] = int(val)
        mask[i] = 1

    context_indices = torch.tensor([padded], dtype=torch.long, device=device)
    context_mask = torch.tensor([mask], dtype=torch.bool, device=device)
    return context_indices, context_mask


@torch.no_grad()
def predict_single(
    model_key: str,
    model,
    crop_path: str,
    prescription_path: Optional[str],
    context_labels: List[int],
    num_classes: int,
    device: torch.device,
    tfm,
):
    pill_tensor = load_image_tensor(crop_path, tfm, device)

    if model_key == "M1":
        logits = model(pill_tensor)

    elif model_key == "M2":
        if prescription_path is None:
            raise FileNotFoundError("[M2] Không tìm thấy prescription image cho crop này.")
        pres_tensor = load_image_tensor(prescription_path, tfm, device)
        logits = model(pill_tensor, pres_tensor)

    elif model_key == "M3":
        context_vector = build_context_vector(context_labels, num_classes, device)
        logits = model(pill_tensor, context_vector)

    elif model_key == "M4":
        if prescription_path is None:
            raise FileNotFoundError("[M4] Không tìm thấy prescription image cho crop này.")
        pres_tensor = load_image_tensor(prescription_path, tfm, device)
        context_vector = build_context_vector(context_labels, num_classes, device)
        logits = model(pill_tensor, pres_tensor, context_vector)

    elif model_key == "M5":
        context_indices, context_mask = build_context_indices_and_mask(context_labels, MAX_CONTEXT_LEN, device)
        logits = model(pill_tensor, context_indices, context_mask)

    elif model_key in {"M6", "M7", "M8", "M9"}:
        if prescription_path is None:
            raise FileNotFoundError(f"[{model_key}] Không tìm thấy prescription image cho crop này.")
        pres_tensor = load_image_tensor(prescription_path, tfm, device)
        context_indices, context_mask = build_context_indices_and_mask(context_labels, MAX_CONTEXT_LEN, device)
        logits = model(pill_tensor, pres_tensor, context_indices, context_mask)

    else:
        raise ValueError(f"Model key không hợp lệ: {model_key}")

    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)

    return int(pred_idx.item()), float(conf.item())


def run_model_two_pass(
    model_key: str,
    model,
    group_df: pd.DataFrame,
    test_root: str,
    pill_to_pres: Dict[str, str],
    prescription_index: Dict[str, str],
    num_classes: int,
    idx_to_label: Dict[int, str],
    device: torch.device,
    tfm,
) -> List[Dict]:
    rows = []

    first_pass_preds = []
    first_pass_confs = []
    prescription_paths = []

    for _, row in group_df.iterrows():
        crop_path = row["crop_path"]
        prescription_path = resolve_prescription_path(
            test_root=test_root,
            row=row,
            pill_to_pres=pill_to_pres,
            prescription_index=prescription_index,
        )
        prescription_paths.append(prescription_path)

        pred_idx, conf = predict_single(
            model_key=model_key,
            model=model,
            crop_path=crop_path,
            prescription_path=prescription_path,
            context_labels=[],
            num_classes=num_classes,
            device=device,
            tfm=tfm,
        )
        first_pass_preds.append(pred_idx)
        first_pass_confs.append(conf)

    for i, (_, row) in enumerate(group_df.iterrows()):
        image_name = row["image_name"]
        crop_name = row["crop_name"]
        crop_path = row["crop_path"]
        detector_score = float(row["score"])
        prescription_path = prescription_paths[i]

        if model_key in {"M3", "M4", "M5", "M6", "M7", "M8", "M9"}:
            context_labels_final = [first_pass_preds[j] for j in range(len(first_pass_preds)) if j != i]
            pred_idx_final, conf_final = predict_single(
                model_key=model_key,
                model=model,
                crop_path=crop_path,
                prescription_path=prescription_path,
                context_labels=context_labels_final,
                num_classes=num_classes,
                device=device,
                tfm=tfm,
            )
            used_context = True
        else:
            pred_idx_final = first_pass_preds[i]
            conf_final = first_pass_confs[i]
            used_context = False

        rows.append({
            "group_key": str(row.get("_group_key", image_name)),
            "image_name": image_name,
            "crop_name": crop_name,
            "crop_path": crop_path,
            "model_key": model_key,
            "detector_score": detector_score,
            "prescription_path": prescription_path if prescription_path is not None else "",
            "pred_idx_first": int(first_pass_preds[i]),
            "pred_label_first": idx_to_label.get(int(first_pass_preds[i]), str(int(first_pass_preds[i]))),
            "conf_first": float(first_pass_confs[i]),
            "pred_idx_final": int(pred_idx_final),
            "pred_label_final": idx_to_label.get(int(pred_idx_final), str(int(pred_idx_final))),
            "conf_final": float(conf_final),
            "used_context_in_final_pass": int(used_context),
        })

    return rows



def build_group_key(row: pd.Series, pill_to_pres: Dict[str, str]) -> str:
    pres_name = lookup_prescription_name_for_row(row, pill_to_pres)
    if pres_name:
        return os.path.splitext(os.path.basename(pres_name))[0]

    if "image_name" in row and pd.notna(row["image_name"]):
        return str(row["image_name"])

    if "crop_name" in row and pd.notna(row["crop_name"]):
        return str(row["crop_name"])

    return "unknown_group"



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    detections_df = pd.read_csv(args.detections_csv)
    if len(detections_df) == 0:
        raise RuntimeError("detections.csv đang rỗng.")

    tfm = get_val_transform(IMAGE_SIZE)

    registry = build_model_registry(
        model_dir=args.model_dir,
        artifact_root=args.artifact_root if args.artifact_root else "./model_artifacts",
    )

    if args.models.lower() == "all":
        model_keys = list(registry.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in model_keys:
            if m not in registry:
                raise ValueError(f"Model key không hợp lệ: {m}")

    pill_to_pres = build_pill_to_prescription_map(args.test_root)
    prescription_index = build_prescription_file_index(args.test_root)

    detections_df["_group_key"] = detections_df.apply(
        lambda row: build_group_key(row, pill_to_pres),
        axis=1,
    )

    grouped = list(detections_df.groupby("_group_key", sort=False))
    print("Num detected pill images:", len(grouped))
    print("Selected models:", model_keys)

    all_rows = []
    failed_models = []

    for model_key in model_keys:
        spec = registry[model_key]
        checkpoint_path = spec.checkpoint_path

        if not os.path.exists(checkpoint_path):
            msg = f"[{model_key}] Không tìm thấy checkpoint: {checkpoint_path}"
            if args.skip_missing_models:
                print(msg)
                failed_models.append((model_key, msg))
                continue
            raise FileNotFoundError(msg)

        try:
            model, num_classes, idx_to_label = build_model_from_spec(
                model_key=model_key,
                checkpoint_path=checkpoint_path,
                artifact_root=args.artifact_root,
                device=device,
            )
            print(f"[{model_key}] Loaded successfully | num_classes={num_classes}")
        except Exception as e:
            msg = f"[{model_key}] Load model/artifact failed: {e}"
            if args.skip_missing_models:
                print(msg)
                failed_models.append((model_key, msg))
                continue
            raise

        for group_idx, (group_key, group_df) in enumerate(grouped, start=1):
            try:
                rows = run_model_two_pass(
                    model_key=model_key,
                    model=model,
                    group_df=group_df.reset_index(drop=True),
                    test_root=args.test_root,
                    pill_to_pres=pill_to_pres,
                    prescription_index=prescription_index,
                    num_classes=num_classes,
                    idx_to_label=idx_to_label,
                    device=device,
                    tfm=tfm,
                )
                all_rows.extend(rows)
            except Exception as e:
                print(f"[{model_key}] Failed on group {group_key}: {e}")

            if group_idx % 20 == 0 or group_idx == len(grouped):
                print(f"[{model_key}] processed {group_idx}/{len(grouped)} images")

    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print("Saved predictions to:", args.output_csv)

    if args.failed_models_log:
        with open(args.failed_models_log, "w", encoding="utf-8") as f:
            for k, msg in failed_models:
                f.write(f"{k}	{msg}\n")
        print("Saved failed model log to:", args.failed_models_log)

    print("Done.")
    print("Total prediction rows:", len(out_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M1-M9 recognition on detector crops from public_test")

    parser.add_argument("--detections_csv", type=str, required=True, help="CSV từ detect_public_test_faster_rcnn.py")
    parser.add_argument("--test_root", type=str, required=True, help="Path tới public_test")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa M1..M9 .pth")
    parser.add_argument("--artifact_root", type=str, default="", help="Thư mục chứa artifact phụ trợ nếu có")
    parser.add_argument("--models", type=str, default="all", help='Ví dụ: "all" hoặc "M1,M2,M9"')
    parser.add_argument("--output_csv", type=str, default="outputs_m1_m9_predictions/m1_m9_predictions.csv")
    parser.add_argument("--failed_models_log", type=str, default="outputs_m1_m9_predictions/failed_models.txt")
    parser.add_argument("--skip_missing_models", action="store_true")

    args = parser.parse_args()
    main(args)

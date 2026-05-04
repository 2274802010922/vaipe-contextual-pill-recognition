import argparse
import ast
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_context_labels(value):
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, list):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "null"]:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            out = []
            for x in parsed:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return out

        if isinstance(parsed, (int, float)):
            return [int(parsed)]

    except Exception:
        pass

    # Fallback for strings like "1,2,3"
    out = []
    for part in text.replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass

    return out


def load_train_metadata(train_csv, label_col, group_col):
    train_csv = Path(train_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in train CSV.")

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in train CSV.")

    df[label_col] = df[label_col].astype(int)
    df[group_col] = df[group_col].astype(str)

    return df


def build_label_mapping(df, label_col):
    labels = sorted(df[label_col].astype(int).unique().tolist())

    label_to_idx = {int(label): int(i) for i, label in enumerate(labels)}
    idx_to_label = {int(i): int(label) for label, i in label_to_idx.items()}

    return labels, label_to_idx, idx_to_label


def build_prescription_label_sets(df, label_col, group_col, context_col=None, include_context_labels=False):
    group_to_labels = {}

    for group_id, g in df.groupby(group_col):
        labels = set(g[label_col].astype(int).tolist())

        if include_context_labels and context_col and context_col in g.columns:
            for value in g[context_col].tolist():
                for lb in parse_context_labels(value):
                    labels.add(int(lb))

        group_to_labels[str(group_id)] = sorted(labels)

    return group_to_labels


def build_cooccurrence_graph(group_to_labels, label_to_idx, num_classes):
    group_count = 0
    class_doc_count = np.zeros(num_classes, dtype=np.float64)
    cooccur_count = np.zeros((num_classes, num_classes), dtype=np.float64)

    group_size_counter = Counter()

    for group_id, labels in group_to_labels.items():
        mapped = sorted({label_to_idx[int(lb)] for lb in labels if int(lb) in label_to_idx})

        if len(mapped) == 0:
            continue

        group_count += 1
        group_size_counter[len(mapped)] += 1

        for i in mapped:
            class_doc_count[i] += 1.0

        # Self count records class appearance in prescription.
        for i in mapped:
            cooccur_count[i, i] += 1.0

        # Pairwise co-occurrence.
        for i, j in combinations(mapped, 2):
            cooccur_count[i, j] += 1.0
            cooccur_count[j, i] += 1.0

    return group_count, class_doc_count, cooccur_count, group_size_counter


def compute_ppmi(
    cooccur_count,
    class_doc_count,
    group_count,
    min_cooccur=1,
    eps=1e-12,
    self_loop_value=1.0,
):
    num_classes = cooccur_count.shape[0]

    p_i = class_doc_count / max(1.0, float(group_count))

    p_ij = cooccur_count / max(1.0, float(group_count))

    pmi = np.zeros_like(cooccur_count, dtype=np.float64)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue

            if cooccur_count[i, j] < min_cooccur:
                continue

            denom = p_i[i] * p_i[j]

            if denom <= 0:
                continue

            value = np.log((p_ij[i, j] + eps) / (denom + eps))
            pmi[i, j] = value

    ppmi = np.maximum(pmi, 0.0)

    if self_loop_value is not None:
        np.fill_diagonal(ppmi, float(self_loop_value))
        np.fill_diagonal(pmi, float(self_loop_value))

    return pmi.astype(np.float32), ppmi.astype(np.float32)


def spectral_embedding(matrix, embedding_dim=64, normalize=True):
    matrix = np.asarray(matrix, dtype=np.float64)

    # Symmetrize for numerical stability.
    matrix = 0.5 * (matrix + matrix.T)

    u, s, _ = np.linalg.svd(matrix, full_matrices=False)

    dim = min(int(embedding_dim), u.shape[1])

    emb = u[:, :dim] * np.sqrt(np.maximum(s[:dim], 0.0))

    if normalize:
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-12)

    return emb.astype(np.float32)


def build_edge_table(cooccur_count, pmi, ppmi, idx_to_label):
    rows = []
    num_classes = cooccur_count.shape[0]

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            c = float(cooccur_count[i, j])
            if c <= 0:
                continue

            rows.append({
                "src_idx": int(i),
                "dst_idx": int(j),
                "src_label": int(idx_to_label[i]),
                "dst_label": int(idx_to_label[j]),
                "cooccur_count": c,
                "pmi": float(pmi[i, j]),
                "ppmi": float(ppmi[i, j]),
            })

    edge_df = pd.DataFrame(rows)

    if len(edge_df) > 0:
        edge_df = edge_df.sort_values(
            ["ppmi", "cooccur_count"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return edge_df


def build_class_stats(labels, class_doc_count, label_to_idx, df, label_col):
    instance_counts = df[label_col].astype(int).value_counts().to_dict()

    rows = []

    for label in labels:
        idx = label_to_idx[int(label)]

        rows.append({
            "idx": int(idx),
            "label": int(label),
            "instance_count": int(instance_counts.get(int(label), 0)),
            "prescription_doc_count": int(class_doc_count[idx]),
        })

    return pd.DataFrame(rows)


def main(args):
    ensure_dir(args.output_dir)

    print("=== M17 PIKA GRAPH EMBEDDING BUILDER ===")
    print("Train CSV:", args.train_csv)
    print("Output dir:", args.output_dir)
    print("Label col:", args.label_col)
    print("Group col:", args.group_col)
    print("Context col:", args.context_col)
    print("Include context labels:", args.include_context_labels)
    print("Embedding dim:", args.embedding_dim)
    print("Min cooccur:", args.min_cooccur)
    print("Self loop value:", args.self_loop_value)

    df = load_train_metadata(
        train_csv=args.train_csv,
        label_col=args.label_col,
        group_col=args.group_col,
    )

    print("\nTrain rows:", len(df))
    print("Train groups:", df[args.group_col].nunique())
    print("Train labels:", df[args.label_col].nunique())

    labels, label_to_idx, idx_to_label = build_label_mapping(df, args.label_col)
    num_classes = len(labels)

    print("Num classes:", num_classes)
    print("Min label:", min(labels))
    print("Max label:", max(labels))

    group_to_labels = build_prescription_label_sets(
        df=df,
        label_col=args.label_col,
        group_col=args.group_col,
        context_col=args.context_col,
        include_context_labels=args.include_context_labels,
    )

    group_count, class_doc_count, cooccur_count, group_size_counter = build_cooccurrence_graph(
        group_to_labels=group_to_labels,
        label_to_idx=label_to_idx,
        num_classes=num_classes,
    )

    print("\nValid prescription groups:", group_count)
    print("Graph cooccur shape:", cooccur_count.shape)
    print("Non-zero cooccur entries:", int((cooccur_count > 0).sum()))
    print("Group size distribution:")
    for k, v in sorted(group_size_counter.items()):
        print(f"  {k} labels/group: {v} groups")

    pmi, ppmi = compute_ppmi(
        cooccur_count=cooccur_count,
        class_doc_count=class_doc_count,
        group_count=group_count,
        min_cooccur=args.min_cooccur,
        self_loop_value=args.self_loop_value,
    )

    embeddings = spectral_embedding(
        matrix=ppmi,
        embedding_dim=args.embedding_dim,
        normalize=not args.no_normalize_embeddings,
    )

    print("\nPMI shape:", pmi.shape)
    print("PPMI shape:", ppmi.shape)
    print("Embeddings shape:", embeddings.shape)

    out_dir = Path(args.output_dir)

    # Compatibility names.
    np.save(out_dir / "graph_cooccur.npy", cooccur_count.astype(np.float32))
    np.save(out_dir / "graph_pmi.npy", ppmi.astype(np.float32))
    np.save(out_dir / "graph_ppmi.npy", ppmi.astype(np.float32))
    np.save(out_dir / "graph_raw_pmi.npy", pmi.astype(np.float32))
    np.save(out_dir / "graph_embeddings.npy", embeddings.astype(np.float32))

    # JSON artifacts.
    save_json([int(x) for x in labels], out_dir / "graph_labels.json")
    save_json({str(k): int(v) for k, v in label_to_idx.items()}, out_dir / "label_to_idx.json")
    save_json({str(k): int(v) for k, v in idx_to_label.items()}, out_dir / "idx_to_label.json")

    class_stats_df = build_class_stats(
        labels=labels,
        class_doc_count=class_doc_count,
        label_to_idx=label_to_idx,
        df=df,
        label_col=args.label_col,
    )

    class_stats_df.to_csv(out_dir / "graph_class_stats.csv", index=False)

    edge_df = build_edge_table(
        cooccur_count=cooccur_count,
        pmi=pmi,
        ppmi=ppmi,
        idx_to_label=idx_to_label,
    )

    edge_df.to_csv(out_dir / "graph_edges.csv", index=False)

    group_rows = []
    for gid, lbs in group_to_labels.items():
        group_rows.append({
            "prescription_key": gid,
            "num_labels": len(lbs),
            "labels": json.dumps([int(x) for x in lbs]),
        })

    pd.DataFrame(group_rows).to_csv(out_dir / "prescription_label_sets.csv", index=False)

    config = {
        "stage": "M17_faithful_pika_graph_embeddings",
        "train_csv": args.train_csv,
        "output_dir": args.output_dir,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "context_col": args.context_col,
        "include_context_labels": args.include_context_labels,
        "num_train_rows": int(len(df)),
        "num_groups": int(group_count),
        "num_classes": int(num_classes),
        "embedding_dim": int(args.embedding_dim),
        "min_cooccur": int(args.min_cooccur),
        "self_loop_value": float(args.self_loop_value),
        "embedding_method": "PPMI_SVD",
        "important_note": "Graph was built from train split only to avoid validation/test leakage.",
    }

    save_json(config, out_dir / "m17_graph_config.json")

    with open(out_dir / "m17_graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== M17 PIKA GRAPH EMBEDDING SUMMARY ===\n")
        f.write(f"Train CSV: {args.train_csv}\n")
        f.write(f"Train rows: {len(df)}\n")
        f.write(f"Train groups: {df[args.group_col].nunique()}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Valid prescription groups: {group_count}\n")
        f.write(f"Graph cooccur shape: {cooccur_count.shape}\n")
        f.write(f"PMI shape: {pmi.shape}\n")
        f.write(f"PPMI shape: {ppmi.shape}\n")
        f.write(f"Embeddings shape: {embeddings.shape}\n")
        f.write("Graph was built from train split only.\n")

    print("\nSaved artifacts:")
    for p in sorted(out_dir.glob("*")):
        print(p)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build M17 paper-like PIKA graph embeddings from train prescriptions only."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M17_faithful_pika/graph_artifacts",
    )

    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--group_col", type=str, default="prescription_key")
    parser.add_argument("--context_col", type=str, default="context_labels")

    parser.add_argument(
        "--include_context_labels",
        action="store_true",
        help="Optional. Default OFF. Graph is normally built from labels observed in each prescription group.",
    )

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--min_cooccur", type=int, default=1)
    parser.add_argument("--self_loop_value", type=float, default=1.0)

    parser.add_argument(
        "--no_normalize_embeddings",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)

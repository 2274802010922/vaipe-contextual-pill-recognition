import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def spectral_embedding(matrix, embedding_dim=64, normalize=True):
    matrix = np.asarray(matrix, dtype=np.float64)

    # Symmetrize for stable SVD
    matrix = 0.5 * (matrix + matrix.T)

    u, s, _ = np.linalg.svd(matrix, full_matrices=False)

    dim = min(int(embedding_dim), u.shape[1])
    emb = u[:, :dim] * np.sqrt(np.maximum(s[:dim], 0.0))

    if normalize:
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-12)

    return emb.astype(np.float32)


def prune_ppmi(ppmi, prune_ratio=0.20, keep_diagonal=True):
    ppmi = np.asarray(ppmi, dtype=np.float32).copy()

    if ppmi.shape[0] != ppmi.shape[1]:
        raise ValueError("PPMI matrix must be square.")

    n = ppmi.shape[0]

    diag = np.diag(ppmi).copy()

    # Only prune positive off-diagonal edges.
    offdiag_mask = ~np.eye(n, dtype=bool)
    positive_mask = (ppmi > 0) & offdiag_mask

    values = ppmi[positive_mask]

    total_positive_edges = int(values.size)

    if total_positive_edges == 0:
        print("No positive off-diagonal edges found. Nothing to prune.")
        pruned = ppmi
        threshold = None
        removed_edges = 0
    else:
        prune_ratio = float(prune_ratio)
        prune_ratio = min(max(prune_ratio, 0.0), 1.0)

        if prune_ratio <= 0:
            threshold = None
            removed_edges = 0
            pruned = ppmi
        else:
            threshold = np.quantile(values, prune_ratio)

            remove_mask = positive_mask & (ppmi <= threshold)
            removed_edges = int(remove_mask.sum())

            pruned = ppmi.copy()
            pruned[remove_mask] = 0.0

    if keep_diagonal:
        np.fill_diagonal(pruned, diag)
    else:
        np.fill_diagonal(pruned, 0.0)

    # Symmetrize again after pruning
    pruned = np.maximum(pruned, pruned.T)

    return pruned.astype(np.float32), {
        "total_positive_offdiag_edges": int(total_positive_edges),
        "removed_edges": int(removed_edges),
        "remaining_positive_offdiag_edges": int(((pruned > 0) & offdiag_mask).sum()),
        "threshold": None if threshold is None else float(threshold),
        "prune_ratio": float(prune_ratio),
        "keep_diagonal": bool(keep_diagonal),
    }


def build_edge_table(matrix, labels):
    rows = []
    n = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            w = float(matrix[i, j])
            if w <= 0:
                continue

            rows.append({
                "src_idx": int(i),
                "dst_idx": int(j),
                "src_label": int(labels[i]),
                "dst_label": int(labels[j]),
                "weight": w,
            })

    edge_df = pd.DataFrame(rows)

    if len(edge_df) > 0:
        edge_df = edge_df.sort_values("weight", ascending=False).reset_index(drop=True)

    return edge_df


def main(args):
    ensure_dir(args.output_dir)

    base_dir = Path(args.base_graph_dir)
    output_dir = Path(args.output_dir)

    print("=== M20 PRUNED GRAPH EMBEDDING BUILDER ===")
    print("Base graph dir:", base_dir)
    print("Output dir:", output_dir)
    print("Prune ratio:", args.prune_ratio)
    print("Embedding dim:", args.embedding_dim)

    graph_labels_path = base_dir / "graph_labels.json"
    label_to_idx_path = base_dir / "label_to_idx.json"
    idx_to_label_path = base_dir / "idx_to_label.json"

    if (base_dir / "graph_ppmi.npy").exists():
        ppmi_path = base_dir / "graph_ppmi.npy"
    elif (base_dir / "graph_pmi.npy").exists():
        ppmi_path = base_dir / "graph_pmi.npy"
    else:
        raise FileNotFoundError("Cannot find graph_ppmi.npy or graph_pmi.npy in base graph dir.")

    for p in [graph_labels_path, label_to_idx_path, idx_to_label_path, ppmi_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    labels = [int(x) for x in load_json(graph_labels_path)]
    label_to_idx = load_json(label_to_idx_path)
    idx_to_label = load_json(idx_to_label_path)

    ppmi = np.load(ppmi_path).astype(np.float32)

    print("Loaded PPMI:", ppmi_path)
    print("PPMI shape:", ppmi.shape)
    print("Num labels:", len(labels))

    pruned_ppmi, stats = prune_ppmi(
        ppmi=ppmi,
        prune_ratio=args.prune_ratio,
        keep_diagonal=not args.no_keep_diagonal,
    )

    embeddings = spectral_embedding(
        matrix=pruned_ppmi,
        embedding_dim=args.embedding_dim,
        normalize=not args.no_normalize_embeddings,
    )

    print("\nPruning stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nPruned PPMI shape:", pruned_ppmi.shape)
    print("Embeddings shape:", embeddings.shape)

    # Save compatible artifacts for train_m19_arch_pika_v1.py
    save_json(labels, output_dir / "graph_labels.json")
    save_json(label_to_idx, output_dir / "label_to_idx.json")
    save_json(idx_to_label, output_dir / "idx_to_label.json")

    np.save(output_dir / "graph_ppmi.npy", pruned_ppmi.astype(np.float32))
    np.save(output_dir / "graph_pmi.npy", pruned_ppmi.astype(np.float32))
    np.save(output_dir / "graph_embeddings.npy", embeddings.astype(np.float32))

    edge_df = build_edge_table(pruned_ppmi, labels)
    edge_df.to_csv(output_dir / "graph_edges_pruned.csv", index=False)

    config = {
        "stage": "M20_pruned_graph_embeddings",
        "base_graph_dir": str(base_dir),
        "base_ppmi_path": str(ppmi_path),
        "output_dir": str(output_dir),
        "prune_ratio": float(args.prune_ratio),
        "embedding_dim": int(args.embedding_dim),
        "num_labels": int(len(labels)),
        "ppmi_shape": list(ppmi.shape),
        "embedding_shape": list(embeddings.shape),
        "stats": stats,
        "note": "Graph pruning is applied only to train-built graph artifacts. No validation/test labels are used.",
    }

    save_json(config, output_dir / "m20_pruned_graph_config.json")

    with open(output_dir / "m20_pruned_graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== M20 PRUNED GRAPH SUMMARY ===\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved artifacts:")
    for p in sorted(output_dir.glob("*")):
        print(p)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build pruned graph embeddings for M20 from train-only M17 graph artifacts."
    )

    parser.add_argument(
        "--base_graph_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument("--prune_ratio", type=float, default=0.20)
    parser.add_argument("--embedding_dim", type=int, default=64)

    parser.add_argument("--no_keep_diagonal", action="store_true")
    parser.add_argument("--no_normalize_embeddings", action="store_true")

    args = parser.parse_args()
    main(args)

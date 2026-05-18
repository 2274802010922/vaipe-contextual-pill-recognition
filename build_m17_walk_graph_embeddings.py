"""
Build M17 graph embeddings via random walks + skip-gram (Node2Vec / DeepWalk
spirit), as an alternative to the PPMI + SVD embeddings produced by
build_m17_pika_graph_embeddings.py.

This is Run B of Phase 2 in the PIKA paper-faithful comparison track and is
closer to PIKA paper section 3.2 (knowledge graph embedding via co-occurrence walks).

Inputs (identical to build_m17_pika_graph_embeddings.py):
    --train_csv  with columns: prescription_key (group), pill_label (class)

Outputs (schema is IDENTICAL to build_m17_pika_graph_embeddings.py so that
train_m17_faithful_pika.py can consume either directory unchanged):
    graph_embeddings.npy           shape (num_classes, vector_size), float32
    graph_labels.json              sorted list of original labels
    label_to_idx.json              { str(original_label) -> idx }
    idx_to_label.json              { str(idx) -> original_label }
    graph_cooccur.npy              raw co-occurrence matrix
    graph_edges.csv                edge table with cooccur counts
    graph_class_stats.csv          per-class counts
    prescription_label_sets.csv    per-prescription label list
    m17_graph_config.json          contains "embedding_method"="walk_word2vec"
                                              "method"="walk"

Dependencies:
    gensim>=4.0   (skip-gram Word2Vec)

If gensim is not installed:
    pip install gensim
"""

import argparse
import ast
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
    _GENSIM_AVAILABLE = True
    _GENSIM_IMPORT_ERROR = None
except Exception as e:
    Word2Vec = None
    _GENSIM_AVAILABLE = False
    _GENSIM_IMPORT_ERROR = e


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


def build_prescription_label_sets(
    df,
    label_col,
    group_col,
    context_col=None,
    include_context_labels=False,
):
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

    for _, labels in group_to_labels.items():
        mapped = sorted({label_to_idx[int(lb)] for lb in labels if int(lb) in label_to_idx})
        if len(mapped) == 0:
            continue

        group_count += 1
        group_size_counter[len(mapped)] += 1

        for i in mapped:
            class_doc_count[i] += 1.0
            cooccur_count[i, i] += 1.0

        for i, j in combinations(mapped, 2):
            cooccur_count[i, j] += 1.0
            cooccur_count[j, i] += 1.0

    return group_count, class_doc_count, cooccur_count, group_size_counter


def build_neighbor_table(cooccur_count, include_self_loops=False):
    """Return list of (neighbors_array, weights_array) per node, normalized."""
    num_classes = cooccur_count.shape[0]
    table = []

    for i in range(num_classes):
        row = cooccur_count[i].copy()
        if not include_self_loops:
            row[i] = 0.0

        idx_nonzero = np.nonzero(row > 0)[0]
        if len(idx_nonzero) == 0:
            table.append((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
            continue

        weights = row[idx_nonzero].astype(np.float64)
        probs = weights / weights.sum()
        table.append((idx_nonzero.astype(np.int64), probs))

    return table


def random_walk(start_node, neighbor_table, walk_length, rng):
    walk = [start_node]
    current = start_node
    for _ in range(walk_length - 1):
        neighbors, probs = neighbor_table[current]
        if len(neighbors) == 0:
            break
        next_node = int(rng.choice(neighbors, p=probs))
        walk.append(next_node)
        current = next_node
    return walk


def generate_walks(neighbor_table, walks_per_node, walk_length, seed):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    num_classes = len(neighbor_table)
    nodes = list(range(num_classes))

    walks = []
    for _ in range(walks_per_node):
        py_rng.shuffle(nodes)
        for n in nodes:
            walks.append(random_walk(n, neighbor_table, walk_length, rng))
    return walks


def walks_to_str_tokens(walks):
    return [[str(int(n)) for n in w] for w in walks]


def train_skipgram(walks, vector_size, window, epochs, min_count, workers, seed):
    if not _GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is required for walk-based embeddings but is not installed. "
            "Install with: pip install gensim. "
            f"Original import error: {_GENSIM_IMPORT_ERROR!r}"
        )

    sentences = walks_to_str_tokens(walks)

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
        negative=10,
        ns_exponent=0.75,
    )
    return model


def vectors_to_matrix(model, num_classes, vector_size, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    embeddings = np.zeros((num_classes, vector_size), dtype=np.float32)

    missing = []
    for i in range(num_classes):
        key = str(i)
        if key in model.wv.key_to_index:
            embeddings[i] = model.wv[key].astype(np.float32)
        else:
            embeddings[i] = rng.standard_normal(vector_size).astype(np.float32) * 0.01
            missing.append(i)

    return embeddings, missing


def normalize_rows(matrix, eps=1e-12):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (matrix / norms).astype(np.float32)


def build_edge_table(cooccur_count, idx_to_label):
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
            })

    edge_df = pd.DataFrame(rows)
    if len(edge_df) > 0:
        edge_df = edge_df.sort_values(
            "cooccur_count", ascending=False
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
    if not _GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is required for walk-based embeddings but is not installed. "
            "Install with: pip install gensim. "
            f"Original import error: {_GENSIM_IMPORT_ERROR!r}"
        )

    ensure_dir(args.output_dir)
    out_dir = Path(args.output_dir)

    print("=== M17 WALK-BASED GRAPH EMBEDDING BUILDER ===")
    print("Train CSV:", args.train_csv)
    print("Output dir:", args.output_dir)
    print("Label col:", args.label_col)
    print("Group col:", args.group_col)
    print("Context col:", args.context_col)
    print("Include context labels:", args.include_context_labels)
    print("Walks per node:", args.walks_per_node)
    print("Walk length:", args.walk_length)
    print("Vector size:", args.vector_size)
    print("Window:", args.window)
    print("Skip-gram epochs:", args.sg_epochs)
    print("Min count:", args.min_count)
    print("Workers:", args.workers)
    print("Seed:", args.seed)

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

    neighbor_table = build_neighbor_table(
        cooccur_count=cooccur_count,
        include_self_loops=False,
    )

    isolated = [i for i, (n, _) in enumerate(neighbor_table) if len(n) == 0]
    print(f"\nIsolated nodes (no neighbors): {len(isolated)}")
    if isolated:
        print("Isolated idx (first 20):", isolated[:20])

    walks = generate_walks(
        neighbor_table=neighbor_table,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        seed=args.seed,
    )
    walk_lengths = [len(w) for w in walks]
    print(f"\nGenerated walks: {len(walks)}")
    print(f"Mean walk length: {float(np.mean(walk_lengths)):.2f}")
    print(f"Max walk length: {int(np.max(walk_lengths))}")
    print(f"Min walk length: {int(np.min(walk_lengths))}")

    model = train_skipgram(
        walks=walks,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.sg_epochs,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
    )

    embeddings, missing = vectors_to_matrix(
        model=model,
        num_classes=num_classes,
        vector_size=args.vector_size,
        rng_seed=args.seed,
    )
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Nodes missing vector (filled with small random): {len(missing)}")
    if missing:
        print("Missing idx (first 20):", missing[:20])

    if not args.no_normalize_embeddings:
        embeddings = normalize_rows(embeddings)
        print("Row-normalized embeddings to unit L2.")

    np.save(out_dir / "graph_cooccur.npy", cooccur_count.astype(np.float32))
    np.save(out_dir / "graph_embeddings.npy", embeddings.astype(np.float32))

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
        "method": "walk",
        "embedding_method": "walk_word2vec",
        "train_csv": args.train_csv,
        "output_dir": args.output_dir,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "context_col": args.context_col,
        "include_context_labels": bool(args.include_context_labels),
        "num_train_rows": int(len(df)),
        "num_groups": int(group_count),
        "num_classes": int(num_classes),
        "walks_per_node": int(args.walks_per_node),
        "walk_length": int(args.walk_length),
        "vector_size": int(args.vector_size),
        "window": int(args.window),
        "sg_epochs": int(args.sg_epochs),
        "min_count": int(args.min_count),
        "negative_samples": 10,
        "skipgram": True,
        "normalize_embeddings": (not args.no_normalize_embeddings),
        "seed": int(args.seed),
        "isolated_nodes": int(len(isolated)),
        "missing_vector_nodes": int(len(missing)),
        "important_note": (
            "Graph and walks were built from train split only. "
            "Skip-gram (Word2Vec sg=1) trained on the random walks, "
            "equivalent to a DeepWalk-style embedding."
        ),
    }
    save_json(config, out_dir / "m17_graph_config.json")

    with open(out_dir / "m17_graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== M17 WALK GRAPH EMBEDDING SUMMARY ===\n")
        f.write(f"Train CSV: {args.train_csv}\n")
        f.write(f"Train rows: {len(df)}\n")
        f.write(f"Train groups: {df[args.group_col].nunique()}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Valid prescription groups: {group_count}\n")
        f.write(f"Graph cooccur shape: {cooccur_count.shape}\n")
        f.write(f"Embeddings shape: {embeddings.shape}\n")
        f.write(f"Walks per node: {args.walks_per_node}\n")
        f.write(f"Walk length: {args.walk_length}\n")
        f.write(f"Vector size: {args.vector_size}\n")
        f.write(f"Window: {args.window}\n")
        f.write(f"Skip-gram epochs: {args.sg_epochs}\n")
        f.write(f"Isolated nodes: {len(isolated)}\n")
        f.write(f"Missing vector nodes: {len(missing)}\n")
        f.write("Method: random walk (weighted by co-occurrence) + Word2Vec sg=1.\n")
        f.write("Graph was built from train split only.\n")

    print("\nSaved artifacts:")
    for p in sorted(out_dir.glob("*")):
        print(p)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build M17 graph embeddings via random walks + skip-gram (Node2Vec / "
            "DeepWalk style). Output schema is identical to "
            "build_m17_pika_graph_embeddings.py."
        )
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M17_faithful_pika/walk_graph_artifacts",
    )

    parser.add_argument("--label_col", type=str, default="pill_label")
    parser.add_argument("--group_col", type=str, default="prescription_key")
    parser.add_argument("--context_col", type=str, default="context_labels")
    parser.add_argument("--include_context_labels", action="store_true")

    parser.add_argument("--walks_per_node", type=int, default=80)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--vector_size", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--sg_epochs", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--no_normalize_embeddings", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

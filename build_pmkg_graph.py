"""Build M17 graph: PMKG edges when available, else co-occurrence fallback."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from build_m17_pika_graph_embeddings import (
    build_class_stats, build_cooccurrence_graph, build_edge_table,
    build_label_mapping, build_prescription_label_sets, compute_ppmi,
    ensure_dir, load_train_metadata, save_json, spectral_embedding,
)

def load_pmkg_edges(pmkg_edges_csv, pmkg_dir):
    candidates = []
    if pmkg_edges_csv:
        candidates.append(Path(pmkg_edges_csv))
    if pmkg_dir:
        candidates += [Path(pmkg_dir) / "pmkg_edges.csv", Path(pmkg_dir) / "edges.csv"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if not {"src_label", "dst_label"}.issubset(df.columns):
                raise ValueError(f"PMKG CSV missing columns at {path}")
            if "weight" not in df.columns:
                df["weight"] = 1.0
            if "relation_type" not in df.columns:
                df["relation_type"] = "pmkg"
            print("Loaded PMKG:", path, "rows=", len(df))
            return df, str(path)
    return None, ""

def build_pmkg_adjacency(pmkg_df, label_to_idx, num_classes):
    adj = np.zeros((num_classes, num_classes), dtype=np.float64)
    mapped = skipped = 0
    for _, row in pmkg_df.iterrows():
        try:
            src, dst = int(float(row["src_label"])), int(float(row["dst_label"]))
        except Exception:
            skipped += 1
            continue
        if src not in label_to_idx or dst not in label_to_idx:
            skipped += 1
            continue
        i, j = label_to_idx[src], label_to_idx[dst]
        w = float(row.get("weight", 1.0))
        adj[i, j] += w
        if i != j:
            adj[j, i] += w
        mapped += 1
    return adj, mapped, skipped

def merge_adjacency(pmkg_adj, cooccur, pw, cw, self_loop):
    out = pw * pmkg_adj + cw * cooccur.astype(np.float64)
    if self_loop is not None:
        np.fill_diagonal(out, float(self_loop))
    return out.astype(np.float32)

def main(args):
    ensure_dir(args.output_dir)
    out = Path(args.output_dir)
    df = load_train_metadata(args.train_csv, args.label_col, args.group_col)
    labels, l2i, i2l = build_label_mapping(df, args.label_col)
    n = len(labels)
    g2l = build_prescription_label_sets(df, args.label_col, args.group_col, args.context_col, args.include_context_labels)
    gc, doc, cooccur, _ = build_cooccurrence_graph(g2l, l2i, n)
    pmkg_df, pmkg_src = load_pmkg_edges(args.pmkg_edges_csv, args.pmkg_dir)
    source = "cooccurrence_fallback"
    pmkg_adj = np.zeros((n, n))
    pmkg_mapped = 0
    if pmkg_df is not None and len(pmkg_df):
        pmkg_adj, pmkg_mapped, _ = build_pmkg_adjacency(pmkg_df, l2i, n)
        if pmkg_mapped > 0:
            source = "pmkg_plus_cooccurrence" if args.merge_with_cooccurrence else "pmkg_only"
    if source == "cooccurrence_fallback":
        matrix, note = cooccur.astype(np.float32), "Co-occurrence only (PMKG unavailable)."
    elif args.merge_with_cooccurrence:
        matrix = merge_adjacency(pmkg_adj, cooccur, args.pmkg_weight, args.cooccur_weight, args.self_loop_value)
        note = "Merged PMKG + co-occurrence."
    else:
        matrix = pmkg_adj.astype(np.float32)
        note = "PMKG only."
    pmi, ppmi = compute_ppmi(matrix.astype(np.float64), doc, gc, args.min_cooccur, args.self_loop_value)
    emb = spectral_embedding(ppmi, args.embedding_dim, normalize=not args.no_normalize_embeddings)
    np.save(out / "graph_cooccur.npy", cooccur.astype(np.float32))
    np.save(out / "graph_pmkg_adj.npy", pmkg_adj.astype(np.float32))
    np.save(out / "graph_merged_adj.npy", matrix.astype(np.float32))
    np.save(out / "graph_pmi.npy", ppmi.astype(np.float32))
    np.save(out / "graph_ppmi.npy", ppmi.astype(np.float32))
    np.save(out / "graph_raw_pmi.npy", pmi.astype(np.float32))
    np.save(out / "graph_embeddings.npy", emb.astype(np.float32))
    save_json([int(x) for x in labels], out / "graph_labels.json")
    save_json({str(k): int(v) for k, v in l2i.items()}, out / "label_to_idx.json")
    save_json({str(k): int(v) for k, v in i2l.items()}, out / "idx_to_label.json")
    build_class_stats(labels, doc, l2i, df, args.label_col).to_csv(out / "graph_class_stats.csv", index=False)
    build_edge_table(cooccur, pmi, ppmi, i2l).to_csv(out / "graph_edges.csv", index=False)
    save_json({"graph_source": source, "merge_note": note, "pmkg_edges_source": pmkg_src, "pmkg_mapped_edges": pmkg_mapped, "num_classes": n}, out / "pmkg_graph_config.json")
    print("graph_source:", source, "|", note)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PMKG graph with co-occurrence fallback")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--output_dir", default="/content/drive/MyDrive/model/M17_faithful_pika/graph_artifacts_pmkg")
    p.add_argument("--pmkg_edges_csv", default="")
    p.add_argument("--pmkg_dir", default="")
    p.add_argument("--merge_with_cooccurrence", action="store_true", default=True)
    p.add_argument("--no_merge_with_cooccurrence", dest="merge_with_cooccurrence", action="store_false")
    p.add_argument("--pmkg_weight", type=float, default=0.5)
    p.add_argument("--cooccur_weight", type=float, default=0.5)
    p.add_argument("--label_col", default="pill_label")
    p.add_argument("--group_col", default="prescription_key")
    p.add_argument("--context_col", default="context_labels")
    p.add_argument("--include_context_labels", action="store_true")
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--min_cooccur", type=int, default=1)
    p.add_argument("--self_loop_value", type=float, default=1.0)
    p.add_argument("--no_normalize_embeddings", action="store_true")
    main(p.parse_args())
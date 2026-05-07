import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_label_list(label_string):
    if pd.isna(label_string) or str(label_string).strip() == "":
        return set()

    labels = []
    for x in str(label_string).split(","):
        x = x.strip()
        if x == "":
            continue
        try:
            labels.append(int(float(x)))
        except Exception:
            labels.append(x)

    return set(labels)


def infer_prediction_columns(df):
    true_col = None
    pred_col = None

    for c in ["true_mapped_label", "mapped_label", "pill_label"]:
        if c in df.columns:
            true_col = c
            break

    for c in [
        "ensemble_pred_mapped_label",
        "pred_mapped_label",
        "pred_idx_final",
        "pred_idx_first",
    ]:
        if c in df.columns:
            pred_col = c
            break

    if true_col is None or pred_col is None:
        raise ValueError(f"Cannot infer columns. Found columns: {df.columns.tolist()}")

    return true_col, pred_col


def compute_metrics(y_true, y_pred, labels_all=None):
    acc = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    result = {
        "accuracy": float(acc),
        "macro_precision_present": float(p),
        "macro_recall_present": float(r),
        "macro_f1_present": float(f1),
        "weighted_f1": float(wf1),
    }

    if labels_all is not None:
        p_all, r_all, f1_all, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_all,
            average="macro",
            zero_division=0,
        )

        result.update({
            "macro_precision_all_candidate": float(p_all),
            "macro_recall_all_candidate": float(r_all),
            "macro_f1_all_candidate": float(f1_all),
        })

    return result


def evaluate_model_on_candidates(pred_path, candidates_df, model_name):
    pred_df = pd.read_csv(pred_path)
    true_col, pred_col = infer_prediction_columns(pred_df)

    pred_df = pred_df.copy()
    pred_df[true_col] = pred_df[true_col].astype(int)
    pred_df[pred_col] = pred_df[pred_col].astype(int)

    rows = []

    for _, cand in candidates_df.iterrows():
        candidate_name = cand["candidate"]
        labels = parse_label_list(cand["labels"])
        labels_int = sorted([int(x) for x in labels])

        sub = pred_df[pred_df[true_col].isin(labels_int)].copy()

        if len(sub) == 0:
            continue

        y_true = sub[true_col].astype(int).tolist()
        y_pred = sub[pred_col].astype(int).tolist()

        metrics_present = compute_metrics(y_true, y_pred, labels_all=labels_int)

        rows.append({
            "model": model_name,
            "candidate": candidate_name,
            "candidate_num_labels": int(cand["num_labels"]),
            "candidate_train_rows": int(cand["train_rows"]),
            "candidate_val_rows": int(cand["val_rows"]),
            "candidate_test_rows": int(cand["test_rows"]),
            "eval_rows": int(len(sub)),
            **metrics_present,
        })

    return pd.DataFrame(rows)


def main(args):
    ensure_dir(args.output_dir)
    output_dir = Path(args.output_dir)

    print("=== EVALUATE CANDIDATE BENCHMARKS ===")
    print("Candidate file:", args.candidates_csv)
    print("M19 predictions:", args.m19_predictions)
    print("M22 predictions:", args.m22_predictions)
    print("Output dir:", output_dir)

    candidates_df = pd.read_csv(args.candidates_csv)

    all_results = []

    if args.m19_predictions:
        m19_df = evaluate_model_on_candidates(
            pred_path=args.m19_predictions,
            candidates_df=candidates_df,
            model_name="M19",
        )
        all_results.append(m19_df)

    if args.m22_predictions:
        m22_df = evaluate_model_on_candidates(
            pred_path=args.m22_predictions,
            candidates_df=candidates_df,
            model_name="M22",
        )
        all_results.append(m22_df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df = result_df.sort_values(
        ["candidate_num_labels", "model"],
        ascending=[False, True],
    )

    out_csv = output_dir / "candidate_benchmark_results.csv"
    result_df.to_csv(out_csv, index=False)

    print("\n=== CANDIDATE BENCHMARK RESULTS ===")
    display_cols = [
        "model",
        "candidate",
        "candidate_num_labels",
        "eval_rows",
        "accuracy",
        "macro_f1_present",
        "macro_f1_all_candidate",
        "weighted_f1",
    ]

    print(result_df[display_cols].to_string(index=False))

    # Pivot for easy comparison
    pivot = result_df.pivot_table(
        index=["candidate", "candidate_num_labels", "eval_rows"],
        columns="model",
        values=["accuracy", "macro_f1_present", "macro_f1_all_candidate", "weighted_f1"],
    )

    pivot_csv = output_dir / "candidate_benchmark_results_pivot.csv"
    pivot.to_csv(pivot_csv)

    print("\nSaved:")
    print(out_csv)
    print(pivot_csv)

    summary = {
        "candidates_csv": args.candidates_csv,
        "m19_predictions": args.m19_predictions,
        "m22_predictions": args.m22_predictions,
        "output_csv": str(out_csv),
        "pivot_csv": str(pivot_csv),
    }

    save_json(summary, output_dir / "candidate_benchmark_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate existing predictions on audit candidate benchmarks."
    )

    parser.add_argument("--candidates_csv", type=str, required=True)
    parser.add_argument("--m19_predictions", type=str, default="")
    parser.add_argument("--m22_predictions", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)

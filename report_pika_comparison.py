"""Summarize audit + M17 eval for paper-track comparison."""
import argparse, json
from pathlib import Path
import pandas as pd

def main(args):
    audit = Path(args.audit_dir)
    print("=== PIKA PAPER-TRACK REPORT ===")
    c = audit / "audit_candidate_benchmarks.csv"
    if c.exists():
        d = pd.read_csv(c)
        print(d[["candidate","num_labels","train_rows","val_rows","test_rows"]].to_string(index=False))
    m = Path(args.m17_eval_dir) / "m17_test_metrics.json"
    if m.exists():
        obj = json.loads(m.read_text(encoding="utf-8"))
        for k in ["accuracy","macro_f1_present_classes","macro_f1_all_classes","weighted_f1"]:
            if k in obj:
                print(f"{k}: {obj[k]}")
    b = Path(args.candidate_eval_dir) / "candidate_benchmark_results.csv"
    if b.exists():
        print(pd.read_csv(b).head(25).to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit_dir", default="/content/drive/MyDrive/model/audit_pika_protocol_v1")
    p.add_argument("--m17_eval_dir", default="/content/drive/MyDrive/model/M17_faithful_pika/test_eval_resnet50")
    p.add_argument("--candidate_eval_dir", default="/content/drive/MyDrive/model/audit_pika_protocol_v1/candidate_eval")
    main(p.parse_args())
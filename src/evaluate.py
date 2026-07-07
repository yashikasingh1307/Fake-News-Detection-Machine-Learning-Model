"""
Step 7: Combine baseline and transformer results into one comparison table.

Run after baseline_model.py and train_transformer.py have both produced
their output files.

Run:
    python src/evaluate.py
Output:
    reports/final_comparison.csv
"""

import os
import json
import pandas as pd

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def main():
    baseline_path = os.path.join(REPORTS_DIR, "baseline_results.csv")
    transformer_path = os.path.join(REPORTS_DIR, "transformer_eval.json")

    rows = []

    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        rows.extend(baseline_df.to_dict("records"))
    else:
        print(f"Missing {baseline_path} — run baseline_model.py first")

    if os.path.exists(transformer_path):
        with open(transformer_path) as f:
            t = json.load(f)
        rows.append({
            "model": "distilbert_finetuned",
            "accuracy": t.get("eval_accuracy"),
            "precision": t.get("eval_precision"),
            "recall": t.get("eval_recall"),
            "f1": t.get("eval_f1"),
        })
    else:
        print(f"Missing {transformer_path} — run train_transformer.py first")

    if not rows:
        print("No results found. Run baseline_model.py and train_transformer.py first.")
        return

    comparison_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("\nFinal model comparison:")
    print(comparison_df.to_string(index=False))

    out_path = os.path.join(REPORTS_DIR, "final_comparison.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()

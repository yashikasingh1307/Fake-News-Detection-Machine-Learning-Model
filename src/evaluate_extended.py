"""
Extended experimental evaluation for the fine-tuned transformer:
confusion matrix, ROC curve, Precision-Recall curve, inference timing,
and a qualitative error analysis (misclassified examples, broken down
by source dataset).

Run:
    python src/evaluate_extended.py

Output (in reports/figures/ and reports/):
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    error_analysis.csv          -- all misclassified test-set rows
    error_analysis_by_source.csv -- misclassification rate per source
    timing.json                 -- inference latency/throughput stats
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from preprocess import apply_cleaning

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MAX_LENGTH = 256
BATCH_SIZE = 32


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def predict_probs_batched(texts, model, tokenizer, batch_size=BATCH_SIZE):
    """Returns array of P(fake) for each text, plus total/average timing."""
    all_probs = []
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()  # P(fake)
        all_probs.append(probs)
    elapsed = time.perf_counter() - start
    return np.concatenate(all_probs), elapsed


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading model and test set...")
    model, tokenizer = load_model()
    test_df = apply_cleaning(pd.read_csv(os.path.join(DATA_DIR, "unified_test.csv")))
    texts = test_df["text_transformer"].tolist()
    y_true = test_df["label"].values

    print(f"Running inference on {len(texts)} test examples...")
    probs_fake, elapsed = predict_probs_batched(texts, model, tokenizer)
    y_pred = (probs_fake >= 0.5).astype(int)

    n = len(texts)
    per_sample_ms = (elapsed / n) * 1000
    throughput = n / elapsed
    timing = {
        "test_set_size": n,
        "total_inference_seconds": round(elapsed, 3),
        "avg_latency_ms_per_sample": round(per_sample_ms, 3),
        "throughput_samples_per_sec": round(throughput, 2),
        "batch_size_used": BATCH_SIZE,
        "note": "Training wall-clock time should be taken from the "
                "'train_runtime' field Trainer prints during "
                "train_transformer.py -- check your Colab run's console output.",
    }
    with open(os.path.join(REPORTS_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print("Timing:", timing)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix -- DistilBERT (fine-tuned)")
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), bbox_inches="tight", dpi=200)
    plt.close()
    print("Saved confusion_matrix.png")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, probs_fake)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- DistilBERT (fine-tuned)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved roc_curve.png (AUC={roc_auc:.3f})")

    # --- Precision-Recall curve ---
    precision, recall, _ = precision_recall_curve(y_true, probs_fake)
    ap = average_precision_score(y_true, probs_fake)
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve -- DistilBERT (fine-tuned)")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curve.png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved pr_curve.png (AP={ap:.3f})")

    # --- Error analysis ---
    test_df = test_df.copy()
    test_df["pred_label"] = y_pred
    test_df["prob_fake"] = probs_fake
    test_df["correct"] = test_df["label"] == test_df["pred_label"]

    errors = test_df[~test_df["correct"]][["text", "source", "label", "pred_label", "prob_fake"]]
    errors_path = os.path.join(REPORTS_DIR, "error_analysis.csv")
    errors.to_csv(errors_path, index=False)
    print(f"Saved {len(errors)} misclassified examples to {errors_path}")

    by_source = (
        test_df.groupby("source")["correct"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "correct_count"})
    )
    by_source["error_rate"] = 1 - (by_source["correct_count"] / by_source["total"])
    by_source_path = os.path.join(REPORTS_DIR, "error_analysis_by_source.csv")
    by_source.to_csv(by_source_path)
    print(f"Saved per-source error rates to {by_source_path}")
    print("\nError rate by source:")
    print(by_source)

    print(f"\nAll done. Figures in {FIGURES_DIR}, tables in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
"""
Estimates total training time WITHOUT doing a full retrain -- runs a small
number of real training steps locally, measures throughput, then
extrapolates to the full training run's step count.

This is an ESTIMATE based on local hardware, not the actual measured
Colab GPU time. Label it as such in any report.

Run:
    python src/estimate_training_time.py
Output:
    Prints an estimated train_runtime in seconds, and writes it to
    reports/train_time_estimate.json
"""

import os
import json
import time
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)

from preprocess import apply_cleaning

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
BENCHMARK_STEPS = 20  # how many real steps to time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "unified_train.csv"))
    train_df = apply_cleaning(train_df)[["text_transformer", "label"]].rename(
        columns={"text_transformer": "text"}
    )
    total_train_rows = len(train_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # Small slice just for timing -- enough batches to get a stable measurement
    sample_rows = BENCHMARK_STEPS * BATCH_SIZE + BATCH_SIZE
    bench_df = train_df.iloc[:sample_rows]
    bench_ds = Dataset.from_pandas(bench_df).map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="/tmp/benchmark_run",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        max_steps=BENCHMARK_STEPS,
        logging_steps=BENCHMARK_STEPS,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=bench_ds,
        processing_class=tokenizer, data_collator=data_collator,
    )

    print(f"Timing {BENCHMARK_STEPS} real training steps on this machine...")
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    seconds_per_step = elapsed / BENCHMARK_STEPS
    total_steps = (total_train_rows // BATCH_SIZE) * EPOCHS
    estimated_total_seconds = seconds_per_step * total_steps

    result = {
        "measured_on": "local CPU/GPU (whatever this machine has)",
        "benchmark_steps": BENCHMARK_STEPS,
        "seconds_per_step_measured": round(seconds_per_step, 3),
        "total_train_rows": total_train_rows,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "total_steps_full_run": total_steps,
        "estimated_train_runtime_seconds": round(estimated_total_seconds, 1),
        "estimated_train_runtime_minutes": round(estimated_total_seconds / 60, 1),
        "note": "This is an extrapolated ESTIMATE from a short local benchmark, "
                "not the actual measured Colab GPU training time. Colab's T4 GPU "
                "is typically faster than a local CPU, so real Colab runtime was "
                "likely shorter than this estimate if this ran on CPU.",
    }

    with open(os.path.join(REPORTS_DIR, "train_time_estimate.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\nEstimate:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
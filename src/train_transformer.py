"""
Step 5-6: Tokenize and fine-tune a transformer (DistilBERT by default) on
the unified dataset using Hugging Face's Trainer API.

DistilBERT is used as the default because it gives a good accuracy/compute
tradeoff (see Anggrainingsih et al. 2022, Alghamdi et al. 2025 in the
reference sheet) and trains comfortably on a free Colab GPU.

Run:
    python src/train_transformer.py
Output:
    models/transformer/            (fine-tuned model + tokenizer)
    reports/transformer_eval.json  (test-set accuracy/precision/recall/F1)
    reports/train_metrics.json     (train_runtime and other training stats
                                     -- saved explicitly so this is never
                                     lost even if the console output is)
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from preprocess import apply_cleaning

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_split(name):
    path = os.path.join(DATA_DIR, f"unified_{name}.csv")
    df = pd.read_csv(path)
    df = apply_cleaning(df)
    return df[["text_transformer", "label"]].rename(columns={"text_transformer": "text"})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    # Save training stats (train_runtime, samples/sec, etc.) explicitly to
    # disk immediately -- don't rely on scrolling back through console
    # output later, especially important on Colab where sessions can reset.
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    with open(os.path.join(REPORTS_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    print("Train metrics:", train_metrics)

    test_results = trainer.evaluate(test_ds)
    print("Test set results:", test_results)

    with open(os.path.join(REPORTS_DIR, "transformer_eval.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)
    print(f"Model saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
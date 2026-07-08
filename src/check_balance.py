"""
Diagnostic: check label balance across datasets and splits.
Run this if predictions look biased toward one class.

Run:
    python src/check_balance.py
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"unified_{split}.csv")
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        df = pd.read_csv(path)
        print(f"\n=== {split.upper()} ({len(df)} rows) ===")
        print("Overall label balance:")
        print(df["label"].value_counts(normalize=True).rename({0: "real", 1: "fake"}))
        print("\nPer-source label balance:")
        print(
            df.groupby("source")["label"]
            .value_counts(normalize=True)
            .rename("proportion")
            .to_string()
        )
        print("\nRows per source:")
        print(df["source"].value_counts())


if __name__ == "__main__":
    main()
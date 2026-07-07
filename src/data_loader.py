"""
Step 1-2: Load LIAR, FakeNewsNet, and Fakeddit, and unify them into a single
schema: text | label (0=real, 1=fake) | source | split

Run:
    python src/data_loader.py
Output:
    data/processed/unified_train.csv
    data/processed/unified_val.csv
    data/processed/unified_test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

RANDOM_SEED = 42


def load_liar():
    """LIAR has 6-way labels; we collapse to binary.
    true/mostly-true/half-true -> real (0)
    barely-true/false/pants-fire -> fake (1)
    """
    cols = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party", "barely_true_c", "false_c", "half_true_c",
        "mostly_true_c", "pants_on_fire_c", "context",
    ]
    frames = []
    for fname, split in [("train.tsv", "train"), ("valid.tsv", "val"), ("test.tsv", "test")]:
        path = os.path.join(RAW_DIR, "liar", fname)
        if not os.path.exists(path):
            print(f"[liar] missing {path}, skipping")
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=cols)
        df["split"] = split
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["text", "label", "source", "split"])

    df = pd.concat(frames, ignore_index=True)

    real_labels = {"true", "mostly-true", "half-true"}
    df["label"] = df["label"].apply(lambda x: 0 if x in real_labels else 1)
    df["text"] = df["statement"]
    df["source"] = "liar"
    return df[["text", "label", "source", "split"]]


def load_fakenewsnet():
    """FakeNewsNet ships as separate fake/real CSVs per sub-source
    (politifact, gossipcop). We combine and assign labels from filename.
    No official train/val/test split, so we split it ourselves later.
    """
    files = {
        "politifact_fake.csv": 1,
        "politifact_real.csv": 0,
        "gossipcop_fake.csv": 1,
        "gossipcop_real.csv": 0,
    }
    frames = []
    for fname, label in files.items():
        path = os.path.join(RAW_DIR, "fakenewsnet", fname)
        if not os.path.exists(path):
            print(f"[fakenewsnet] missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        # FakeNewsNet CSVs typically have a 'title' column; use that as text
        text_col = "title" if "title" in df.columns else df.columns[0]
        out = pd.DataFrame({
            "text": df[text_col],
            "label": label,
            "source": "fakenewsnet",
        })
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["text", "label", "source", "split"])

    df = pd.concat(frames, ignore_index=True)
    df["split"] = None  # assigned later via train_test_split
    return df


def load_fakeddit():
    """Fakeddit provides its own train/validate/test TSVs with a 2-way label
    column ('2_way_label'): 0 = real, 1 = fake.
    """
    frames = []
    for fname, split in [
        ("train.tsv", "train"),
        ("validate.tsv", "val"),
        ("test.tsv", "test"),
    ]:
        path = os.path.join(RAW_DIR, "fakeddit", fname)
        if not os.path.exists(path):
            print(f"[fakeddit] missing {path}, skipping")
            continue
        df = pd.read_csv(path, sep="\t")
        text_col = "clean_title" if "clean_title" in df.columns else "title"
        out = pd.DataFrame({
            "text": df[text_col],
            "label": df["2_way_label"],
            "source": "fakeddit",
            "split": split,
        })
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["text", "label", "source", "split"])

    return pd.concat(frames, ignore_index=True)


def assign_missing_splits(df, val_size=0.15, test_size=0.15):
    """FakeNewsNet has no built-in split; stratified-split it per source."""
    result = []
    for source, group in df.groupby("source"):
        if group["split"].isna().all():
            train, temp = train_test_split(
                group, test_size=(val_size + test_size),
                stratify=group["label"], random_state=RANDOM_SEED,
            )
            val, test = train_test_split(
                temp, test_size=test_size / (val_size + test_size),
                stratify=temp["label"], random_state=RANDOM_SEED,
            )
            train = train.copy(); train["split"] = "train"
            val = val.copy(); val["split"] = "val"
            test = test.copy(); test["split"] = "test"
            result.extend([train, val, test])
        else:
            result.append(group)
    return pd.concat(result, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    liar = load_liar()
    fnn = load_fakenewsnet()
    fakeddit = load_fakeddit()

    combined = pd.concat([liar, fnn, fakeddit], ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined["label"] = combined["label"].astype(int)

    combined = assign_missing_splits(combined)

    print("Rows per source:")
    print(combined.groupby("source").size())
    print("\nLabel balance:")
    print(combined.groupby(["source", "label"]).size())

    for split in ["train", "val", "test"]:
        subset = combined[combined["split"] == split]
        out_path = os.path.join(OUT_DIR, f"unified_{split}.csv")
        subset[["text", "label", "source"]].to_csv(out_path, index=False)
        print(f"Wrote {len(subset)} rows to {out_path}")


if __name__ == "__main__":
    main()

"""
Step 3: Text cleaning.

Two variants are provided because classical ML (TF-IDF) and transformers
benefit from different preprocessing:
  - clean_for_classical: lowercases, strips punctuation/stopwords-friendly
  - clean_for_transformer: light touch only (transformers use casing/punctuation
    as signal, and their own tokenizer handles the rest)
"""

import re


URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def _base_clean(text: str) -> str:
    text = str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_for_transformer(text: str) -> str:
    """Minimal cleaning — keep case and punctuation, transformers use them."""
    return _base_clean(text)


def clean_for_classical(text: str) -> str:
    """Aggressive cleaning for TF-IDF baselines."""
    text = _base_clean(text).lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def apply_cleaning(df, column="text"):
    """Returns a copy of df with two new columns:
    text_classical, text_transformer
    """
    df = df.copy()
    df["text_classical"] = df[column].apply(clean_for_classical)
    df["text_transformer"] = df[column].apply(clean_for_transformer)
    return df

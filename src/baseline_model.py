"""
Step 4: Classical ML baselines (TF-IDF + Naive Bayes / SVM / Logistic
Regression / Random Forest), matching the approaches in Sajjad Ahmed et al.
2020 and Khanam et al. 2021 from the reference sheet.

Run:
    python src/baseline_model.py
Output:
    reports/baseline_results.csv
    models/baseline_<model_name>.joblib + models/tfidf_vectorizer.joblib
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from preprocess import apply_cleaning

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_split(name):
    path = os.path.join(DATA_DIR, f"unified_{name}.csv")
    df = pd.read_csv(path)
    return apply_cleaning(df)


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"model": name, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = load_split("train")
    test_df = load_split("test")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text_classical"])
    X_test = vectorizer.transform(test_df["text_classical"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_svm": LinearSVC(),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(evaluate(name, y_test, preds))
        joblib.dump(model, os.path.join(MODELS_DIR, f"baseline_{name}.joblib"))

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\nBaseline results:")
    print(results_df.to_string(index=False))

    out_path = os.path.join(REPORTS_DIR, "baseline_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

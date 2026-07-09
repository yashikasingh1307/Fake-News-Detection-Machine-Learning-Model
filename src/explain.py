"""
Step 9: Explainability layer. Uses SHAP to compute per-word feature
attributions for a transformer prediction, showing which words pushed
the model toward FAKE or REAL.

Run interactively:
    python src/explain.py

Or import and use programmatically:
    from explain import explain_text, load_explainer
    explainer, model, tokenizer = load_explainer()
    result = explain_text("Some headline", explainer)
"""

import os
import shap
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import clean_for_transformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")
MAX_LENGTH = 256
LABELS = {0: "REAL", 1: "FAKE"}


def load_explainer(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    def predict_proba(texts):
        """SHAP needs a function: list[str] -> probability array."""
        inputs = tokenizer(
            list(texts), return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1).numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, output_names=["REAL", "FAKE"])

    return explainer, model, tokenizer


def explain_text(text, explainer, model, tokenizer, top_k=8):
    """Returns predicted label, confidence, and the top contributing words
    for that prediction (positive = pushed toward the predicted label).
    """
    cleaned = clean_for_transformer(text)

    # Get the model's actual prediction directly -- this must match
    # predict.py exactly, so we don't infer the label from SHAP sums
    # (which can disagree with the real prediction in close-call cases
    # since they don't include the SHAP baseline/expected value).
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_label = LABELS[pred_idx]

    shap_values = explainer([cleaned])
    values = shap_values.values[0]  # shape: (tokens, num_classes)
    tokens = shap_values.data[0]

    # Contributions toward the model's actual predicted class, per token
    contribs = values[:, pred_idx]
    ranked = sorted(
        zip(tokens, contribs), key=lambda x: abs(x[1]), reverse=True
    )[:top_k]

    return {
        "text": text,
        "label": pred_label,
        "confidence": probs[pred_idx],
        "top_words": [
            {"word": str(tok).strip(), "impact": float(val)}
            for tok, val in ranked if str(tok).strip()
        ],
    }


def print_explanation(result):
    print(f"\nPrediction: {result['label']} (confidence: {result['confidence']:.2%})")
    print("Top contributing words (+ pushes toward this label, - pushes away):")
    for item in result["top_words"]:
        sign = "+" if item["impact"] >= 0 else "-"
        print(f"  {sign} {item['word']:<20} impact={item['impact']:.4f}")


def main():
    print("Loading model and building SHAP explainer (this can take a moment)...")
    explainer, model, tokenizer = load_explainer()
    print("Ready. Type a headline/article to explain its prediction, or 'quit' to exit.\n")

    while True:
        text = input("Text> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        result = explain_text(text, explainer, model, tokenizer)
        print_explanation(result)


if __name__ == "__main__":
    main()
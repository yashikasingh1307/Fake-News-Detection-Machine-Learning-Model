"""
Step 8: Load the fine-tuned transformer and predict on new text.

Run interactively:
    python src/predict.py

Or import and use programmatically:
    from predict import predict, load_model
    model, tokenizer = load_model()
    predict("Some headline or article text", model, tokenizer)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import clean_for_transformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")
MAX_LENGTH = 256
LABELS = {0: "REAL", 1: "FAKE"}


def load_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict(text, model, tokenizer):
    """Returns dict with label, confidence, and raw probabilities."""
    cleaned = clean_for_transformer(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())

    return {
        "text": text,
        "label": LABELS[pred_idx],
        "confidence": probs[pred_idx],
        "prob_real": probs[0],
        "prob_fake": probs[1],
    }


def main():
    print("Loading fine-tuned model...")
    model, tokenizer = load_model()
    print("Model loaded. Type a headline/article to classify it, or 'quit' to exit.\n")

    while True:
        text = input("Text> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        result = predict(text, model, tokenizer)
        print(
            f"  -> {result['label']} "
            f"(confidence: {result['confidence']:.2%}, "
            f"P(real)={result['prob_real']:.2%}, P(fake)={result['prob_fake']:.2%})\n"
        )


if __name__ == "__main__":
    main()
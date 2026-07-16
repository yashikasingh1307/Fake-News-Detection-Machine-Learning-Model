"""
Generates SHAP and LIME visualizations for a few chosen example predictions
(one clearly-fake, one clearly-real, one borderline) and saves them as
image/HTML files, ready to drop into your report.

Run:
    python src/shap_visualization.py

Output (in reports/figures/):
    shap_waterfall_<tag>.png   -- bar chart of word-level SHAP contributions
    shap_text_<tag>.html       -- inline red/blue highlighted text view
    lime_<tag>.png             -- LIME's word-importance bar chart
    lime_<tag>.html            -- LIME's classic highlighted HTML view
"""

import os
import shap
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe: works in Colab and with no display
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

from preprocess import clean_for_transformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
MAX_LENGTH = 256
LABELS = ["REAL", "FAKE"]

# Pick one clearly-fake, one clearly-real, and one borderline example so the
# figures in your report match a qualitative range of cases. Edit these to
# match whatever examples you've already discussed in your write-up.
EXAMPLES = [
    ("Government secretly controls weather using hidden satellites, insider claims.", "clearly_fake"),
    ("The Federal Reserve raised interest rates by a quarter point on Wednesday.", "clearly_real"),
    ("Scientists confirm the moon landing was staged using Hollywood sets.", "borderline"),
]


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def make_predict_proba(model, tokenizer):
    def predict_proba(texts):
        inputs = tokenizer(
            list(texts), return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1).numpy()
    return predict_proba


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    predict_proba = make_predict_proba(model, tokenizer)

    print("Building SHAP explainer...")
    masker = shap.maskers.Text(tokenizer)
    shap_explainer = shap.Explainer(predict_proba, masker, output_names=LABELS)

    lime_explainer = LimeTextExplainer(class_names=["real", "fake"])

    for text, tag in EXAMPLES:
        cleaned = clean_for_transformer(text)
        print(f"\nProcessing [{tag}]: {text}")

        probs = predict_proba([cleaned])[0]
        pred_idx = int(probs.argmax())
        pred_label = LABELS[pred_idx]
        print(f"  Prediction: {pred_label} ({probs[pred_idx]:.2%})")

        # --- SHAP waterfall plot ---
        try:
            shap_values = shap_explainer([cleaned])
            shap.plots.waterfall(shap_values[0, :, pred_idx], show=False)
            out_path = os.path.join(OUTPUT_DIR, f"shap_waterfall_{tag}.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close()
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  SHAP waterfall failed: {e}")

        # --- SHAP text highlight (HTML) ---
        try:
            html = shap.plots.text(shap_values[0, :, pred_idx], display=False)
            out_path = os.path.join(OUTPUT_DIR, f"shap_text_{tag}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  SHAP text view failed: {e}")

        # --- LIME ---
        try:
            exp = lime_explainer.explain_instance(cleaned, predict_proba, num_features=10)
            fig = exp.as_pyplot_figure()
            out_path = os.path.join(OUTPUT_DIR, f"lime_{tag}.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  Saved {out_path}")

            html_path = os.path.join(OUTPUT_DIR, f"lime_{tag}.html")
            exp.save_to_file(html_path)
            print(f"  Saved {html_path}")
        except Exception as e:
            print(f"  LIME failed: {e}")

    print(f"\nAll done. Files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
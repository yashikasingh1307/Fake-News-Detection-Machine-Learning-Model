"""
Runs the full SHAP -> Gemini explanation pipeline on a few chosen examples
and saves the EXACT prompt sent to Gemini and the EXACT response
received, so you can paste real examples into your report.

Requires:
    GEMINI_API_KEY

Run:
    python save_explanation_examples.py

Output:
    reports/genai_examples.json
    reports/genai_examples.txt
"""

import os
import json

from google import genai

from explain import load_explainer, explain_text
from generate_explanation import (
    build_prompt,
    generate_explanation,
    MODEL_NAME,
)

REPORTS_DIR = "reports"

EXAMPLES = [
    (
        "Government secretly controls weather using hidden satellites, insider claims.",
        "clearly_fake",
    ),
    (
        "The Federal Reserve raised interest rates by a quarter point on Wednesday.",
        "clearly_real",
    ),
    (
        "Scientists confirm the moon landing was staged using Hollywood sets.",
        "borderline",
    ),
]


def main():

    os.makedirs(REPORTS_DIR, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not found.")
        return

    client = genai.Client(api_key=api_key)

    print("Loading model and building SHAP explainer...")

    explainer, model, tokenizer = load_explainer()

    results = []
    text_lines = []

    for text, tag in EXAMPLES:

        print(f"\nProcessing [{tag}]")

        result = explain_text(
            text,
            explainer,
            model,
            tokenizer,
        )

        prompt = build_prompt(result)

        try:

            explanation = generate_explanation(
                result,
                client,
            )

        except Exception as e:

            explanation = f"[ERROR] {e}"

        entry = {
            "tag": tag,
            "input_text": text,
            "predicted_label": result["label"],
            "confidence": float(result["confidence"]),
            "top_words": result["top_words"],
            "exact_prompt_sent": prompt,
            "exact_llm_response": explanation,
            "model_used": MODEL_NAME,
        }

        results.append(entry)

        text_lines.append("=" * 80)
        text_lines.append(f"Example : {tag}")
        text_lines.append("=" * 80)
        text_lines.append(f"Input News:\n{text}\n")
        text_lines.append(
            f"Prediction : {result['label']} ({result['confidence']:.2%})"
        )

        text_lines.append("\nTop SHAP Words:")

        for word in result["top_words"]:
            text_lines.append(
                f'  {word["word"]:<20} {word["impact"]:+.4f}'
            )

        text_lines.append("\nPrompt Sent To Gemini:\n")
        text_lines.append(prompt)

        text_lines.append("\nGemini Response:\n")
        text_lines.append(explanation)
        text_lines.append("\n")

    json_path = os.path.join(
        REPORTS_DIR,
        "genai_examples.json",
    )

    txt_path = os.path.join(
        REPORTS_DIR,
        "genai_examples.txt",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    print("\nDone!")
    print(f"JSON saved to : {json_path}")
    print(f"TEXT saved to : {txt_path}")


if __name__ == "__main__":
    main()
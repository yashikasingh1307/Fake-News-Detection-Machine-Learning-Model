"""
Step 10: Uses Gemini to convert SHAP feature attributions into
a human-readable explanation for why the ML model predicted
FAKE or REAL news.

Requires:

Windows:
    $env:GEMINI_API_KEY="your-api-key"

Permanent:
    setx GEMINI_API_KEY "your-api-key"

Run:
    python src/generate_explanation.py
"""

import os

from google import genai
from google.genai import types

from explain import load_explainer, explain_text

MODEL_NAME = "gemini-2.5-flash"


def build_prompt(result):
    words_str = "\n".join(
        f'- "{w["word"]}" (impact {w["impact"]:+.3f})'
        for w in result["top_words"]
    )

    return f"""
You are explaining the prediction of a Fake News Detection AI.

Prediction:
{result['label']}

Confidence:
{result['confidence']:.2%}

News:

{result['text']}

Most influential words:

{words_str}

Write ONLY a concise explanation.

Requirements:

- Exactly 2-3 sentences.
- Mention the important words naturally.
- Explain why they influenced the prediction.
- Don't mention SHAP.
- Don't mention machine learning.
- Don't mention AI.
- Don't add headings.
- Don't use bullet points.
- Don't add warnings or disclaimers.
"""


def generate_explanation(result, client):
    prompt = build_prompt(result)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=300,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            ),
        ),
    )

    return response.text.strip()

    print("\n========== FULL RESPONSE ==========")
    print(response)
    print("===================================\n")

    if hasattr(response, "text"):
        print("TEXT:", repr(response.text))

    return response.text if response.text else "No explanation generated."


def main():

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    print("Loading model and building SHAP explainer...")

    explainer, model, tokenizer = load_explainer()

    print("Ready! Type a news article or headline.\n")

    while True:

        text = input("Text> ").strip()

        if text.lower() in {"quit", "exit"}:
            break

        if not text:
            continue

        result = explain_text(
            text,
            explainer,
            model,
            tokenizer,
        )

        print(
            f"\nPrediction: {result['label']} "
            f"(confidence: {result['confidence']:.2%})"
        )

        print("\nGenerating explanation...\n")

        try:

            explanation = generate_explanation(result, client)

            print(explanation)
            print()

        except Exception as e:

            print("\nGemini Error\n")
            print(e)
            print()


if __name__ == "__main__":
    main()
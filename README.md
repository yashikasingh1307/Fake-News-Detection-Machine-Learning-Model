# Fake News Detection using Machine Learning and Gen AI

Detects fake news using a fine-tuned transformer model, and generates
human-readable explanations for each prediction using SHAP/LIME feature
attributions fed into an LLM.

## Datasets
- **LIAR** — https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **FakeNewsNet** — https://github.com/KaiDMML/FakeNewsNet
- **Fakeddit** — https://github.com/entitize/Fakeddit

Download each dataset and place the raw files in `data/raw/<dataset_name>/`.
Raw data is git-ignored 
`data/raw/README.md`.

## Project structure
```
fake-news-detection/
├── data/
│   ├── raw/            # original downloaded datasets (git-ignored)
│   └── processed/      # cleaned, unified CSVs (git-ignored, regenerate locally)
├── notebooks/          # exploratory Colab/Jupyter notebooks
├── src/
│   ├── data_loader.py      # Step 1: load + unify all 3 datasets
│   ├── preprocess.py       # Step 3: cleaning for classical ML vs transformer
│   ├── baseline_model.py   # Step 4: TF-IDF + classical ML baselines
│   ├── train_transformer.py# Step 5-6: tokenize + fine-tune DistilBERT
│   └── evaluate.py         # Step 7: shared evaluation metrics
├── models/              # saved model checkpoints (git-ignored)
├── reports/             # metrics, plots, writeups
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone <your-repo-url>
cd fake-news-detection
pip install -r requirements.txt
```

## Running the pipeline
```bash
# Step 1-2: build unified dataset
python src/data_loader.py

# Step 4: run classical ML baselines
python src/baseline_model.py

# Step 5-6: fine-tune transformer
python src/train_transformer.py

# Step 7: evaluate everything and print comparison table
python src/evaluate.py
```

## Team workflow (GitHub)
- `main` branch is always in a working state — no direct pushes.
- One branch per task, e.g. `data-loader`, `baseline-svm`, `bert-finetune`, `xai-shap`, `genai-explain`.
- Open a PR into `main` when a piece works; at least one other teammate reviews before merge.
- Use GitHub Issues (or a linked Trello/Notion board) with one issue per pipeline step from the README structure above, so work on data / baselines / transformer / XAI / GenAI can run in parallel.
- Keep dataset files and model checkpoints OUT of git (see `.gitignore`) — share via Drive, and keep only code + configs + small metric reports in the repo.
- Notebooks: do exploration in Colab, then move finalized logic into `src/*.py` before merging, so the codebase stays reviewable/diffable.

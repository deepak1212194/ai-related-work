# 03 · SBERT Pair Trainer

A minimal, production-shaped toolkit for **fine-tuning a sentence encoder** on labelled (sentence-A, sentence-B, score) triples and evaluating it on a held-out test split with R², MAE, and RMSE.

The training recipe is the same one I use at work: `all-mpnet-base-v2` + `CosineSimilarityLoss`, a small batch size, conservative learning rate, and a strict held-out split. The dataset here is the **public STS Benchmark** (Semantic Textual Similarity) — *no proprietary data*.

## Why this design

Three things go wrong when teams fine-tune sentence encoders for matching/ranking:

1. **Catastrophic forgetting** from too-aggressive learning rates on a small base.
2. **Train/test leakage** because the held-out split was sliced from the same source the index was built on.
3. **No model-quality metric** — teams ship without ever computing R² / MAE on a clean held-out set.

This toolkit pins all three: small LR, a clean test split that's never seen training, and a single `eval.py` that produces R²/MAE/RMSE in one command.

## Quick start

```bash
cd 03-sbert-pair-trainer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download + split the public STS-B dataset
python -m src.data

# 2. Fine-tune (CPU-friendly: 1 epoch on a tiny subset by default)
python -m src.train --epochs 1 --batch-size 16

# 3. Evaluate on the held-out test set
python -m src.eval --checkpoint artifacts/checkpoint
```

## Project layout

```
03-sbert-pair-trainer/
├── src/
│   ├── data.py     # Module 1: download + 80/10/10 stratified split
│   ├── train.py    # Module 2: SentenceTransformer fine-tuning loop
│   └── eval.py     # Module 3: held-out R² / MAE / RMSE
├── data/           # populated by data.py
└── requirements.txt
```

## Training notes

| Setting | Value | Why |
|---|---|---|
| Base model | `sentence-transformers/all-mpnet-base-v2` | 768-dim, strong general-purpose encoder |
| Loss | `CosineSimilarityLoss` | Direct optimisation of cosine-sim toward the label |
| Batch size | 16 | Conservative; works on CPU and small GPUs |
| Learning rate | 2e-5 | Prevents catastrophic forgetting on small fine-tunes |
| Epochs | 1 (default) | Keep small for demo; bump for real runs |
| Eval | held-out test split, never seen during training | The single most important rule |

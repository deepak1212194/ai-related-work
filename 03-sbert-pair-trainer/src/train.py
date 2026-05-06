"""
train.py — SentenceTransformer Fine-tuning
============================================
SBERT Pair Trainer — Module 2

Fine-tunes a sentence encoder on labelled (sentence_a, sentence_b, score)
triples using CosineSimilarityLoss. Pinned to a conservative learning
rate to avoid catastrophic forgetting on the small base.

Usage:
    python -m src.train --epochs 1 --batch-size 16
"""

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoint"

TRAIN_CSV = DATA_DIR / "train.csv"

BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_FRAC = 0.1
DEFAULT_SUBSET = 2000   # cap rows for quick demo runs; set to 0 for full set


# ──────────────────────────────────────────────────────────────────────
# Module 1: Data loading
# ──────────────────────────────────────────────────────────────────────
def load_train_examples(subset: int) -> list[InputExample]:
    """Load the train split and (optionally) cap it for demo speed."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Train CSV not found at {TRAIN_CSV}. Run `python -m src.data` first."
        )

    df = pd.read_csv(TRAIN_CSV)
    if subset and subset > 0:
        df = df.head(subset).copy()
    print(f"[TRAIN] Loaded {len(df):,} training pairs")

    return [
        InputExample(
            texts=[str(row.sentence_a), str(row.sentence_b)],
            label=float(row.score),
        )
        for row in df.itertuples(index=False)
    ]


# ──────────────────────────────────────────────────────────────────────
# Module 2: Train
# ──────────────────────────────────────────────────────────────────────
def train(epochs: int, batch_size: int, lr: float, warmup_frac: float,
          subset: int) -> Path:
    """Run the SentenceTransformer fit loop and persist a checkpoint."""
    print(f"[TRAIN] Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    examples = load_train_examples(subset)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)

    # CosineSimilarityLoss directly optimises cosine(a, b) toward `label`.
    train_loss = losses.CosineSimilarityLoss(model=model)

    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * warmup_frac)

    print(
        f"[TRAIN] epochs={epochs}  batch_size={batch_size}  lr={lr}  "
        f"steps={total_steps}  warmup={warmup_steps}"
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(CHECKPOINT_DIR),
        save_best_model=True,
        show_progress_bar=False,
    )

    print(f"[SAVE] Checkpoint → {CHECKPOINT_DIR}")
    return CHECKPOINT_DIR


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the sentence encoder.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-frac", type=float, default=DEFAULT_WARMUP_FRAC)
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET,
                        help="Cap training rows (0 = use all).")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_frac=args.warmup_frac,
        subset=args.subset,
    )
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()

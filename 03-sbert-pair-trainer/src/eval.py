"""
eval.py — Held-Out Evaluation (R² / MAE / RMSE)
================================================
SBERT Pair Trainer — Module 3

Loads a fine-tuned checkpoint, encodes the held-out test split, computes
diagonal cosine similarity per pair, and reports R² / MAE / RMSE versus
the gold labels. Also reports the share of predictions within +/- 0.1 of
the label — a useful intuitive metric alongside the regression numbers.

Usage:
    python -m src.eval --checkpoint artifacts/checkpoint
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_CSV = DATA_DIR / "test.csv"

EVAL_BATCH_SIZE = 64
WITHIN_TOL = 0.1   # report % of predictions within +/- 0.1 of the label


# ──────────────────────────────────────────────────────────────────────
# Module 1: Load
# ──────────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path) -> SentenceTransformer:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Run `python -m src.train` first."
        )
    print(f"[EVAL] Loading checkpoint: {path}")
    return SentenceTransformer(str(path))


def load_test_set() -> pd.DataFrame:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found at {TEST_CSV}. Run `python -m src.data` first."
        )
    df = pd.read_csv(TEST_CSV)
    print(f"[EVAL] Test pairs: {len(df):,}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Module 2: Predict cosine similarity per pair
# ──────────────────────────────────────────────────────────────────────
def predict(model: SentenceTransformer, df: pd.DataFrame) -> np.ndarray:
    """Encode both sides separately, then take the diagonal of cosine sim."""
    a = model.encode(
        df["sentence_a"].tolist(),
        batch_size=EVAL_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    b = model.encode(
        df["sentence_b"].tolist(),
        batch_size=EVAL_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return (a * b).sum(axis=1)   # diagonal cosine sim because both are L2-normed


# ──────────────────────────────────────────────────────────────────────
# Module 3: Score
# ──────────────────────────────────────────────────────────────────────
def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "within_0.1": float(np.mean(np.abs(y_true - y_pred) <= WITHIN_TOL)),
    }


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned encoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    df = load_test_set()

    y_pred = predict(model, df)
    y_true = df["score"].to_numpy()
    metrics = score(y_true, y_pred)

    print("\n[EVAL] Test-set metrics:")
    print(f"  N         : {metrics['n']:,}")
    print(f"  R^2       : {metrics['r2']:.4f}")
    print(f"  MAE       : {metrics['mae']:.4f}")
    print(f"  RMSE      : {metrics['rmse']:.4f}")
    print(f"  Within ±0.1: {metrics['within_0.1']*100:.1f}%")


if __name__ == "__main__":
    main()

"""
data.py — Dataset Download & Split
===================================
SBERT Pair Trainer — Module 1

Downloads the public STS Benchmark dataset (sentence pair + similarity
score in [0, 5]), normalises labels to [0, 1], and writes train/val/test
CSVs to disk. The test split is the dataset's official one — never sliced
from train — so there is no leakage.

This script runs end-to-end: data.py → train.py → eval.py
"""

import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

DATASET_NAME = "sentence-transformers/stsb"
LABEL_MAX = 5.0   # STS-B scores are in [0, 5]; we normalise to [0, 1]


# ──────────────────────────────────────────────────────────────────────
# Module 1: Download
# ──────────────────────────────────────────────────────────────────────
def download_stsb() -> dict[str, pd.DataFrame]:
    """Download STS-B and return a dict of train / validation / test splits."""
    from datasets import load_dataset

    print(f"[DATA] Downloading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME)

    splits = {
        "train": ds["train"].to_pandas(),
        "val": ds["validation"].to_pandas(),
        "test": ds["test"].to_pandas(),
    }
    for name, df in splits.items():
        print(f"[DATA] {name}: {len(df):,} rows")
    return splits


# ──────────────────────────────────────────────────────────────────────
# Module 2: Normalise
# ──────────────────────────────────────────────────────────────────────
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise STS-B scores from [0, 5] to [0, 1] for cosine-sim training."""
    df = df.rename(columns={"sentence1": "sentence_a", "sentence2": "sentence_b"})
    df["score"] = (df["score"] / LABEL_MAX).clip(0.0, 1.0)
    return df[["sentence_a", "sentence_b", "score"]]


# ──────────────────────────────────────────────────────────────────────
# Module 3: Persist
# ──────────────────────────────────────────────────────────────────────
def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}
    for name, df in splits.items():
        df.to_csv(paths[name], index=False, encoding="utf-8")
        print(f"[SAVE] {name} → {paths[name]}  ({len(df):,} rows)")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        raw = download_stsb()
    except Exception as e:                        # noqa: BLE001
        sys.exit(f"[ERR] Could not download dataset: {e}")

    splits = {name: normalise(df) for name, df in raw.items()}
    save_splits(splits)
    print("[DONE] Data prep complete.")


if __name__ == "__main__":
    main()

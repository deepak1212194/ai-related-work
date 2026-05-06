"""
stage_eval.py — Stage 3: Held-out evaluation
=============================================
SBERT Pair Trainer — Pipeline

Loads the fine-tuned checkpoint, encodes the held-out test split,
and computes R² / MAE / RMSE / within-tolerance.

Writes both `metrics.json` (machine-readable) and a
`metrics_summary.txt` (human-readable, used by the dashboard).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config_loader import TrainConfig, common_argparser, load_config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Predict diagonal cosine similarities
# ──────────────────────────────────────────────────────────────────────
def _predict(model: SentenceTransformer, df: pd.DataFrame,
             batch_size: int) -> np.ndarray:
    a = model.encode(df["sentence_a"].tolist(),
                     batch_size=batch_size,
                     normalize_embeddings=True,
                     convert_to_numpy=True,
                     show_progress_bar=False)
    b = model.encode(df["sentence_b"].tolist(),
                     batch_size=batch_size,
                     normalize_embeddings=True,
                     convert_to_numpy=True,
                     show_progress_bar=False)
    return (a * b).sum(axis=1)


# ──────────────────────────────────────────────────────────────────────
# Module 2: Score
# ──────────────────────────────────────────────────────────────────────
def _score(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> dict:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "within_tolerance": float(np.mean(np.abs(y_true - y_pred) <= tol)),
        "tolerance": tol,
    }


# ──────────────────────────────────────────────────────────────────────
# Module 3: Run
# ──────────────────────────────────────────────────────────────────────
def run(cfg: TrainConfig) -> dict:
    test_csv = Path("data/test.csv")
    if not test_csv.exists():
        raise FileNotFoundError("data/test.csv missing — run stage_data first.")

    if not cfg.checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint at {cfg.checkpoint_dir} — run stage_train first."
        )

    log.info("[EVAL] checkpoint=%s", cfg.checkpoint_dir)
    model = SentenceTransformer(str(cfg.checkpoint_dir))

    df = pd.read_csv(test_csv)
    log.info("[EVAL] test pairs: %d", len(df))

    y_pred = _predict(model, df, cfg.eval_batch_size)
    y_true = df["score"].to_numpy()
    metrics = _score(y_true, y_pred, cfg.within_tolerance)

    log.info("[EVAL] R²=%.4f  MAE=%.4f  RMSE=%.4f  within±%.2f=%.1f%%",
             metrics["r2"], metrics["mae"], metrics["rmse"],
             metrics["tolerance"], metrics["within_tolerance"] * 100)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = common_argparser("SBERT eval stage").parse_args()
    cfg = load_config(args.config, args.override)
    metrics = run(cfg)

    metrics_path = cfg.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists() else {}
    )
    existing["eval"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # Also a simple text summary the dashboard can render
    summary_path = metrics_path.with_name("metrics_summary.txt")
    summary_path.write_text(
        f"R²            : {metrics['r2']:.4f}\n"
        f"MAE           : {metrics['mae']:.4f}\n"
        f"RMSE          : {metrics['rmse']:.4f}\n"
        f"Within ±{metrics['tolerance']:.2f}: {metrics['within_tolerance']*100:.1f}%\n"
        f"N             : {metrics['n']}\n",
        encoding="utf-8",
    )
    log.info("[EVAL] metrics → %s", metrics_path)


if __name__ == "__main__":
    main()

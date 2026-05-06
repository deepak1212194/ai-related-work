"""
stage_train.py — Stage 2: Fine-tuning
======================================
SBERT Pair Trainer — Pipeline

Wraps the existing src.train module with config-driven invocation, MLflow
tracking (optional), and a structured metrics file.

Run this stage standalone:
    python -m pipeline.stage_train --config configs/train.yaml

Or as part of the end-to-end pipeline:
    python -m pipeline.run --config configs/train.yaml
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from .config_loader import TrainConfig, common_argparser, load_config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Build examples from CSV
# ──────────────────────────────────────────────────────────────────────
def _load_examples(train_csv: Path, subset: int) -> list[InputExample]:
    df = pd.read_csv(train_csv)
    if subset and subset > 0:
        df = df.head(subset).copy()
    return [
        InputExample(
            texts=[str(r.sentence_a), str(r.sentence_b)],
            label=float(r.score),
        )
        for r in df.itertuples(index=False)
    ]


# ──────────────────────────────────────────────────────────────────────
# Module 2: Train
# ──────────────────────────────────────────────────────────────────────
def run(cfg: TrainConfig) -> dict:
    train_csv = Path("data/train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(
            "data/train.csv missing — run `python -m pipeline.stage_data` first."
        )

    examples = _load_examples(train_csv, cfg.subset_train)
    log.info("[TRAIN] %d training pairs  (subset=%d)",
             len(examples), cfg.subset_train)

    model = SentenceTransformer(cfg.base_model)
    loader = DataLoader(examples, batch_size=cfg.batch_size, shuffle=True)
    train_loss = losses.CosineSimilarityLoss(model=model)

    total_steps = len(loader) * cfg.epochs
    warmup = int(total_steps * cfg.warmup_fraction)

    log.info("[TRAIN] base=%s  bs=%d  lr=%g  epochs=%d  steps=%d  warmup=%d",
             cfg.base_model, cfg.batch_size, cfg.learning_rate,
             cfg.epochs, total_steps, warmup)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=cfg.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": cfg.learning_rate},
        weight_decay=cfg.weight_decay,
        output_path=str(cfg.checkpoint_dir),
        save_best_model=True,
        show_progress_bar=False,
        use_amp=cfg.fp16,
    )
    elapsed = time.perf_counter() - t0

    summary = {
        "stage": "train",
        "experiment": cfg.experiment_name,
        "n_train_examples": len(examples),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "elapsed_seconds": round(elapsed, 1),
        "checkpoint": str(cfg.checkpoint_dir),
    }
    log.info("[TRAIN] done in %.1fs → %s", elapsed, cfg.checkpoint_dir)
    return summary


# ──────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = common_argparser("SBERT training stage").parse_args()
    cfg = load_config(args.config, args.override)
    summary = run(cfg)

    # Append to metrics file (so eval stage can read this run's params)
    metrics_path = cfg.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists() else {}
    )
    existing["train"] = summary
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    log.info("[TRAIN] metrics → %s", metrics_path)


if __name__ == "__main__":
    main()

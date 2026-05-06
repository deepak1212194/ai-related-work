"""
config_loader.py — Frozen-config loader with CLI overrides
============================================================
SBERT Pair Trainer — Pipeline

Loads the YAML config, optionally applies `--override key=value` pairs,
and returns a Pydantic-validated `TrainConfig`. The pipeline stages
(train, eval, register) all consume the same config object so behaviour
is identical whether the pipeline is run end-to-end or stage-by-stage.
"""

import argparse
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────
class TrainConfig(BaseModel):
    experiment_name: str
    seed: int = 42

    # Data
    dataset: str
    label_max: float = 5.0
    subset_train: int = 0
    subset_val: int = 0
    subset_test: int = 0

    # Model
    base_model: str
    pooling: str = "mean"

    # Optimiser
    loss: str = "CosineSimilarityLoss"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 1
    warmup_fraction: float = 0.1
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    fp16: bool = False

    # Eval
    eval_batch_size: int = 64
    within_tolerance: float = 0.1

    # Artifacts
    artifacts_dir: Path = Field(default=Path("artifacts"))
    checkpoint_dir: Path = Field(default=Path("artifacts/checkpoint"))
    metrics_path: Path = Field(default=Path("artifacts/metrics.json"))
    mlflow_tracking: bool = False


# ──────────────────────────────────────────────────────────────────────
# Module 1: Loader
# ──────────────────────────────────────────────────────────────────────
def load_config(path: Path, overrides: Iterable[str] = ()) -> TrainConfig:
    """Read YAML + apply `--override key=value` strings."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"--override expected key=value, got '{o}'")
        k, v = o.split("=", 1)
        raw[k] = _coerce(v)
    return TrainConfig(**raw)


def _coerce(s: str) -> Any:
    """Best-effort scalar coercion of an --override RHS."""
    sl = s.strip()
    if sl.lower() in ("true", "false"):
        return sl.lower() == "true"
    try:
        return int(sl)
    except ValueError:
        pass
    try:
        return float(sl)
    except ValueError:
        pass
    return sl


# ──────────────────────────────────────────────────────────────────────
# Module 2: Shared CLI parser
# ──────────────────────────────────────────────────────────────────────
def common_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--override", action="append", default=[],
                   help="key=value (repeatable)")
    return p

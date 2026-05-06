"""
run.py — End-to-end pipeline driver
=====================================
SBERT Pair Trainer — Pipeline

Runs the four stages in order — data → train → eval → register —
with a single config file. Each stage is also independently runnable
via its own `python -m pipeline.stage_*` entry.

Usage:
    python -m pipeline.run --config configs/train.yaml
    python -m pipeline.run --config configs/train.yaml --override epochs=2
"""

import logging
import subprocess
import sys
from pathlib import Path

from .config_loader import common_argparser, load_config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Stage data (downloads + splits)
# ──────────────────────────────────────────────────────────────────────
def _stage_data() -> None:
    """Re-uses the existing src/data.py; here for pipeline orchestration."""
    log.info("[PIPELINE] stage 1/4: data")
    from src import data as data_mod
    data_mod.main()


# ──────────────────────────────────────────────────────────────────────
# Module 2: Run all stages
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = common_argparser("SBERT end-to-end pipeline").parse_args()
    cfg = load_config(args.config, args.override)
    log.info("[PIPELINE] experiment=%s  config=%s",
             cfg.experiment_name, args.config)

    # Stage 1: data prep
    _stage_data()

    # Stages 2–4: spawn each stage as a subprocess so failures don't
    # leak partial state into the next stage's process memory.
    overrides = []
    for o in args.override:
        overrides += ["--override", o]

    for stage in ("stage_train", "stage_eval", "stage_register"):
        log.info("[PIPELINE] running %s", stage)
        cmd = [
            sys.executable, "-m", f"pipeline.{stage}",
            "--config", str(args.config), *overrides,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            log.error("[PIPELINE] %s failed (rc=%d)", stage, rc)
            sys.exit(rc)

    log.info("[PIPELINE] complete.")


if __name__ == "__main__":
    main()

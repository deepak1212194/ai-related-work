"""
stage_register.py — Stage 4: Model registration
================================================
SBERT Pair Trainer — Pipeline

Packages the fine-tuned checkpoint + metrics into a versioned bundle
under `artifacts/registry/<experiment>-<version>/`. In a real Azure ML
deployment this stage would also push to an Azure ML Workspace registry
and a Managed Online Endpoint; here we keep it filesystem-local so the
demo is self-contained.

The registry layout mirrors a real model-registry:
    artifacts/registry/<experiment>-v1/
        ├── model/                  # the SBERT checkpoint
        ├── metrics.json            # train + eval metrics
        ├── config.yaml             # frozen config used for training
        └── MODEL_CARD.md           # auto-generated card
"""

import json
import logging
import shutil
from pathlib import Path

import yaml

from .config_loader import TrainConfig, common_argparser, load_config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Pick a version number (auto-increment)
# ──────────────────────────────────────────────────────────────────────
def _next_version(registry_root: Path, experiment: str) -> int:
    if not registry_root.exists():
        return 1
    existing = [
        p.name for p in registry_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{experiment}-v")
    ]
    if not existing:
        return 1
    versions = []
    for name in existing:
        try:
            versions.append(int(name.split("-v")[-1]))
        except ValueError:
            continue
    return max(versions, default=0) + 1


# ──────────────────────────────────────────────────────────────────────
# Module 2: Generate model card
# ──────────────────────────────────────────────────────────────────────
def _model_card(cfg: TrainConfig, metrics: dict, version: int) -> str:
    eval_m = metrics.get("eval", {})
    train_m = metrics.get("train", {})
    return (
        f"# {cfg.experiment_name} · v{version}\n\n"
        f"Fine-tuned `{cfg.base_model}` on `{cfg.dataset}` with "
        f"`{cfg.loss}`.\n\n"
        f"## Training\n\n"
        f"- Examples: {train_m.get('n_train_examples', 'n/a')}\n"
        f"- Epochs: {cfg.epochs}\n"
        f"- Batch size: {cfg.batch_size}\n"
        f"- Learning rate: {cfg.learning_rate}\n"
        f"- Wall time: {train_m.get('elapsed_seconds', 'n/a')}s\n\n"
        f"## Evaluation (held-out test set)\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| N | {eval_m.get('n', 'n/a')} |\n"
        f"| R² | {eval_m.get('r2', float('nan')):.4f} |\n"
        f"| MAE | {eval_m.get('mae', float('nan')):.4f} |\n"
        f"| RMSE | {eval_m.get('rmse', float('nan')):.4f} |\n"
        f"| Within ±{eval_m.get('tolerance', 0.1):.2f} | "
        f"{eval_m.get('within_tolerance', 0)*100:.1f}% |\n\n"
        f"## Reproducibility\n\n"
        f"`config.yaml` in this folder is the frozen config used for the run.\n"
    )


# ──────────────────────────────────────────────────────────────────────
# Module 3: Run
# ──────────────────────────────────────────────────────────────────────
def run(cfg: TrainConfig) -> dict:
    if not cfg.checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoint at {cfg.checkpoint_dir}")
    if not cfg.metrics_path.exists():
        raise FileNotFoundError(f"No metrics at {cfg.metrics_path}")

    metrics = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))

    registry_root = cfg.artifacts_dir / "registry"
    version = _next_version(registry_root, cfg.experiment_name)
    target = registry_root / f"{cfg.experiment_name}-v{version}"

    log.info("[REGISTER] target=%s", target)
    target.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint, metrics, config; write model card
    shutil.copytree(cfg.checkpoint_dir, target / "model", dirs_exist_ok=True)
    shutil.copy2(cfg.metrics_path, target / "metrics.json")
    (target / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (target / "MODEL_CARD.md").write_text(
        _model_card(cfg, metrics, version), encoding="utf-8",
    )

    log.info("[REGISTER] registered as %s-v%d", cfg.experiment_name, version)
    return {"version": version, "path": str(target)}


# ──────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = common_argparser("SBERT register stage").parse_args()
    cfg = load_config(args.config, args.override)
    info = run(cfg)
    log.info("[REGISTER] %s", info)


if __name__ == "__main__":
    main()

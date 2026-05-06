# 03 · SBERT Training Pipeline

A **production-shaped fine-tuning pipeline** for sentence encoders, with an Azure-ML-style stage layout, frozen YAML configs, a model registry on disk, and a small live metrics dashboard.

The pipeline runs against the public **STS Benchmark** dataset — *no proprietary data* is used or referenced.

## Highlights

- **Four config-driven stages:** `data → train → eval → register`. Run them individually or end-to-end.
- **Frozen YAML config** — every hyperparameter pinned in `configs/train.yaml`, with `--override key=value` for one-off changes.
- **Strict held-out evaluation** — uses STS-B's official test split. R² / MAE / RMSE / within-tolerance.
- **Filesystem model registry** at `artifacts/registry/<experiment>-v<n>/` — auto-incrementing versions, model card, frozen config, metrics.
- **Live metrics dashboard** (FastAPI + Chart.js) at `http://localhost:8003/` — KPI cards + registry table.
- **Same recipe, different scale** — the loss (`CosineSimilarityLoss`) and held-out discipline are exactly what I use in production at OneForma; this demo keeps them on a public dataset.
- **Dockerised** — one image runs either the pipeline or the dashboard.

## Architecture

```
configs/train.yaml
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ stage_data   │ →  │ stage_train  │ →  │ stage_eval   │ →  │ stage_register│
│ (STS-B DL)   │    │ (CosineSim)  │    │ (R² MAE …)   │    │ (versioned)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
   data/*.csv         artifacts/         artifacts/         artifacts/
                      checkpoint/        metrics.json       registry/
                                                              └─ <name>-v1/
                                                                  ├─ model/
                                                                  ├─ metrics.json
                                                                  ├─ config.yaml
                                                                  └─ MODEL_CARD.md

           ┌─────────────────────┐
           │  Dashboard (8003)   │  reads artifacts/metrics.json
           │  KPI cards + table  │  reads artifacts/registry/*
           └─────────────────────┘
```

## Project layout

```
03-sbert-pair-trainer/
├── configs/
│   └── train.yaml             # frozen, version-controlled config
├── pipeline/
│   ├── config_loader.py       # YAML + --override → TrainConfig
│   ├── stage_train.py         # stage 2: fine-tune
│   ├── stage_eval.py          # stage 3: held-out eval
│   ├── stage_register.py      # stage 4: filesystem registry
│   └── run.py                 # end-to-end driver
├── src/
│   ├── data.py                # stage 1: STS-B download + 0–1 normalise
│   ├── train.py               # original CLI trainer (still works standalone)
│   ├── eval.py                # original CLI evaluator
│   └── inference.py
├── dashboard/
│   ├── app.py                 # FastAPI: /api/metrics, /api/registry
│   └── index.html             # KPI cards + registry table
├── Dockerfile                 # serves dashboard by default
└── requirements.txt
```

## Run the pipeline end-to-end

```bash
cd 03-sbert-pair-trainer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run all four stages with the default config
python -m pipeline.run --config configs/train.yaml

# 2. Or override one knob without editing YAML
python -m pipeline.run --config configs/train.yaml --override epochs=2

# 3. Or run a single stage
python -m pipeline.stage_train  --config configs/train.yaml
python -m pipeline.stage_eval   --config configs/train.yaml
python -m pipeline.stage_register --config configs/train.yaml
```

## Open the dashboard

```bash
uvicorn dashboard.app:app --port 8003
open http://localhost:8003/                  # macOS / Linux
start http://localhost:8003/                 # Windows
```

The dashboard auto-reads `artifacts/metrics.json`, so re-running the pipeline updates the page on refresh.

## Run with Docker

```bash
# Default: serve the dashboard
docker build -t sbert-pipeline .
docker run --rm -p 8003:8003 -v $(pwd)/artifacts:/app/artifacts sbert-pipeline

# Or run a training pipeline inside the container
docker run --rm -v $(pwd)/artifacts:/app/artifacts -v $(pwd)/data:/app/data \
    sbert-pipeline python -m pipeline.run --config configs/train.yaml
```

## Configuration knobs (`configs/train.yaml`)

The full schema is enforced by `pipeline.config_loader.TrainConfig` (Pydantic) — so an invalid YAML field fails fast with a typed error instead of corrupting an artifact mid-run.

| Group | Field | Default |
|---|---|---|
| Data | `dataset`, `subset_train`, `subset_val`, `subset_test` | STS-B, 2000, 0, 0 |
| Model | `base_model`, `pooling` | `all-mpnet-base-v2`, `mean` |
| Optimiser | `loss`, `learning_rate`, `batch_size`, `epochs`, `warmup_fraction`, `optimizer`, `fp16` | CosineSim, 2e-5, 16, 1, 0.1, AdamW, false |
| Eval | `eval_batch_size`, `within_tolerance` | 64, 0.10 |
| Tracking | `mlflow_tracking` | false |

## Why this design

Three things go wrong when teams fine-tune sentence encoders for matching/ranking — and the pipeline pins all three by construction:

1. **Catastrophic forgetting** — guarded by the conservative `2e-5` learning rate and a 10% warmup.
2. **Train/test leakage** — guarded by using STS-B's *official* held-out test split, not a slice of train.
3. **No defensible model-quality metric** — guarded by `stage_eval` writing R²/MAE/RMSE/within-tol to `metrics.json` *before* `stage_register` is allowed to run.

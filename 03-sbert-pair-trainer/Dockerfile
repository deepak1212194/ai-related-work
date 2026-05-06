# syntax=docker/dockerfile:1.6
# Single image that can either run the training pipeline OR serve the
# metrics dashboard. Pick by overriding the CMD at run time.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pipeline/ ./pipeline
COPY src/ ./src
COPY dashboard/ ./dashboard
COPY configs/ ./configs

# Default: serve the dashboard. Override for training:
#   docker run --rm <image> python -m pipeline.run --config configs/train.yaml
EXPOSE 8003
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8003"]

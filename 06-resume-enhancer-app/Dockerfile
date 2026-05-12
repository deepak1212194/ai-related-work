FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RESUME_UI_PORT=7860

WORKDIR /app

# System packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for layer cache
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code
COPY app ./app
COPY skills ./skills
COPY data ./data
COPY ui ./ui
COPY examples ./examples

# Non-root user
RUN useradd -ms /bin/bash resume && chown -R resume:resume /app
USER resume

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=4s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1

CMD ["python", "-m", "ui.app"]

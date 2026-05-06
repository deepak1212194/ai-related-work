"""
config.py — Service configuration
==================================
RAG FAQ Service — App layer

Environment-driven configuration via pydantic-settings. Defaults are safe
for local dev; override via .env or real env vars in production.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────────────────────────────
# Path roots
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
UI_DIR = PROJECT_ROOT / "ui"


# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    """All runtime knobs in one place — overrideable via env."""

    # --- Service ---
    service_name: str = Field(default="rag-faq-service")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Retrieval ---
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = Field(default=3, ge=1, le=20)
    min_sim: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- LLM ---
    llm_model: str = Field(default="gpt-4o-mini")
    llm_max_tokens: int = Field(default=400, ge=1, le=4000)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_api_key: str | None = Field(default=None)

    # --- Storage ---
    docs_path: Path = Field(default=DATA_DIR / "sample_faqs.txt")
    index_path: Path = Field(default=ARTIFACTS_DIR / "faqs.faiss")
    meta_path: Path = Field(default=ARTIFACTS_DIR / "chunks.json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",          # e.g. RAG_TOP_K=5 overrides top_k
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()

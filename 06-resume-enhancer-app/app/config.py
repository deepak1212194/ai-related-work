"""
config.py — Resume Enhancer settings
======================================

LLM backend is selectable via env:
    RESUME_LLM_BACKEND=huggingface   (default; uses HF Inference API)
    RESUME_LLM_BACKEND=anthropic     (swap-in; uses Claude API)
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UI_DIR = PROJECT_ROOT / "ui"
WORK_DIR = PROJECT_ROOT / "data" / "jobs"


class Settings(BaseSettings):
    service_name: str = Field(default="resume-enhancer")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- LLM backend ---
    llm_backend: Literal["huggingface", "anthropic"] = Field(default="huggingface")

    # Hugging Face (default)
    hf_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")

    # Anthropic Claude (optional swap)
    anthropic_model: str = Field(default="claude-opus-4-7")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # --- LaTeX compilation ---
    pdflatex_cmd: str = Field(default="pdflatex")    # or "tectonic" if installed
    compile_timeout_seconds: int = Field(default=60, ge=5, le=300)

    # --- Upload limits ---
    max_upload_size_mb: int = Field(default=5, ge=1, le=20)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RESUME_",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()

"""
config.py — Resume Enhancer settings
======================================

LLM backend is selectable via env:
    RESUME_LLM_BACKEND=huggingface   (default; uses HF Inference API)
    RESUME_LLM_BACKEND=anthropic     (swap-in; uses Claude API)

All limits below are hard caps that prevent runaway loops or hung
requests. The app is designed to ALWAYS return a structured response
even if the LLM is unreachable or slow.
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

    # --- Hard limits to prevent runaway / infinite loop ---
    llm_call_timeout_seconds: int = Field(default=30, ge=5, le=120)
    overall_job_timeout_seconds: int = Field(default=180, ge=30, le=600)
    max_bullets_to_enhance: int = Field(default=30, ge=1, le=100)
    max_skills_buckets_to_enhance: int = Field(default=10, ge=1, le=20)

    # --- Agentic Enhancer→Critic loop ---
    # When enabled, every drafted section is graded by a Critic agent and
    # iterated up to `agent_max_iterations` times. Hard-bounded by the
    # overall job timeout above, so enabling cannot cause runaway.
    agent_critic_enabled: bool = Field(default=True)
    agent_max_iterations: int = Field(default=3, ge=1, le=5)
    agent_accept_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    agent_min_delta_to_continue: float = Field(default=3.0, ge=0.0, le=20.0)
    # Sections to which the critic loop applies (others use single-call path)
    agent_critique_sections: list[str] = Field(
        default_factory=lambda: ["summary", "bullet"]
    )

    # --- LaTeX compilation ---
    pdflatex_cmd: str = Field(default="pdflatex")
    compile_timeout_seconds: int = Field(default=60, ge=5, le=300)

    # --- Upload limits ---
    max_upload_size_mb: int = Field(default=5, ge=1, le=20)

    # --- Default target role ---
    default_role: str = Field(default="ai_ml_engineer")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RESUME_",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()

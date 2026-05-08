"""
config.py - environment-driven settings for the Resume Enhancer app.

Single source of truth. Every other module imports `settings` and never
touches os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "skills"
JD_DIR = PROJECT_ROOT / "data" / "jds"
TEMPLATE_DIR = APP_ROOT / "render"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
WORK_DIR = PROJECT_ROOT / ".work"


@dataclass(frozen=True)
class Settings:
    # Service
    service_name: str = field(default_factory=lambda: _env_str("RESUME_SERVICE_NAME", "resume-enhancer"))
    log_level: str = field(default_factory=lambda: _env_str("RESUME_LOG_LEVEL", "INFO"))

    # LLM backend - "auto" picks the best available backend at runtime.
    llm_backend: str = field(default_factory=lambda: _env_str("RESUME_LLM_BACKEND", "auto"))
    anthropic_api_key: str = field(default_factory=lambda: _env_str("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    hf_api_key: str = field(default_factory=lambda: _env_str("HF_API_KEY", ""))
    hf_model: str = field(default_factory=lambda: _env_str("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))

    # Iteration bounds
    max_iterations: int = field(default_factory=lambda: _env_int("RESUME_MAX_ITERATIONS", 3))
    accept_threshold: int = field(default_factory=lambda: _env_int("RESUME_ACCEPT_THRESHOLD", 82))
    min_delta_to_continue: int = field(default_factory=lambda: _env_int("RESUME_MIN_DELTA", 3))

    # Time budgets (seconds)
    section_budget_s: float = field(default_factory=lambda: _env_float("RESUME_SECTION_BUDGET_S", 60.0))
    job_budget_s: float = field(default_factory=lambda: _env_float("RESUME_JOB_BUDGET_S", 600.0))
    llm_call_timeout_s: float = field(default_factory=lambda: _env_float("RESUME_LLM_TIMEOUT_S", 90.0))

    # Upload limits
    max_upload_kb: int = field(default_factory=lambda: _env_int("RESUME_MAX_UPLOAD_KB", 512))

    # Default role
    default_role: str = field(default_factory=lambda: _env_str("RESUME_DEFAULT_ROLE", "ai_ml_engineer"))

    # Toggles
    critic_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_CRITIC_ENABLED", True))
    role_review_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_ROLE_REVIEW_ENABLED", True))
    jd_match_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_JD_MATCH_ENABLED", True))

    # JD validation roles (5 by default)
    cross_validate_roles: tuple[str, ...] = (
        "ai_ml_engineer",
        "data_scientist",
        "software_engineer",
        "devops_cloud_engineer",
        "product_manager",
    )


settings = Settings()


def workdir() -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR

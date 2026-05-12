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
    groq_api_key: str = field(default_factory=lambda: _env_str("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env_str("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))
    hf_api_key: str = field(default_factory=lambda: _env_str("HF_API_KEY", ""))
    hf_model: str = field(default_factory=lambda: _env_str("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    hf_extraction_model: str = field(default_factory=lambda: _env_str("HF_EXTRACTION_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    enable_multi_llm: bool = field(default_factory=lambda: _env_bool("RESUME_ENABLE_MULTI_LLM", True))

    # Iteration bounds
    # max_iterations: run up to N critique loops per section; hard-capped at 4.
    # accept_threshold: stop early when section score >= this (target: 95 = near-perfect).
    # min_delta_to_continue: stop iterating if improvement between rounds is tiny.
    max_iterations: int = field(default_factory=lambda: _env_int("RESUME_MAX_ITERATIONS", 4))
    accept_threshold: int = field(default_factory=lambda: _env_int("RESUME_ACCEPT_THRESHOLD", 95))
    min_delta_to_continue: int = field(default_factory=lambda: _env_int("RESUME_MIN_DELTA", 2))

    # Time budgets (seconds)
    section_budget_s: float = field(default_factory=lambda: _env_float("RESUME_SECTION_BUDGET_S", 60.0))
    job_budget_s: float = field(default_factory=lambda: _env_float("RESUME_JOB_BUDGET_S", 600.0))
    llm_call_timeout_s: float = field(default_factory=lambda: _env_float("RESUME_LLM_TIMEOUT_S", 90.0))

    # Upload limits
    max_upload_kb: int = field(default_factory=lambda: _env_int("RESUME_MAX_UPLOAD_KB", 512))

    # Rate limiting (per session, per hour)
    max_runs_per_hour: int = field(default_factory=lambda: _env_int("RESUME_MAX_RUNS_PER_HOUR", 5))

    # Work directory TTL (hours) — jobs older than this are cleaned up
    work_dir_ttl_hours: int = field(default_factory=lambda: _env_int("RESUME_WORK_TTL_HOURS", 24))

    # LLM retry settings
    llm_max_retries: int = field(default_factory=lambda: _env_int("RESUME_LLM_MAX_RETRIES", 3))
    llm_retry_base_delay: float = field(default_factory=lambda: _env_float("RESUME_LLM_RETRY_BASE_DELAY", 1.0))
    max_section_calls: int = field(default_factory=lambda: _env_int("RESUME_MAX_SECTION_CALLS", 120))

    # Default role
    default_role: str = field(default_factory=lambda: _env_str("RESUME_DEFAULT_ROLE", "ai_ml_engineer"))

    # Toggles
    critic_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_CRITIC_ENABLED", True))
    role_review_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_ROLE_REVIEW_ENABLED", True))
    jd_match_enabled: bool = field(default_factory=lambda: _env_bool("RESUME_JD_MATCH_ENABLED", True))

    # Optional basic auth (set both to enable)
    auth_user: str = field(default_factory=lambda: _env_str("RESUME_AUTH_USER", ""))
    auth_pass: str = field(default_factory=lambda: _env_str("RESUME_AUTH_PASS", ""))

    # JD validation roles (5 by default)
    cross_validate_roles: tuple[str, ...] = (
        "ai_ml_engineer",
        "data_scientist",
        "software_engineer",
        "devops_cloud_engineer",
        "product_manager",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_kb * 1024

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_user and self.auth_pass)


settings = Settings()


def workdir() -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR


# -- Dangerous LaTeX commands that should be rejected from uploads --
DANGEROUS_TEX_COMMANDS = [
    r"\write18",
    r"\immediate\write18",
    r"\input{/",
    r"\include{/",
    r"\openin",
    r"\openout",
    r"\catcode",
    r"\csname",
]

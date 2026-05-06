"""
config.py — Service configuration
==================================
Multi-Agent Research API — App layer
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"


class Settings(BaseSettings):
    service_name: str = Field(default="multi-agent-research")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # LLM
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=600, ge=1, le=4000)
    openai_api_key: str | None = Field(default=None)

    # Streaming
    sse_keepalive_seconds: int = Field(default=15)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CREW_",
        extra="ignore",
    )


settings = Settings()

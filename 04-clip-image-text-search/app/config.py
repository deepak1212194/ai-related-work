"""
config.py — CLIP service settings
==================================
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"


class Settings(BaseSettings):
    service_name: str = Field(default="clip-search")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Model
    model_name: str = Field(default="openai/clip-vit-base-patch32")

    # Storage
    images_dir: Path = Field(default=PROJECT_ROOT / "data" / "images")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")

    # Search
    default_top_k: int = Field(default=6, ge=1, le=50)
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CLIP_",
        extra="ignore",
    )


settings = Settings()

"""App configuration for Semantic Search & Classification Service."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    cors_origins: list = ["*"]
    top_k: int = int(os.environ.get("TOP_K", "5"))

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

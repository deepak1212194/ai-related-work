"""
config.py — Person tracker service settings
=============================================
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"


class Settings(BaseSettings):
    service_name: str = Field(default="edge-person-tracker")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Detector
    yolo_model: str = Field(default="yolov8n.pt")
    person_class_id: int = Field(default=0)
    detect_conf: float = Field(default=0.4, ge=0.0, le=1.0)

    # Tracker
    iou_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_missed_frames: int = Field(default=15, ge=1)

    # Streaming
    stream_every_n_frames: int = Field(default=1, ge=1)

    # Storage
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    max_upload_size_mb: int = Field(default=200, ge=1, le=2000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRACKER_",
        extra="ignore",
    )


settings = Settings()

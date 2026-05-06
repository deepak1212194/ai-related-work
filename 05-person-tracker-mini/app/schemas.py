"""
schemas.py — Person tracker contracts
=======================================
"""

from typing import Literal

from pydantic import BaseModel


class TrackBox(BaseModel):
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float


class FrameUpdate(BaseModel):
    """One frame's detections, sent over WebSocket."""

    frame: int
    timestamp_s: float
    n_tracks: int
    tracks: list[TrackBox]


class UploadResponse(BaseModel):
    status: Literal["ok"]
    video_id: str
    filename: str
    size_bytes: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    iou_threshold: float
    max_missed: int

"""
detector.py — YOLO Wrapper (persons only)
==========================================
Person Tracker Mini — Module 1

Thin wrapper around `ultralytics.YOLO` that returns only `person`
detections (COCO class 0) above a confidence threshold, in the simple
(x1, y1, x2, y2, conf) tuple form the tracker expects.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "yolov8n.pt"      # ultralytics auto-downloads on first run
PERSON_CLASS_ID = 0               # COCO 'person'
DEFAULT_CONF = 0.4


# ──────────────────────────────────────────────────────────────────────
# Detection record
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


# ──────────────────────────────────────────────────────────────────────
# Module 1: Detector class
# ──────────────────────────────────────────────────────────────────────
class PersonDetector:
    """Run a single forward pass and return person-class detections."""

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 conf: float = DEFAULT_CONF) -> None:
        print(f"[DETECT] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return person detections for `frame` (BGR uint8 numpy array)."""
        results = self.model(frame, conf=self.conf, verbose=False)
        if not results:
            return []

        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, c, p in zip(xyxy, cls, confs):
                if c != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box
                out.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    conf=float(p),
                ))
        return out

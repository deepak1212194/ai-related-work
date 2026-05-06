"""
tracker.py — IoU-Greedy Multi-Object Tracker
=============================================
Person Tracker Mini — Module 2

A small, self-contained tracker. Each frame:

  1. Compute IoU between every active track and every new detection.
  2. Greedily match high-IoU pairs (>= IOU_THRESHOLD) — a track gets at
     most one detection, and a detection gets at most one track.
  3. Unmatched detections start new tracks.
  4. Unmatched tracks have their `missed` counter incremented; tracks
     that exceed MAX_MISSED frames are retired.

This is teachable rather than state-of-the-art. For crowded scenes,
swap in OC-SORT or DeepSORT.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .detector import Detection

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.30
MAX_MISSED = 15           # frames a track can stay alive without a match


# ──────────────────────────────────────────────────────────────────────
# Track record
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    age: int = 0          # frames since track was created
    missed: int = 0       # consecutive frames without a match
    history: List[Tuple[float, float]] = field(default_factory=list)   # centroid trail


# ──────────────────────────────────────────────────────────────────────
# Module 1: IoU helper
# ──────────────────────────────────────────────────────────────────────
def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    """Standard intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _centroid(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ──────────────────────────────────────────────────────────────────────
# Module 2: Tracker
# ──────────────────────────────────────────────────────────────────────
class IouTracker:
    """Frame-by-frame greedy IoU matcher with track aging."""

    def __init__(self, iou_threshold: float = IOU_THRESHOLD,
                 max_missed: int = MAX_MISSED) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: List[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection]) -> List[Track]:
        """Match `detections` to existing tracks; return live tracks."""
        det_boxes = [d.as_xyxy() for d in detections]
        unmatched_dets = set(range(len(det_boxes)))
        unmatched_tracks = set(range(len(self.tracks)))

        # Build (track_idx, det_idx, iou) triples and sort descending
        candidates: List[Tuple[int, int, float]] = []
        for ti, t in enumerate(self.tracks):
            for di, db in enumerate(det_boxes):
                v = iou(t.bbox, db)
                if v >= self.iou_threshold:
                    candidates.append((ti, di, v))
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Greedy 1-to-1 assignment
        for ti, di, _ in candidates:
            if ti in unmatched_tracks and di in unmatched_dets:
                self.tracks[ti].bbox = det_boxes[di]
                self.tracks[ti].age += 1
                self.tracks[ti].missed = 0
                self.tracks[ti].history.append(_centroid(det_boxes[di]))
                unmatched_tracks.discard(ti)
                unmatched_dets.discard(di)

        # Unmatched tracks: bump missed; retire if over the limit
        for ti in list(unmatched_tracks):
            self.tracks[ti].missed += 1

        # Unmatched detections: start new tracks
        for di in unmatched_dets:
            new = Track(track_id=self._next_id, bbox=det_boxes[di])
            new.history.append(_centroid(det_boxes[di]))
            self.tracks.append(new)
            self._next_id += 1

        # Retire dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks

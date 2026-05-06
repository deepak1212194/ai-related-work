"""
main.py — Video Loop & Annotation
==================================
Person Tracker Mini — Module 3

Runs detection + tracking over an input video and writes an annotated
output MP4 with per-track bounding boxes and IDs.

Usage:
    python -m src.main --video data/sample.mp4 --out artifacts/annotated.mp4
"""

import argparse
from pathlib import Path

import cv2

from .detector import PersonDetector
from .tracker import IouTracker, Track

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BOX_COLOR = (0, 200, 0)     # green BGR
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────
# Module 1: Annotation
# ──────────────────────────────────────────────────────────────────────
def draw_tracks(frame, tracks: list[Track]):
    """Draw per-track bounding box + ID label on `frame` (in place)."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID {t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────────────
# Module 2: Run
# ──────────────────────────────────────────────────────────────────────
def run(video_path: Path, out_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {video_path.name}  {w}x{h} @ {fps:.1f}fps  ({n_frames} frames)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    detector = PersonDetector()
    tracker = IouTracker()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        dets = detector.detect(frame)
        tracks = tracker.update(dets)
        annotated = draw_tracks(frame, tracks)

        writer.write(annotated)
        if frame_idx % 25 == 0:
            print(f"[RUN] frame {frame_idx}/{n_frames}  tracks={len(tracks)}")

    cap.release()
    writer.release()
    print(f"[SAVE] Annotated video → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + IoU person tracker.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/annotated.mp4"))
    args = parser.parse_args()

    run(args.video, args.out)
    print("[DONE] Tracking complete.")


if __name__ == "__main__":
    main()

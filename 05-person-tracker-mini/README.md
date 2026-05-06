# 05 · Person Tracker Mini

A **minimal real-time person tracker** that combines **YOLO detection** with a **hand-rolled IoU-based tracker** to assign stable IDs across video frames.

```
   frame  →  YOLO (persons only)  →  IoU matcher  →  tracked IDs
                    │                       │
                    ▼                       ▼
              detections           tracks (id, bbox, age, missed)
```

## Why this design

A simple IoU matcher is enough for many real-world surveillance / retail scenarios where people walk slowly across a frame. Going to SORT / DeepSORT / OC-SORT adds Kalman filtering and appearance embeddings, which help in crowded scenes but add dependencies and tuning surface. Keeping the tracker hand-rolled here makes the code teachable and the failure modes obvious.

## Quick start

```bash
cd 05-person-tracker-mini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Drop a video into ./data/sample.mp4 (or pass --video PATH)
python -m src.main --video data/sample.mp4 --out artifacts/annotated.mp4
```

## Project layout

```
05-person-tracker-mini/
├── src/
│   ├── detector.py    # Module 1: YOLO wrapper, person-only detections
│   ├── tracker.py     # Module 2: IoU-greedy track-management
│   └── main.py        # Module 3: video loop, draw, write
├── data/              # drop your own video here (gitignored)
└── requirements.txt
```

## Notes

- Default detector is **YOLOv8n** via `ultralytics` — the smallest model, runs on CPU.
- The tracker keeps a track alive for a configurable number of missed frames (`MAX_MISSED`) before retiring it.
- Output is a side-by-side annotated MP4 with track IDs printed on each box.
- This is **generic CV code** — no proprietary tracking logic, no employer-specific dwell-time rules. Use it as a starting point for any real-time multi-object-tracking project.

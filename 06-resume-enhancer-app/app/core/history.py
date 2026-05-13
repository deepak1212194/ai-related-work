"""
history.py — Lightweight per-session run history and keyword memory.

Persists to PROJECT_ROOT/.work/history.json so it survives restarts.
Used to:
  1. Show the last N run summaries in the UI sidebar.
  2. Track which keywords were consistently missing across runs so the
     user gets a "persistent gaps" signal on their next run.
  3. Track which keywords were successfully introduced so they don't
     get flagged again.

No PII is stored — only role, scores, keyword lists, and elapsed time.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import workdir

log = logging.getLogger(__name__)

_HISTORY_FILE_NAME = "history.json"
_MAX_RUNS = 20       # keep last N runs
_lock = threading.Lock()


@dataclass
class RunRecord:
    run_id: str
    timestamp: float
    role: str
    ats_score: float
    hm_score: float
    jd_avg_after: float
    jd_avg_delta: float
    sections_changed: int
    sections_total: int
    elapsed_s: float
    missing_keywords: List[str] = field(default_factory=list)
    presentation_gaps: List[str] = field(default_factory=list)
    real_gaps: List[str] = field(default_factory=list)
    manual_action_count: int = 0
    custom_jd_used: bool = False


def _history_path() -> Path:
    return workdir() / _HISTORY_FILE_NAME


def _load_raw() -> dict:
    p = _history_path()
    if not p.exists():
        return {"runs": [], "persistent_gaps": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return {"runs": [], "persistent_gaps": {}}


def _save_raw(data: dict) -> None:
    try:
        _history_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:                                   # noqa: BLE001
        log.warning("[history] could not save history: %s", e)


def save_run(record: RunRecord) -> None:
    """Persist a run record and update the persistent-gaps tracker."""
    with _lock:
        data = _load_raw()
        runs: list = data.get("runs", [])
        runs.insert(0, {
            "run_id": record.run_id,
            "timestamp": record.timestamp,
            "role": record.role,
            "ats_score": record.ats_score,
            "hm_score": record.hm_score,
            "jd_avg_after": record.jd_avg_after,
            "jd_avg_delta": record.jd_avg_delta,
            "sections_changed": record.sections_changed,
            "sections_total": record.sections_total,
            "elapsed_s": record.elapsed_s,
            "missing_keywords": record.missing_keywords,
            "presentation_gaps": record.presentation_gaps,
            "real_gaps": record.real_gaps,
            "manual_action_count": record.manual_action_count,
            "custom_jd_used": record.custom_jd_used,
        })
        data["runs"] = runs[:_MAX_RUNS]

        # Update persistent gap counter
        pg: Dict[str, int] = data.get("persistent_gaps", {})
        for kw in record.missing_keywords:
            pg[kw] = pg.get(kw, 0) + 1
        # Decay: remove keywords that appeared in fewer than 2 runs and
        # aren't in the current missing set (they were resolved).
        current_missing = set(record.missing_keywords)
        pg = {
            k: v for k, v in pg.items()
            if v >= 2 or k in current_missing
        }
        data["persistent_gaps"] = pg
        _save_raw(data)


def load_runs() -> List[dict]:
    """Return the last N run summaries (newest first)."""
    with _lock:
        return _load_raw().get("runs", [])


def get_persistent_gaps(top_n: int = 10) -> List[str]:
    """Return keywords that have been missing across 2+ recent runs."""
    with _lock:
        pg = _load_raw().get("persistent_gaps", {})
    return [k for k, v in sorted(pg.items(), key=lambda x: -x[1]) if v >= 2][:top_n]


def clear_history() -> None:
    with _lock:
        _save_raw({"runs": [], "persistent_gaps": {}})

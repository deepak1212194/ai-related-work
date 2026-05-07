"""
dwell.py — Dwell-Time Analytics Engine
========================================
Person Tracker Mini — Module 3

ROI-based zone monitoring with dwell-time tracking:
  1. Define rectangular zones (regions of interest)
  2. Track how long each person stays within each zone
  3. Raise alerts when dwell time exceeds threshold
  4. Compute zone occupancy counts

This is inspired by production dwell-time systems used in retail
analytics and smart building monitoring. The production version
integrates with IoT Hub for real-time alerting and uses DeepStream
for GPU-accelerated video analytics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_DWELL_THRESHOLD = 30.0    # seconds before flagging as dwelling
DEFAULT_ALERT_INTERVAL = 10.0    # seconds between alert emissions
STALE_TRACK_FRAMES = 75          # frames before removing inactive track IDs


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Zone:
    """A rectangular region of interest."""
    zone_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    def contains_point(self, cx: float, cy: float) -> bool:
        """Check if a point (centroid) is inside this zone."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


@dataclass
class DwellRecord:
    """Dwell-time state for a single track within a zone."""
    track_id: int
    zone_id: str
    entry_time: float
    last_seen_time: float
    last_seen_frame: int
    dwell_seconds: float = 0.0
    is_dwelling: bool = False


@dataclass
class DwellAlert:
    """Alert raised when a person dwells beyond the threshold."""
    track_id: int
    zone_id: str
    dwell_seconds: float
    timestamp: float


@dataclass
class ZoneStats:
    """Aggregated statistics for a zone."""
    zone_id: str
    current_occupancy: int = 0
    total_entries: int = 0
    active_dwellers: int = 0
    avg_dwell_seconds: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Dwell Time Engine
# ──────────────────────────────────────────────────────────────────────
class DwellTimeEngine:
    """
    Tracks per-person dwell time within configurable zones.

    Usage:
        engine = DwellTimeEngine(threshold=30.0)
        engine.add_zone(Zone("entrance", 0, 0, 200, 400))

        # Each frame:
        alerts = engine.update(tracks, frame_num)
    """

    def __init__(
        self,
        dwell_threshold: float = DEFAULT_DWELL_THRESHOLD,
        alert_interval: float = DEFAULT_ALERT_INTERVAL,
    ):
        self.dwell_threshold = dwell_threshold
        self.alert_interval = alert_interval

        self.zones: Dict[str, Zone] = {}
        self.records: Dict[Tuple[int, str], DwellRecord] = {}  # (track_id, zone_id)
        self.last_alert_time: Optional[float] = None
        self.alert_history: List[DwellAlert] = []

        # Heatmap accumulator: pixel counts
        self.heatmap_counts: Dict[Tuple[int, int], int] = {}

        # Track re-ID heuristic: map new IDs to old ones when track count is stable
        self._prev_track_ids: List[int] = []
        self._prev_count: int = 0

    def add_zone(self, zone: Zone) -> None:
        """Register a new monitoring zone."""
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone and its associated records."""
        self.zones.pop(zone_id, None)
        to_remove = [k for k in self.records if k[1] == zone_id]
        for k in to_remove:
            del self.records[k]

    def update(self, tracks, frame_num: int) -> List[DwellAlert]:
        """
        Update dwell state for all tracks across all zones.

        Args:
            tracks: list of Track objects (from IouTracker.update)
            frame_num: current frame number

        Returns:
            List of alerts raised this frame (empty if within cooldown)
        """
        current_time = time.time()
        alerts: List[DwellAlert] = []

        # Apply re-ID heuristic (handle tracker ID switches)
        current_ids = [t.track_id for t in tracks]
        self._apply_reid_heuristic(current_ids, current_time, frame_num)
        self._prev_track_ids = current_ids
        self._prev_count = len(current_ids)

        # Update dwell time for each track in each zone
        for track in tracks:
            cx, cy = self._centroid(track.bbox)

            # Update heatmap
            grid_x, grid_y = int(cx) // 10, int(cy) // 10  # 10px grid
            self.heatmap_counts[(grid_x, grid_y)] = self.heatmap_counts.get((grid_x, grid_y), 0) + 1

            for zone_id, zone in self.zones.items():
                key = (track.track_id, zone_id)
                inside = zone.contains_point(cx, cy)

                if inside:
                    if key not in self.records:
                        # New entry into zone
                        self.records[key] = DwellRecord(
                            track_id=track.track_id,
                            zone_id=zone_id,
                            entry_time=current_time,
                            last_seen_time=current_time,
                            last_seen_frame=frame_num,
                        )
                    else:
                        record = self.records[key]
                        record.last_seen_time = current_time
                        record.last_seen_frame = frame_num
                        record.dwell_seconds = current_time - record.entry_time

                        # Check dwell threshold
                        is_dwelling = self._check_dwelling(record, tracks)
                        record.is_dwelling = is_dwelling

                        if is_dwelling:
                            alerts.append(DwellAlert(
                                track_id=track.track_id,
                                zone_id=zone_id,
                                dwell_seconds=record.dwell_seconds,
                                timestamp=current_time,
                            ))

        # Apply alert cooldown
        if alerts and self.last_alert_time is not None:
            if (current_time - self.last_alert_time) < self.alert_interval:
                alerts = []  # suppress during cooldown

        if alerts:
            self.last_alert_time = current_time
            self.alert_history.extend(alerts)

        # Periodic cleanup of stale records
        if frame_num % STALE_TRACK_FRAMES == 0:
            self._cleanup_stale(frame_num)

        return alerts

    def _check_dwelling(self, record: DwellRecord, tracks) -> bool:
        """
        Determine if a person is dwelling, using two heuristics:

        1. Primary: dwell time exceeds threshold
        2. Secondary: dwell time >= 90% of threshold AND zone is crowded (>3 people)
           (crowded zones are more likely to cause genuine dwelling even with shorter times)
        """
        if record.dwell_seconds > self.dwell_threshold:
            return True

        # Secondary heuristic: near-threshold in crowded zone
        if record.dwell_seconds > self.dwell_threshold * 0.9 and len(tracks) > 3:
            return True

        return False

    def _apply_reid_heuristic(self, current_ids: List[int], current_time: float, frame_num: int):
        """
        Handle tracker ID switches when the same number of detections
        produce different IDs (common with IoU trackers during occlusions).

        If detection count is stable but IDs changed, try to transfer
        dwell records from lost IDs to new IDs.
        """
        if len(current_ids) != self._prev_count or self._prev_count == 0:
            return

        lost = set(self._prev_track_ids) - set(current_ids)
        gained = set(current_ids) - set(self._prev_track_ids)

        if not lost or len(lost) != len(gained):
            return

        lost_sorted = sorted(lost)
        gained_sorted = sorted(gained)

        for old_id, new_id in zip(lost_sorted, gained_sorted):
            if new_id > old_id:
                # Transfer dwell records
                for zone_id in self.zones:
                    old_key = (old_id, zone_id)
                    new_key = (new_id, zone_id)
                    if old_key in self.records and new_key not in self.records:
                        self.records[new_key] = self.records.pop(old_key)
                        self.records[new_key].track_id = new_id

    def get_zone_stats(self) -> List[ZoneStats]:
        """Get current statistics for all zones."""
        stats = []
        for zone_id in self.zones:
            records = [r for k, r in self.records.items() if k[1] == zone_id]
            active = [r for r in records if not self._is_stale(r)]
            dwellers = [r for r in active if r.is_dwelling]
            avg_dwell = (
                sum(r.dwell_seconds for r in active) / len(active)
                if active else 0.0
            )
            stats.append(ZoneStats(
                zone_id=zone_id,
                current_occupancy=len(active),
                total_entries=len(records),
                active_dwellers=len(dwellers),
                avg_dwell_seconds=round(avg_dwell, 1),
            ))
        return stats

    def get_heatmap_data(self, grid_size: int = 10) -> List[Dict]:
        """Return heatmap data as a list of {x, y, count} dicts."""
        return [
            {"x": gx * grid_size, "y": gy * grid_size, "count": cnt}
            for (gx, gy), cnt in self.heatmap_counts.items()
        ]

    def get_dwelling_tracks(self) -> List[DwellRecord]:
        """Get all currently dwelling tracks."""
        return [r for r in self.records.values() if r.is_dwelling]

    def _is_stale(self, record: DwellRecord) -> bool:
        """Check if a record is stale (track not seen recently)."""
        return (time.time() - record.last_seen_time) > 5.0  # 5 second timeout

    def _cleanup_stale(self, frame_num: int) -> None:
        """Remove records for tracks no longer active."""
        stale_keys = [
            k for k, r in self.records.items()
            if (frame_num - r.last_seen_frame) > STALE_TRACK_FRAMES
        ]
        for k in stale_keys:
            del self.records[k]

    @staticmethod
    def _centroid(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

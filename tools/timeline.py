"""
Timeline builder — groups raw detections into event windows.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class EventWindow:
    start_sec: float
    end_sec: float
    peak_sec: float
    event_type: str = "unknown"
    highlight_score: int = 0
    description: str = ""
    keyframe_path: str = ""
    confidence: str = "low"

    def to_dict(self) -> dict:
        # Off-ball movement events need more pre-roll so the run developing is visible
        OFF_BALL_TYPES = {
            "run_creating_space", "pressing_trigger", "diagonal_run",
            "recovery_run", "late_run_box", "building_angle", "scanning_buildup",
        }
        pre_roll  = 6.0 if self.event_type in OFF_BALL_TYPES else 3.0
        post_roll = 3.0 if self.event_type in OFF_BALL_TYPES else 2.0
        return {
            "event_id":       str(uuid.uuid4())[:8],
            "timestamp_sec":  round(self.peak_sec, 2),
            "duration_sec":   round(self.end_sec - self.start_sec, 2),
            "clip_start_sec": max(0, round(self.start_sec - pre_roll, 2)),
            "clip_end_sec":   round(self.end_sec + post_roll, 2),
            "event_type":     self.event_type,
            "highlight_score": self.highlight_score,
            "description":    self.description,
            "keyframe_path":  self.keyframe_path,
            "player_visible": True,
            "confidence":     self.confidence,
        }


def build_event_windows(
    active_timestamps: list[float],
    gap_threshold_sec: float = 3.0,
    min_window_sec: float = 1.5,
) -> list[EventWindow]:
    """
    Groups timestamps where the target player is active into event windows.
    Timestamps closer than gap_threshold_sec are merged into one window.
    """
    if not active_timestamps:
        return []

    sorted_ts = sorted(active_timestamps)
    windows   = []
    start     = sorted_ts[0]
    prev      = sorted_ts[0]

    for ts in sorted_ts[1:]:
        if ts - prev > gap_threshold_sec:
            if prev - start >= min_window_sec:
                windows.append(EventWindow(
                    start_sec=start,
                    end_sec=prev,
                    peak_sec=(start + prev) / 2,
                ))
            start = ts
        prev = ts

    if prev - start >= min_window_sec:
        windows.append(EventWindow(
            start_sec=start,
            end_sec=prev,
            peak_sec=(start + prev) / 2,
        ))

    return windows


def rank_events(events: list[dict], top_n: int = 10) -> list[dict]:
    return sorted(events, key=lambda e: e["highlight_score"], reverse=True)[:top_n]

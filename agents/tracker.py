"""
Player Tracker — DeepSORT multi-object tracking across frames.
Assigns consistent track IDs to players so we can follow them
across a 90-minute game without re-identifying from scratch each frame.
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


_tracker = None

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = DeepSort(
            max_age=30,           # frames before a lost track is dropped
            n_init=3,             # frames needed to confirm a new track
            nms_max_overlap=0.7,
            embedder="mobilenet", # lightweight re-ID embedder
        )
    return _tracker


def update_tracks(frame: np.ndarray, detections: list[dict]) -> list[dict]:
    """
    Feed current frame detections into DeepSORT.
    Returns active tracks: [{track_id, bbox, confidence}]
    """
    tracker = _get_tracker()

    # DeepSORT expects [[x1,y1,w,h], confidence, class]
    raw = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        raw.append(([x1, y1, w, h], d["confidence"], "person"))

    tracks = tracker.update_tracks(raw, frame=frame)

    results = []
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = t.to_ltrb()
        results.append({
            "track_id":   t.track_id,
            "bbox":       (x1, y1, x2, y2),
            "confidence": t.get_det_conf() or 0.5,
        })

    return results

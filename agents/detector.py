"""
Player Detector — YOLOv8 person detection on video frames.
Returns bounding boxes for all detected players per frame.
"""

import numpy as np
from ultralytics import YOLO

# Load once at import time — use nano model for speed, swap to yolov8s for accuracy
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # auto-downloads on first run
    return _model


def detect_players(frame: np.ndarray, conf_threshold: float = 0.4) -> list[dict]:
    """
    Run YOLO on a single frame, return list of player detections.
    Each detection: {bbox: (x1,y1,x2,y2), confidence: float}
    Only returns class 0 (person).
    """
    model   = _get_model()
    results = model(frame, classes=[0], conf=conf_threshold, verbose=False)[0]
    players = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        players.append({
            "bbox":       (x1, y1, x2, y2),
            "confidence": conf,
        })

    return players

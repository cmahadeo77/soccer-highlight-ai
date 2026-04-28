"""
Player + Ball Detector — YOLOv8 detection on video frames.
Detects players (class 0) and sports ball (class 32) in one pass.
Ball proximity to target player is used to prioritize possession moments.
"""

import numpy as np
from ultralytics import YOLO

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8s.pt")
    return _model


def detect_players(frame: np.ndarray, conf_threshold: float = 0.3) -> list[dict]:
    """Detect all players in frame. Returns [{bbox, confidence}]"""
    model   = _get_model()
    results = model(frame, classes=[0], conf=conf_threshold, verbose=False)[0]
    players = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        players.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
    return players


def detect_ball(frame: np.ndarray, conf_threshold: float = 0.25) -> list[tuple]:
    """
    Detect sports ball (COCO class 32) in frame.
    Returns list of (cx, cy) center points for each detected ball.
    Lower confidence threshold since balls are small in wide-angle footage.
    """
    model   = _get_model()
    results = model(frame, classes=[32], conf=conf_threshold, verbose=False)[0]
    balls   = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        balls.append((cx, cy))
    return balls


def player_near_ball(player_bbox: tuple, ball_centers: list[tuple], radius: float = 120) -> bool:
    """
    Returns True if any detected ball center is within `radius` pixels
    of the player bounding box center. 120px covers a natural play radius
    at Veo's wide-angle zoom level.
    """
    if not ball_centers:
        return False
    x1, y1, x2, y2 = player_bbox
    px = (x1 + x2) / 2
    py = (y1 + y2) / 2
    for bx, by in ball_centers:
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        if dist <= radius:
            return True
    return False

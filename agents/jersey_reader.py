"""
Jersey Reader — OCR jersey numbers from player bounding boxes.
Uses EasyOCR with upscaling for small Veo wide-angle footage.
"""

import numpy as np
import easyocr
from tools.video import crop_bounding_box, upscale

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def read_jersey(frame: np.ndarray, bbox: tuple) -> str | None:
    """
    Crop player bbox, upscale, run OCR, return jersey number string or None.
    Focuses on the upper-torso region where numbers appear.
    """
    reader = _get_reader()

    x1, y1, x2, y2 = bbox
    height = y2 - y1

    # Jersey numbers appear on upper 60% of the body (torso)
    torso_bbox = (x1, y1, x2, y1 + height * 0.6)
    crop = crop_bounding_box(frame, torso_bbox, padding=5)

    if crop.size == 0:
        return None

    crop_up = upscale(crop, scale=3.0)
    results  = reader.readtext(crop_up, allowlist="0123456789", min_size=10)

    for (_, text, confidence) in results:
        text = text.strip()
        if text.isdigit() and confidence > 0.4:
            return text

    return None


def find_target_player(
    frame: np.ndarray,
    tracks: list[dict],
    target_jersey: str,
    jersey_cache: dict,
) -> dict | None:
    """
    Given current tracks, find the one matching target_jersey.
    jersey_cache maps track_id -> confirmed jersey number to avoid re-reading every frame.
    Returns the matching track dict or None.
    """
    for track in tracks:
        tid = track["track_id"]

        # Use cached assignment if we've already confirmed this track's jersey
        if tid in jersey_cache:
            if jersey_cache[tid] == target_jersey:
                return track
            continue

        jersey = read_jersey(frame, track["bbox"])
        if jersey is not None:
            jersey_cache[tid] = jersey
            if jersey == target_jersey:
                return track

    return None

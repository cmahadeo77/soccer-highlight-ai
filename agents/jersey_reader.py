"""
Jersey Reader — Claude Vision jersey number identification.
Replaces EasyOCR: sends player crops to Claude Haiku to read jersey numbers,
which works on Veo's wide-angle aerial footage where EasyOCR fails.
"""

import anthropic
import base64
import cv2
import numpy as np
import os
from tools.video import crop_bounding_box

_client = None
_attempt_log: dict[int, int] = {}  # track_id -> frame_idx of last read attempt
RETRY_INTERVAL = 60  # re-attempt unconfirmed tracks every 60 sampled frames (~30s at 2fps)


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def read_jersey(frame: np.ndarray, bbox: tuple) -> str | None:
    """Send a player crop to Claude Haiku to read the jersey number."""
    crop = crop_bounding_box(frame, bbox, padding=8)
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 5 or w < 3:
        return None

    # Upscale to at least 100px tall so Claude can read tiny numbers from wide shots
    if h < 100:
        scale = 100 / h
        crop = cv2.resize(crop, (max(1, int(w * scale)), 100), interpolation=cv2.INTER_CUBIC)

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": "What jersey number is on this soccer player? Reply with ONLY the number (e.g. '11'), or 'none' if not visible.",
                    },
                ],
            }],
        )
        text = response.content[0].text.strip()
        digits = "".join(c for c in text if c.isdigit())
        return digits if digits else None
    except Exception:
        return None


def find_target_player(
    frame: np.ndarray,
    tracks: list[dict],
    target_jersey: str,
    jersey_cache: dict,
    frame_idx: int = 0,
    force_read: bool = False,
) -> dict | None:
    """
    Find the track matching target_jersey.

    jersey_cache: track_id -> confirmed jersey string.
    Unconfirmed tracks are retried every RETRY_INTERVAL frames.

    force_read=True: skip cache and re-read all tracks from scratch.
    Use at Veo moment anchor points to re-identify the player after a track drop.
    """
    for track in tracks:
        tid = track["track_id"]

        if not force_read and tid in jersey_cache:
            if jersey_cache[tid] == target_jersey:
                return track
            continue

        if not force_read:
            last_attempt = _attempt_log.get(tid, -RETRY_INTERVAL)
            if frame_idx - last_attempt < RETRY_INTERVAL:
                continue

        _attempt_log[tid] = frame_idx
        jersey = read_jersey(frame, track["bbox"])
        if jersey is not None:
            jersey_cache[tid] = jersey
            if jersey == target_jersey:
                return track

    return None

"""
Play Classifier — sends keyframes to Claude Vision to identify
event type, highlight value, and plain-English description.
"""

import anthropic
import base64
import cv2
import numpy as np
import os
from tools.video import crop_bounding_box

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


SYSTEM_PROMPT = """You are a soccer recruiting analyst reviewing highlight clips.
You will receive a keyframe image from a youth soccer game. A specific player is marked with a red bounding box.

Classify the play and respond with JSON only — no markdown, no extra text:
{
  "event_type": one of: skill_move | key_pass | defensive_play | recovery_run | 50_50_ball | shot | tackle | interception | clearance | sprint | other,
  "highlight_score": integer 1-10 (10 = must-include in recruiting reel),
  "description": "one sentence plain-English description of the play",
  "confidence": "high" | "medium" | "low"
}

Scoring guide:
- 9-10: Exceptional — goal, goal-saving tackle, elite skill move, threading pass through tight space
- 7-8: Strong — clean tackle, smart recovery, incisive pass, winning 50/50
- 5-6: Solid — competent defensive positioning, standard pass, basic skill move
- 3-4: Marginal — out of position but recovers, routine play
- 1-2: Not highlight-worthy"""


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _draw_target_box(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """Draw red bounding box around target player for Claude context."""
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return annotated


def classify_event(frame: np.ndarray, player_bbox: tuple) -> dict:
    """
    Send an annotated keyframe to Claude Vision and get event classification.
    Returns the parsed event dict.
    """
    import json

    annotated  = _draw_target_box(frame, player_bbox)
    img_b64    = _encode_frame(annotated)

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Classify the play involving the player in the red box.",
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "event_type":      "other",
            "highlight_score": 3,
            "description":     raw[:200],
            "confidence":      "low",
        }


def generate_recruiting_summary(events: list[dict], jersey: str) -> str:
    """
    Generate a recruiting paragraph summarizing the player's performance
    based on the top detected highlights.
    """
    if not events:
        return ""

    top = sorted(events, key=lambda e: e["highlight_score"], reverse=True)[:8]
    descriptions = "\n".join(f"- {e['event_type']}: {e['description']}" for e in top)

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"""Based on these detected highlights for player #{jersey}, write a concise 3-4 sentence recruiting summary suitable for a tryout highlight reel. Focus on technical ability, soccer IQ, and work rate. Be specific.

Highlights:
{descriptions}

Write the summary paragraph only, no headers.""",
            }
        ],
    )

    return response.content[0].text.strip()

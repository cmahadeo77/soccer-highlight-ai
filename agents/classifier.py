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


SYSTEM_PROMPT = """You are a soccer recruiting analyst evaluating a box-to-box midfielder for ECNL (top club level) recruitment.
The player currently competes at ECNL Regional League and is being evaluated for an ECNL club roster.
You will receive a keyframe from a youth soccer game. The target player is marked with a red bounding box.

ECNL scouts evaluating a box-to-box midfielder prioritize: soccer IQ, scanning and awareness, technical quality
under pressure, ability to progress the ball, defensive work rate, and the capacity to impact both halves of the field.

Classify the play and respond with JSON only — no markdown, no extra text:
{
  "event_type": one of: scanning_buildup | progressive_pass | switch_of_play | recovery_run | winning_50_50 | beating_defender | defensive_press | interception | tackle | late_run_box | combination_play | shot | clearance | other,
  "highlight_score": integer 1-10 (10 = must-include in ECNL recruiting reel),
  "description": "one sentence description focused on the decision, technique, and soccer IQ shown",
  "confidence": "high" | "medium" | "low"
}

Scoring guide — weighted for box-to-box midfielder at ECNL level:

9-10 MUST INCLUDE:
  - Scans before receiving and immediately plays through a defensive line under pressure
  - Switches the field with a driven cross-field pass to relieve pressure or exploit space
  - Wins a contested 50/50 in central midfield and immediately plays forward
  - Beats a defender 1v1 in midfield with a purposeful carry or skill move
  - Late run into the box arriving at the right moment to finish or support
  - Recovery run at full pace to win the ball back or prevent a breakaway
  - High press that directly wins the ball in the opponent's half
  - Interception that reads the game and immediately launches a counter

7-8 STRONG:
  - Receives under pressure, opens body, plays a clean progressive pass forward
  - Drives through midfield with the ball, breaking lines with her run
  - Wins a physical duel in midfield and keeps possession
  - Sharp one-two combination that advances play through pressure
  - Defensive block or tackle that shows positional intelligence
  - Drops into space intelligently to offer a buildout option under pressure

5-6 SOLID — include if reel needs variety:
  - Clean simple pass that maintains possession in a tight situation
  - Good defensive shape — cuts off a passing lane or delays a counter
  - Receives and turns away from pressure competently
  - Tracks a runner and stays goalside

3-4 MARGINAL — do not include:
  - Routine pass with no pressure or decision required
  - Jogging or standing still
  - Out of position with no recovery effort

1-2 EXCLUDE:
  - Uncontested ball movement
  - Dead ball situations
  - Player not clearly involved in active play"""


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
                "content": f"""Based on these detected highlights for player #{jersey}, write a concise 3-4 sentence recruiting summary for an ECNL club coach evaluating a box-to-box midfielder moving up from ECNL Regional League.

Focus specifically on: scanning and awareness before receiving, ability to progress the ball through lines, defensive work rate and recovery, composure under pressure, and box-to-box impact. Use language a club coach or DOC would use when evaluating a central midfielder. Be specific about what the clips show — do not use generic phrases.

Highlights detected:
{descriptions}

Write the summary paragraph only, no headers.""",
            }
        ],
    )

    return response.content[0].text.strip()

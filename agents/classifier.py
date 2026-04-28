"""
Play Classifier — soccer analyst evaluation of player moment sequences.
Sends a 3-frame sequence per moment to Claude for deep read, not single-snapshot classification.
"""

import anthropic
import base64
import cv2
import json
import numpy as np
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


ANALYST_SYSTEM = """You are an experienced soccer analyst evaluating a box-to-box midfielder for ECNL club recruitment.
The player (jersey #11) currently plays ECNL Regional League and is being evaluated for a full ECNL roster spot.

You will receive a sequence of 3 frames from a play — before, during, and after the key moment.
The target player is marked with a red bounding box in each frame.

Read the sequence the way an analyst watches film: look at what the player does BEFORE the ball arrives
(scanning, positioning, body shape), what decision she makes WITH the ball or off the ball, and what
the outcome is for the team. A single good pass is not enough — what matters is the read, the timing,
and the pressure context.

Respond with JSON only — no markdown, no extra text:
{
  "event_type": one of: scanning_buildup | progressive_pass | switch_of_play | recovery_run | winning_50_50 | beating_defender | defensive_press | interception | tackle | late_run_box | combination_play | shot | clearance | other,
  "highlight_score": integer 1-10,
  "description": "2 sentences max — what she did and why it shows ECNL-level quality. Be specific about the decision and pressure context.",
  "analyst_note": "1 sentence a DOC would say when reviewing film — e.g. 'Plays through the press without looking — that composure is what separates her.'",
  "confidence": "high" | "medium" | "low"
}

SCORING — weighted for box-to-box midfielder at ECNL level:

9-10 MUST INCLUDE in reel:
  - Scans before receiving, immediately plays through a defensive line under pressure
  - Switches field with driven cross-field pass to exploit space or relieve pressure
  - Wins contested 50/50 in central midfield and immediately plays forward
  - Beats a defender 1v1 in midfield with purposeful carry or skill move
  - Late run into the box arriving at the right moment to finish or support
  - Recovery run at full pace to deny a breakaway or win the ball back
  - High press that directly wins the ball in opponent's half
  - Interception that reads the game and immediately launches counter

7-8 STRONG — include:
  - Receives under pressure, opens body, plays clean progressive pass forward
  - Drives through midfield breaking lines with her run
  - Wins physical duel in midfield and keeps possession
  - Sharp one-two combination advancing play through pressure
  - Defensive block or tackle showing positional intelligence
  - Drops into space intelligently to offer buildout option under pressure

5-6 SOLID — include only if reel needs variety:
  - Clean simple pass maintaining possession in tight situation
  - Good defensive shape cutting off lane or delaying counter
  - Receives and turns away from pressure competently
  - Tracks runner and stays goalside

3-4 MARGINAL — omit:
  - Routine pass with no pressure or decision required
  - Jogging or standing, not actively involved
  - Out of position with no recovery effort

1-2 EXCLUDE:
  - Uncontested ball movement
  - Dead ball situations
  - Wrong player or player not visible/involved"""


def _encode(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _annotate(frame: np.ndarray, bbox: tuple, label: str = "") -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if label:
        cv2.putText(out, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def classify_event(
    frame: np.ndarray,
    player_bbox: tuple,
    has_ball: bool = False,
    sequence: list[tuple[np.ndarray, tuple]] | None = None,
) -> dict:
    """
    Evaluate a play moment using a 3-frame sequence when available.
    sequence: list of (frame, bbox) tuples for before/peak/after frames.
    Falls back to single frame if sequence is None.
    """
    # Build the image content blocks
    image_blocks = []

    frames_to_send = sequence if sequence else [(frame, player_bbox)]
    labels = ["BEFORE", "DURING", "AFTER"] if len(frames_to_send) == 3 else [""]

    for (f, bbox), label in zip(frames_to_send, labels):
        annotated = _annotate(f, bbox, label)
        image_blocks.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       _encode(annotated),
            },
        })

    seq_desc = f"{len(frames_to_send)}-frame sequence" if len(frames_to_send) > 1 else "single frame"
    ball_note = (
        "Ball tracking confirms the player is near the ball — likely a possession, passing, or receiving moment."
        if has_ball else
        "Player is not detected near the ball — likely movement, positioning, pressing, or recovery."
    )

    image_blocks.append({
        "type": "text",
        "text": (
            f"Analyze this {seq_desc} and evaluate the play involving the player in the red box.\n"
            f"{ball_note}\n"
            "Read the decision-making across the full sequence, not just the peak frame."
        ),
    })

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=400,
        system=ANALYST_SYSTEM,
        messages=[{"role": "user", "content": image_blocks}],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "event_type":      "other",
            "highlight_score": 3,
            "description":     raw[:200],
            "analyst_note":    "",
            "confidence":      "low",
        }


def generate_recruiting_summary(events: list[dict], jersey: str) -> str:
    if not events:
        return ""

    top = sorted(events, key=lambda e: e["highlight_score"], reverse=True)[:10]
    lines = []
    for e in top:
        note = e.get("analyst_note", "")
        lines.append(f"- [{e['event_type']} | {e['highlight_score']}/10] {e['description']}" + (f" — {note}" if note else ""))
    descriptions = "\n".join(lines)

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=350,
        messages=[{
            "role": "user",
            "content": f"""You are writing a recruiting evaluation for player #{jersey} — a box-to-box midfielder being evaluated for an ECNL club roster (moving up from ECNL Regional League).

Based on the film analysis below, write a 4-5 sentence evaluation that a club Director of Coaching would read before watching the highlight reel. Focus on: scanning and awareness before the ball, ability to progress through lines, defensive work rate and recovery speed, composure under pressure, and box-to-box impact. Be specific — reference what the film actually shows. Use the language a DOC uses when writing a recruiting note, not generic adjectives.

Film analysis:
{descriptions}

Write the evaluation only, no headers.""",
        }],
    )

    return response.content[0].text.strip()

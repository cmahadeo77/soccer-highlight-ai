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

You will receive a sequence of 3 frames — before, during, and after the key moment.
The target player is marked with a red bounding box in each frame.

CRITICAL: Off-ball movement is as important as on-ball skill at this level. Watch for:
- Runs that create space for teammates BEFORE the ball arrives
- Pressing triggers that force turnovers without direct contact
- Diagonal runs into the half-space that stretch the defensive shape
- Recovery runs that show engine and defensive discipline
- Building angles — dropping or checking away to offer a passing line
- Third-man runs — timing a run knowing two passes ahead

Read the full sequence like a coach watching film: what is she doing 3-6 seconds before the peak
moment? Is she making a run, scanning, pressing, or positioning? The WHY matters more than the WHAT.

Respond with JSON only — no markdown, no extra text:
{
  "event_type": one of: scanning_buildup | progressive_pass | switch_of_play | recovery_run | winning_50_50 | beating_defender | defensive_press | interception | tackle | late_run_box | combination_play | run_creating_space | pressing_trigger | diagonal_run | building_angle | shot | clearance | other,
  "highlight_score": integer 1-10,
  "description": "2 sentences max — what she did and why it shows ECNL-level quality. Name the specific movement or decision, the context/pressure, and the outcome for the team.",
  "analyst_note": "1 sentence a DOC would write in a recruiting note — e.g. 'Third-man run at full pace before the ball is played — she sees the game two moves ahead.'",
  "confidence": "high" | "medium" | "low"
}

SCORING GUIDE — weighted for box-to-box midfielder at ECNL level:

9-10 MUST INCLUDE — shows ECNL-level spatial intelligence or work rate:
  ON-BALL:
  - Scans before receiving, immediately plays through a defensive line under pressure
  - Switches field with driven cross-field pass to exploit space
  - Wins contested 50/50 in central midfield, immediately plays forward
  - Beats a defender 1v1 with purposeful carry or skill move
  - Shot or finish from late box run
  - Interception that reads the game and launches a counter
  OFF-BALL:
  - Diagonal run into the half-space that pulls a defender and opens a lane for a teammate
  - Third-man run timed perfectly — she's already moving before the second pass is played
  - Full-pace recovery run that denies a counter or wins the ball back in transition
  - Press trigger — she commits first, her pressure forces the turnover, teammates react to her
  - Late run into the box arriving at the right moment to finish or create a chance
  - Drops into a tight pocket of space at the right moment to give a clean exit line under pressure

7-8 STRONG — include:
  ON-BALL:
  - Receives under pressure, opens body, plays clean progressive pass forward
  - Drives through midfield breaking lines with her carry
  - Sharp one-two combination advancing play through pressure
  - Defensive block or tackle showing positional intelligence
  OFF-BALL:
  - Checks away and offers a building angle that moves the ball forward
  - Tracks a runner goalside for an extended defensive sequence
  - Presses intelligently, forcing a back-pass or long ball that resets possession
  - Shows for the ball in a tight moment, offering a release valve under pressure

5-6 SOLID — include only if reel needs variety:
  - Clean simple pass maintaining possession in a tight situation
  - Receives and turns away from pressure competently
  - Good defensive shape cutting off a lane
  - Standard run that doesn't pull defenders but keeps shape

3-4 MARGINAL — omit:
  - Jogging or standing still without a clear purpose
  - Routine pass with no pressure or decision required
  - Out of position with no recovery effort shown

1-2 EXCLUDE:
  - Dead ball situations
  - Player not visibly involved in the play
  - Wrong player in focus"""


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
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""You are writing a recruiting evaluation for player #{jersey} — a box-to-box midfielder being evaluated for an ECNL club roster (moving up from ECNL Regional League).

Based on the film analysis below, write a 4-5 sentence evaluation that a club Director of Coaching would read before watching the highlight reel.

Address ALL of the following if the film supports it:
- Off-ball movement: does she create space, make runs before the ball, trigger the press?
- Spatial intelligence: does she see the game two moves ahead — third-man runs, diagonal movement, building angles?
- On-ball quality: composure under pressure, ability to play through lines, technical execution
- Defensive work rate: recovery pace, tracking runners, winning duels
- Box-to-box impact: does she affect both halves of the field?

Be specific — reference what the film actually shows. Use language a DOC uses in a recruiting note, not generic adjectives. If the film shows strong off-ball movement, lead with that.

Film analysis:
{descriptions}

Write the evaluation only, no headers.""",
        }],
    )

    return response.content[0].text.strip()

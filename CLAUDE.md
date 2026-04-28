# Soccer Highlight AI — Project Instructions

## What this is
AI recruiting highlight reel generator for **Karina Mahadeo, jersey #11**, box-to-box midfielder.
Currently playing ECNL Regional League, being evaluated for a full ECNL club roster spot.

## Player
- Name: Karina Mahadeo
- Jersey: #11
- Veo player_id: `68ab5b2a4fd3128d45d26c15`
- Team: Mustang 2013G ECNL RL

## Games (all downloaded to `examples/`)
- `mustang_vs_davis_apr25.mp4`
- `mustang_april25.mp4`
- `mustang_apr12.mp4`
- `mustang_vs_fury.mp4`
- `mustang_vs_placer_oct12.mp4`
- `mustang_sep06.mp4`

## Standard run command (single game)
```bash
python run_highlight.py \
  --video examples/mustang_vs_davis_apr25.mp4 \
  --jersey 11 \
  --ball-in-play moments_apr25_ball_in_play.json \
  --moments moments_apr25.json
```
Output goes to `output/jersey_11_YYYYMMDD_HHMMSS/`

## Veo metadata files (Apr 25 game, already captured)
- `moments_apr25.json` — 17 Veo anchor moments
- `moments_apr25_ball_in_play.json` — 86 live play windows (45.5 min of 83 min)

To capture for a different game:
```bash
python capture_veo_moments.py \
  --url "https://app.veo.co/matches/MATCH-SLUG/" \
  --jersey 11 \
  --output moments_LABEL.json \
  --debug
```

## What Claude should evaluate (non-negotiable)
Off-ball movement is as important as on-ball skill. Always look for:
- Runs creating space before the ball arrives
- Press triggers — she commits first, teammates react
- Diagonal runs into the half-space
- Third-man runs (she's moving before the second pass is played)
- Building angles — checking away to offer a clean exit line
- Recovery runs at full pace

Clips must include pre-run context. Off-ball event types get 6s pre-roll so the scout sees the movement developing, not just the endpoint.

## Recruiting lens
ECNL club DOC language. Evaluate: scanning/awareness, progression through lines, defensive work rate, composure under pressure, box-to-box impact. Lead with off-ball spatial intelligence if film supports it.

## Key architecture decisions
- **Jersey reading**: Claude Haiku vision (replaced EasyOCR — players too small in Veo aerial shots)
- **Detection**: yolov8s, conf=0.3
- **Classification**: Claude Opus, 3-frame sequence (before/peak/after)
- **Frame skipping**: ball_in_play windows skip dead ball frames (~45% faster)
- **Veo tracking**: `has_tracking_data: false` on all matches — no player-specific moments from Veo. Their highlights API only covers goals/shots. All passes, 50/50s, off-ball work must come from our own scan.

## Known issues / next steps
- Run all 5 games and review output
- Build batch script to process all games overnight
- Consider adding a viewer/review UI to browse clips before assembling reel

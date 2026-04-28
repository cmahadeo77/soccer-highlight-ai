"""
Soccer Highlight AI — Main Runner
Usage: python run_highlight.py --video path/to/game.mp4 --jersey 10
"""

import argparse
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from tools.video       import extract_frames, save_keyframe, get_video_info
from tools.timeline    import build_event_windows, rank_events
from agents.detector   import detect_players, detect_ball, player_near_ball
from agents.tracker    import update_tracks
from agents.jersey_reader import find_target_player
from agents.classifier import classify_event, generate_recruiting_summary
from agents.assembler  import extract_clips, assemble_reel, save_brief


def parse_args():
    p = argparse.ArgumentParser(description="Generate soccer recruiting highlights by jersey number")
    p.add_argument("--video",   required=True,  help="Path to Veo MP4 game file")
    p.add_argument("--jersey",  required=True,  help="Target player jersey number (e.g. 10)")
    p.add_argument("--output",  default="output", help="Output directory")
    p.add_argument("--fps",     type=int, default=2, help="Frames per second to sample (default: 2)")
    p.add_argument("--top",     type=int, default=10, help="Number of top clips to include in reel")
    p.add_argument("--no-reel", action="store_true", help="Skip final reel assembly, just output clips + brief")
    p.add_argument("--moments", default=None,
                   help="Path to Veo moments JSON (from capture_veo_moments.py). "
                        "Used as anchor timestamps to re-identify the player at known points; "
                        "the full game is still scanned independently.")
    return p.parse_args()


def main():
    args   = parse_args()
    jersey = str(args.jersey).strip()
    output = os.path.join(args.output, f"jersey_{jersey}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output, exist_ok=True)

    print(f"\n=== Soccer Highlight AI ===")
    print(f"Video:   {args.video}")
    print(f"Jersey:  #{jersey}")
    print(f"Output:  {output}\n")

    # ── 1. Video info ────────────────────────────────────────────────────────
    info = get_video_info(args.video)
    print(f"[Info] {info['duration_sec']/60:.1f} min | {info['fps']:.1f} fps | {info['width']}x{info['height']}")

    # ── 1b. Load Veo moments anchors (optional) ───────────────────────────────
    veo_anchor_timestamps = set()
    veo_moments_count = 0
    if args.moments:
        try:
            with open(args.moments) as f:
                veo_moments = json.load(f)
            # Collect timestamps ± 2s as a re-identification window
            for m in veo_moments:
                center = m.get("start_sec", 0)
                for offset in range(-2, 10):   # cover the moment window
                    veo_anchor_timestamps.add(round(center + offset, 1))
            veo_moments_count = len(veo_moments)
            print(f"[Moments] Loaded {veo_moments_count} Veo anchor moments — "
                  f"will force re-identification at those timestamps.\n"
                  f"  Full game will still be scanned independently for moments Veo missed.")
        except Exception as e:
            print(f"[Moments] Could not load {args.moments}: {e} — continuing without anchors.")

    # ── 2. Extract frames + run detection + tracking ─────────────────────────
    print(f"\n[Step 1/4] Scanning full game at {args.fps} fps...")
    total_sample_frames = int(info["duration_sec"] * args.fps)
    jersey_cache  = {}   # track_id -> confirmed jersey number
    active_ts     = []   # timestamps where target player is visible
    active_data   = {}   # timestamp -> {frame, bbox} for keyframe saving

    for i, (frame_idx, ts, frame) in enumerate(extract_frames(args.video, sample_fps=args.fps)):
        if i % 50 == 0:
            pct = (i / total_sample_frames) * 100 if total_sample_frames > 0 else 0
            print(f"  {pct:.0f}% — {ts/60:.1f} min scanned", end="\r", flush=True)

        detections  = detect_players(frame)
        tracks      = update_tracks(frame, detections)

        # At Veo anchor timestamps, force re-identification in case the track was lost
        at_anchor = round(ts, 1) in veo_anchor_timestamps
        target    = find_target_player(frame, tracks, jersey, jersey_cache,
                                       frame_idx=i, force_read=at_anchor)

        if target:
            balls       = detect_ball(frame)
            has_ball    = player_near_ball(target["bbox"], balls)
            active_ts.append(ts)
            active_data[ts] = {
                "frame":    frame,
                "bbox":     target["bbox"],
                "has_ball": has_ball,   # True = possession/passing moment
            }

    print(f"\n  Found #{jersey} in {len(active_ts)} sampled frames")

    # ── 3. Group into event windows ──────────────────────────────────────────
    print(f"\n[Step 2/4] Building event windows...")
    windows = build_event_windows(active_ts, gap_threshold_sec=4.0)
    print(f"  {len(windows)} candidate event windows detected")

    # ── 4. Classify each window via Claude Vision ────────────────────────────
    print(f"\n[Step 3/4] Classifying events with Claude Vision...")
    kf_dir = os.path.join(output, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    events = []

    all_ts = sorted(active_data.keys())

    for i, window in enumerate(windows):
        # Prefer a frame where she's near the ball; fall back to peak timestamp
        window_frames = {t: d for t, d in active_data.items()
                         if window.start_sec <= t <= window.end_sec}
        ball_frames   = {t: d for t, d in window_frames.items() if d.get("has_ball")}

        if ball_frames:
            peak_ts = min(ball_frames.keys(), key=lambda t: abs(t - window.peak_sec))
        elif window_frames:
            peak_ts = min(window_frames.keys(), key=lambda t: abs(t - window.peak_sec))
        else:
            peak_ts = min(active_data.keys(), key=lambda t: abs(t - window.peak_sec))

        data     = active_data[peak_ts]
        frame    = data["frame"]
        bbox     = data["bbox"]
        has_ball = data.get("has_ball", False)

        kf_path  = os.path.join(kf_dir, f"kf_{i:04d}_{int(peak_ts)}s.jpg")
        save_keyframe(frame, kf_path)

        # Build 3-frame sequence: before / peak / after
        # Find the nearest active frames ~3s before and ~3s after peak
        def nearest_ts(target_t):
            candidates = [t for t in all_ts if t != peak_ts]
            if not candidates:
                return peak_ts
            return min(candidates, key=lambda t: abs(t - target_t))

        before_ts = nearest_ts(peak_ts - 3.0)
        after_ts  = nearest_ts(peak_ts + 3.0)

        sequence = [
            (active_data[before_ts]["frame"], active_data[before_ts]["bbox"]),
            (frame, bbox),
            (active_data[after_ts]["frame"],  active_data[after_ts]["bbox"]),
        ]

        try:
            result = classify_event(frame, bbox, has_ball=has_ball, sequence=sequence)
        except Exception as e:
            print(f"  [Classify] Error on window {i}: {e}")
            result = {"event_type": "other", "highlight_score": 3, "description": "Classification failed", "analyst_note": "", "confidence": "low"}

        window.event_type      = result.get("event_type", "other")
        window.highlight_score = result.get("highlight_score", 3)
        window.description     = result.get("description", "")
        window.confidence      = result.get("confidence", "low")
        window.keyframe_path   = kf_path

        event = window.to_dict()
        events.append(event)

        score = event['highlight_score']
        print(f"  [{i+1}/{len(windows)}] {event['event_type']:20s} score={score}/10  {event['description'][:60]}")

    # ── 5. Extract clips + assemble reel ─────────────────────────────────────
    print(f"\n[Step 4/4] Extracting top {args.top} clips...")
    clips = extract_clips(args.video, events, output, top_n=args.top)

    reel_path = None
    if not args.no_reel and clips:
        reel_path = os.path.join(output, f"highlight_reel_jersey_{jersey}.mp4")
        print(f"\n[Assembling] Concatenating {len(clips)} clips into reel...")
        try:
            assemble_reel(clips, reel_path)
            print(f"  Reel saved: {reel_path}")
        except Exception as e:
            print(f"  [Assembler] Reel assembly failed (clips still saved): {e}")
            reel_path = None

    # ── 6. Generate recruiting summary + save brief ───────────────────────────
    print(f"\n[Summary] Generating recruiting notes...")
    try:
        recruiting_notes = generate_recruiting_summary(events, jersey)
        print(f"\n--- Recruiting Summary ---\n{recruiting_notes}\n")
    except Exception as e:
        print(f"  [Summary] Failed to generate notes: {e}")
        recruiting_notes = ""

    # Event type counts
    from collections import Counter
    type_counts = Counter(e["event_type"] for e in events)

    brief = {
        "player_jersey":     jersey,
        "video_source":      args.video,
        "processed_at":      datetime.utcnow().isoformat() + "Z",
        "total_events":      len(events),
        "reel_duration_sec": sum((e["clip_end_sec"] - e["clip_start_sec"]) for e in rank_events(events, top_n=args.top)),
        "events":            events,
        "summary":           dict(type_counts),
        "top_clips":         clips,
        "reel_path":         reel_path,
        "recruiting_notes":  recruiting_notes,
    }

    brief_path = save_brief(brief, output)
    print(f"[Done] Brief saved: {brief_path}")
    print(f"[Done] {len(clips)} clips extracted | Reel: {reel_path or 'skipped'}")
    print(f"\nOpen {output}/ to review clips and reel.\n")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    main()

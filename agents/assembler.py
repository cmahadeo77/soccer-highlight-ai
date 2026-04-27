"""
Assembler — cuts clips from the source video based on the event timeline,
then concatenates them into a highlight reel.
"""

import os
import json
import subprocess
from pathlib import Path
from tools.video import cut_clip, FFMPEG
from tools.timeline import rank_events


def extract_clips(video_path: str, events: list[dict], output_dir: str, top_n: int = 10) -> list[str]:
    """
    Cut individual clips for the top N events.
    Returns list of clip file paths.
    """
    top     = rank_events(events, top_n=top_n)
    clips   = []
    clip_dir = os.path.join(output_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    for i, event in enumerate(top):
        out_path = os.path.join(clip_dir, f"clip_{i+1:02d}_{event['event_type']}_{event['event_id']}.mp4")
        try:
            cut_clip(
                video_path,
                start_sec=event["clip_start_sec"],
                end_sec=event["clip_end_sec"],
                output_path=out_path,
            )
            print(f"  [Assembler] Clip {i+1}: {event['event_type']} @ {event['timestamp_sec']:.1f}s — score {event['highlight_score']}/10")
            clips.append(out_path)
        except RuntimeError as e:
            print(f"  [Assembler] Failed to cut clip {i+1}: {e}")

    return clips


def assemble_reel(clips: list[str], output_path: str) -> str:
    """
    Concatenate clips into a single highlight reel using ffmpeg concat.
    Clips are ordered by highlight_score (already ranked by extract_clips).
    """
    if not clips:
        raise ValueError("No clips to assemble.")

    concat_list = output_path.replace(".mp4", "_list.txt")
    with open(concat_list, "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    cmd = [
        FFMPEG, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    os.remove(concat_list)
    return output_path


def save_brief(brief: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "reel_brief.json")
    with open(path, "w") as f:
        json.dump(brief, f, indent=2)
    return path

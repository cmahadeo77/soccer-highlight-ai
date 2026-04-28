"""
Video tools — frame extraction, clip cutting, keyframe export.
Wraps ffmpeg and OpenCV.
"""

import os
import subprocess
import cv2
import numpy as np
from pathlib import Path

FFMPEG = os.environ.get(
    "FFMPEG_PATH",
    r"C:\Users\cmaha\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = total / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "total_frames": total, "width": width, "height": height, "duration_sec": duration}


def extract_frames(video_path: str, sample_fps: int = 2):
    """
    Yields (frame_index, timestamp_sec, frame) at sample_fps rate.
    Generator — never loads more than one frame into memory at a time.
    """
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / sample_fps))
    idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / fps
            yield (idx, ts, frame)
        idx += 1

    cap.release()


def save_keyframe(frame: np.ndarray, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def cut_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> str:
    """Extract a clip using ffmpeg. Fast keyframe-seeking, no re-encode."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = end_sec - start_sec
    cmd = [
        FFMPEG, "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip cut failed: {result.stderr}")
    return output_path


def crop_bounding_box(frame: np.ndarray, bbox: tuple, padding: int = 10) -> np.ndarray:
    """Crop and return a player bounding box from a frame, with padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    return frame[y1:y2, x1:x2]


def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale image for better OCR on small jersey numbers."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

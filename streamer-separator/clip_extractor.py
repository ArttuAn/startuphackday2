"""
clip_extractor.py — Extract frames and audio for a single clip window.

Output layout for clip_001:
  <out_dir>/
    clip_001/
      gameplay/
        frame_0001.jpg   (full frame at FRAMES_FPS)
        frame_0002.jpg
        ...
      facecam/
        frame_0001.jpg   (cropped bbox region)
        frame_0002.jpg
        ...
      audio.wav           (16 kHz mono PCM)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2

from config import AUDIO_RATE, FRAMES_FPS
from clip_cutter import ClipWindow
from detector import BBox

log = logging.getLogger("clip_extractor")


def _ffmpeg_extract_audio(
    src: Path,
    t_start: float,
    duration: float,
    out_path: Path,
) -> bool:
    """Slice t_start..t_start+duration from src (video or wav) → out_path wav."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(t_start),
        "-i", str(src),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_RATE),
        "-ac", "1",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as exc:
        log.error("Audio extraction failed: %s", exc)
        return False


def extract_clip(
    video_path:  Path,
    bbox:        BBox,
    window:      ClipWindow,
    clip_dir:    Path,
    voice_path:  Path | None = None,   # Block 3 output: separated voice wav
) -> dict:
    """
    Extract gameplay frames, facecam frames, mixed audio, and voice slice.

    Returns a dict with paths (relative to clip_dir.parent):
      {
        "gameplay_frames": "clip_001/gameplay",
        "facecam_frames":  "clip_001/facecam",
        "audio":           "clip_001/audio.wav",   # mixed (from video)
        "voice":           "clip_001/voice.wav",   # streamer voice only (if available)
      }
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    gp_dir = clip_dir / "gameplay"
    fc_dir = clip_dir / "facecam"
    gp_dir.mkdir(exist_ok=True)
    fc_dir.mkdir(exist_ok=True)
    audio_path = clip_dir / "audio.wav"

    cap     = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step    = max(1, int(src_fps / FRAMES_FPS))

    start_frame = int(window.t_start * src_fps)
    end_frame   = int(window.t_end   * src_fps)

    frame_idx  = start_frame
    saved_idx  = 1

    log.info("Extracting %s  (%.1fs – %.1fs)", clip_dir.name, window.t_start, window.t_end)

    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        name = f"frame_{saved_idx:04d}.jpg"

        # Full frame → gameplay
        cv2.imwrite(str(gp_dir / name), frame)

        # Cropped bbox → facecam
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size > 0:
            cv2.imwrite(str(fc_dir / name), face_crop)

        frame_idx += step
        saved_idx += 1

    cap.release()
    log.debug("  → %d frames saved", saved_idx - 1)

    # Extract mixed audio segment (from raw video)
    _ffmpeg_extract_audio(
        video_path,
        t_start=window.t_start,
        duration=window.duration,
        out_path=audio_path,
    )

    # Slice separated voice wav if Block 3 produced one
    voice_clip_path = None
    if voice_path and voice_path.exists():
        voice_clip_path = clip_dir / "voice.wav"
        _ffmpeg_extract_audio(
            voice_path,
            t_start=window.t_start,
            duration=window.duration,
            out_path=voice_clip_path,
        )

    rel = clip_dir.name
    return {
        "gameplay_frames": f"{rel}/gameplay",
        "facecam_frames":  f"{rel}/facecam",
        "audio":           f"{rel}/audio.wav",
        "voice":           f"{rel}/voice.wav" if voice_clip_path else "",
    }

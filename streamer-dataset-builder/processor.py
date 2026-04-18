"""
processor.py — ffmpeg-based audio extraction and frame sampling.

For each downloaded video:
  A. Extract 16 kHz mono WAV  → data/raw_videos/<id>/audio.wav
  B. Sample 1 frame/sec JPEGs → data/frames/<id>/*.jpg
  C. (Optional) Facecam crop  → data/facecam/<id>.mp4  (requires OpenCV)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import (
    AUDIO_SAMPLE_RATE,
    FACECAM_DIR,
    FRAMES_DIR,
    FRAMES_PER_SECOND,
    RAW_VIDEOS_DIR,
)
from logger_setup import get_logger

log = get_logger("processor")


# ── Helpers ────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command; return True on success."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log.error("%s failed:\n%s", label, exc.stderr.decode(errors="replace")[:500])
        return False


# ── A. Audio extraction ────────────────────────────────────────────────────

def extract_audio(vid_id: str) -> Path | None:
    """
    Extract audio from raw.mp4 → audio.wav (16 kHz, mono, PCM s16le).

    Returns path to .wav on success, None on failure.
    """
    video_path = RAW_VIDEOS_DIR / vid_id / "raw.mp4"
    audio_path = RAW_VIDEOS_DIR / vid_id / "audio.wav"

    if not video_path.exists():
        log.warning("Missing raw video for %s", vid_id)
        return None

    if audio_path.exists():
        log.debug("Audio already extracted for %s", vid_id)
        return audio_path

    log.info("Extracting audio: %s", vid_id)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        str(audio_path),
    ]
    if _run(cmd, f"audio-extract/{vid_id}"):
        return audio_path
    return None


# ── B. Frame sampling ─────────────────────────────────────────────────────

def extract_frames(vid_id: str) -> Path | None:
    """
    Sample frames at FRAMES_PER_SECOND from raw.mp4.
    Stores JPEGs in data/frames/<vid_id>/.

    Returns frame directory on success, None on failure.
    """
    video_path  = RAW_VIDEOS_DIR / vid_id / "raw.mp4"
    frames_dir  = FRAMES_DIR / vid_id

    if not video_path.exists():
        log.warning("Missing raw video for %s", vid_id)
        return None

    if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
        log.debug("Frames already extracted for %s", vid_id)
        return frames_dir

    frames_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting frames (%d fps): %s", FRAMES_PER_SECOND, vid_id)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={FRAMES_PER_SECOND}",
        "-q:v", "5",                         # JPEG quality (1=best, 31=worst)
        str(frames_dir / "frame_%06d.jpg"),
    ]
    if _run(cmd, f"frames/{vid_id}"):
        count = len(list(frames_dir.glob("*.jpg")))
        log.info("  → %d frames saved", count)
        return frames_dir
    return None


# ── C. (Optional) Facecam detection & crop ────────────────────────────────

def extract_facecam(vid_id: str) -> Path | None:
    """
    Very simple facecam crop heuristic using OpenCV Haar cascades.
    Finds the most common face bounding box across sampled frames,
    then ffmpeg-crops the video to that region.

    Requires: opencv-python to be installed.
    Returns path to cropped mp4, or None if skipped/failed.
    """
    try:
        import cv2
    except ImportError:
        log.debug("opencv-python not installed; skipping facecam extraction for %s", vid_id)
        return None

    frames_dir = FRAMES_DIR / vid_id
    if not frames_dir.exists():
        log.warning("No frames for %s; run extract_frames first", vid_id)
        return None

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    boxes: list[tuple[int, int, int, int]] = []
    for img_path in sorted(frames_dir.glob("*.jpg"))[:300]:  # sample up to 300 frames
        img  = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        log.info("No faces detected in %s; skipping facecam crop", vid_id)
        return None

    # Median bounding box across all detections
    xs  = sorted(b[0] for b in boxes)
    ys  = sorted(b[1] for b in boxes)
    xs2 = sorted(b[2] for b in boxes)
    ys2 = sorted(b[3] for b in boxes)
    mid = len(boxes) // 2
    x, y, x2, y2 = xs[mid], ys[mid], xs2[mid], ys2[mid]
    w, h = x2 - x, y2 - y

    if w < 50 or h < 50:
        log.info("Face region too small for %s; skipping", vid_id)
        return None

    out_path   = FACECAM_DIR / f"{vid_id}.mp4"
    video_path = RAW_VIDEOS_DIR / vid_id / "raw.mp4"

    log.info("Cropping facecam for %s: x=%d y=%d w=%d h=%d", vid_id, x, y, w, h)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:a", "copy",
        str(out_path),
    ]
    if _run(cmd, f"facecam/{vid_id}"):
        return out_path
    return None


# ── Process one video ─────────────────────────────────────────────────────

def process_video(vid_id: str, do_frames: bool = True, do_facecam: bool = False) -> dict:
    """
    Run the full processing pipeline for one video.

    Returns a dict summarising what was produced.
    """
    result = {
        "vid_id":       vid_id,
        "audio_path":   None,
        "frames_path":  None,
        "facecam_path": None,
    }

    audio = extract_audio(vid_id)
    result["audio_path"] = str(audio) if audio else None

    if do_frames:
        frames = extract_frames(vid_id)
        result["frames_path"] = str(frames) if frames else None

    if do_facecam:
        facecam = extract_facecam(vid_id)
        result["facecam_path"] = str(facecam) if facecam else None

    return result

"""
validate.py — Quality-filter scraped videos before further processing.

Takes SNAPSHOTS evenly-spaced frames from each raw.mp4 and applies five checks:

  1. Face presence  — Haar cascade must detect a face in MIN_FACE_HITS frames
  2. Face size      — face must be at least MIN_FACE_PX wide
  3. Sharpness      — Laplacian variance of face crop >= MIN_SHARPNESS
  4. Brightness     — mean V channel (HSV) of face crop >= MIN_BRIGHTNESS
                      catches streamers sitting in the dark
  5. Real-face tex. — texture score of face crop >= MIN_TEXTURE_SCORE
                      catches VTubers / animated avatars:
                      real skin has noisy micro-texture; anime fills are flat

Videos that fail any check are deleted from RAW_VIDEOS_DIR so later pipeline
steps never see them.

Config knobs are at the top — tune if you're getting too many false rejects.

Usage:
    python validate.py
"""

import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
RAW_VIDEOS_DIR = Path(r"C:\Users\Admin\OneDrive\Desktop\startuphackYT\streamer-dataset-builder\data\raw_videos")

SNAPSHOTS        = 10    # frames to sample per video
MIN_FACE_HITS    = 7     # raised from 5 — need face in 7/10 frames (was 5)
MIN_SHARPNESS    = 30.0  # Laplacian variance of face crop
MIN_FACE_PX      = 20    # minimum face width (native pixels)
MIN_BRIGHTNESS   = 40    # mean HSV-V of face crop, 0-255 — rejects dark rooms
MIN_TEXTURE_SCORE = 12.0 # std-dev of normalised face crop — rejects flat/anime faces

DETECT_MIN_WIDTH = 640   # upscale frames to this width before detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Per-frame helpers ───────────────────────────────────────────────────────

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(face_crop: np.ndarray) -> float:
    """Mean V channel (HSV) of face crop, 0–255. Low = dark room."""
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def texture_score(face_crop: np.ndarray) -> float:
    """
    Std-dev of the face crop after CLAHE normalisation.

    Real skin has micro-texture (pores, shadow, hair) → high std-dev even
    after equalising contrast.  Anime / VTuber faces are flat fills → low
    std-dev even after equalisation.

    Typical values:
        real face, good light  → 25–60
        real face, dim light   → 15–35
        anime / flat avatar    → 3–14
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    return float(eq.std())


def detect_face(frame: np.ndarray):
    """
    Run Haar cascade (with upscale) and return the largest face crop + metrics.

    Returns:
        (found: bool, sharp: float, face_width_px: int,
         bright: float, tex: float)
    """
    h, w = frame.shape[:2]
    scale    = max(1.0, DETECT_MIN_WIDTH / w)
    upscaled = cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_LINEAR) if scale > 1 else frame

    gray  = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=3, minSize=(20, 20))

    if len(faces) == 0:
        return False, sharpness(upscaled), 0, 0.0, 0.0

    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    face_crop  = upscaled[fy:fy + fh, fx:fx + fw]
    native_fw  = int(fw / scale)

    if face_crop.size == 0:
        return True, 0.0, native_fw, 0.0, 0.0

    return (
        True,
        sharpness(face_crop),
        native_fw,
        brightness(face_crop),
        texture_score(face_crop),
    )


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


# ── Per-video validation ───────────────────────────────────────────────────

def validate_video(vid_dir: Path) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str)."""
    raw = vid_dir / "raw.mp4"
    if not raw.exists():
        return False, "raw.mp4 not found"

    frames = sample_frames(raw, SNAPSHOTS)
    if not frames:
        return False, "could not read frames"

    hits              = 0
    sharpness_scores  = []
    face_widths       = []
    brightness_scores = []
    texture_scores    = []

    for frame in frames:
        found, sharp, fw, bright, tex = detect_face(frame)
        if found:
            hits += 1
            sharpness_scores.append(sharp)
            face_widths.append(fw)
            brightness_scores.append(bright)
            texture_scores.append(tex)

    # ── Check 1: face presence ─────────────────────────────────────────────
    if hits < MIN_FACE_HITS:
        return False, (
            f"no consistent face — detected in only {hits}/{SNAPSHOTS} frames "
            f"(need {MIN_FACE_HITS})"
        )

    avg_sharp   = float(np.mean(sharpness_scores))
    avg_fw      = float(np.mean(face_widths))
    avg_bright  = float(np.mean(brightness_scores))
    avg_tex     = float(np.mean(texture_scores))

    # ── Check 2: face size ─────────────────────────────────────────────────
    if avg_fw < MIN_FACE_PX:
        return False, f"face too small ({avg_fw:.0f}px < {MIN_FACE_PX}px)"

    # ── Check 3: sharpness ────────────────────────────────────────────────
    if avg_sharp < MIN_SHARPNESS:
        return False, f"too blurry (sharpness={avg_sharp:.1f} < {MIN_SHARPNESS})"

    # ── Check 4: brightness — catches dark-room streamers ─────────────────
    if avg_bright < MIN_BRIGHTNESS:
        return False, (
            f"room too dark (brightness={avg_bright:.1f} < {MIN_BRIGHTNESS}) "
            f"— emotion detection will fail"
        )

    # ── Check 5: texture — catches VTubers / animated avatars ─────────────
    if avg_tex < MIN_TEXTURE_SCORE:
        return False, (
            f"face looks animated / VTuber (texture={avg_tex:.1f} < {MIN_TEXTURE_SCORE}) "
            f"— not a real person"
        )

    return True, (
        f"ok — face {hits}/{SNAPSHOTS} frames, "
        f"sharp={avg_sharp:.0f}, size={avg_fw:.0f}px, "
        f"bright={avg_bright:.0f}, tex={avg_tex:.1f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    vid_dirs = sorted(d for d in RAW_VIDEOS_DIR.iterdir()
                      if d.is_dir() and (d / "raw.mp4").exists())

    if not vid_dirs:
        log.error("No raw.mp4 files found in %s", RAW_VIDEOS_DIR)
        sys.exit(1)

    log.info("Validating %d videos (%d snapshots each)...", len(vid_dirs), SNAPSHOTS)
    log.info("Thresholds: face_hits>=%d, sharp>=%.0f, size>=%dpx, "
             "bright>=%d, texture>=%.1f",
             MIN_FACE_HITS, MIN_SHARPNESS, MIN_FACE_PX,
             MIN_BRIGHTNESS, MIN_TEXTURE_SCORE)

    passed  = []
    removed = []

    for i, vid_dir in enumerate(vid_dirs, 1):
        vid_id = vid_dir.name
        log.info("[%d/%d] %s", i, len(vid_dirs), vid_id)

        ok, reason = validate_video(vid_dir)

        if ok:
            log.info("  ✓ %s", reason)
            passed.append(vid_id)
        else:
            log.warning("  ✗ %s — deleting", reason)
            try:
                shutil.rmtree(vid_dir)
                removed.append(vid_id)
            except PermissionError as e:
                log.warning("  could not delete %s (file locked by Windows/OneDrive): %s", vid_id, e)
                log.warning("  skipping deletion — video will be filtered out in later steps")
                removed.append(vid_id)  # still mark as rejected so it's not assembled

    log.info("")
    log.info("Results: %d passed, %d removed", len(passed), len(removed))
    if removed:
        log.info("Removed: %s", ", ".join(removed))


if __name__ == "__main__":
    main()

"""
extract_frames.py — Extract cropped face frames as JPEGs instead of facecam.mp4.

Much faster than video re-encoding — works directly with images.

For each video:
  audio.wav              — 16kHz mono audio (via ffmpeg)
  frames/                — cropped + upscaled face JPEGs at 1fps
      frame_000001.jpg
      frame_000002.jpg
      ...

Usage:
    python extract_frames.py
"""

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FRAMES_FPS = 1  # frames per second to extract


def read_bbox(bbox_file: Path) -> dict | None:
    try:
        parts = dict(t.split("=") for t in bbox_file.read_text().split())
        return {k: int(v) for k, v in parts.items()}
    except Exception:
        return None


def extract_video(vid_dir: Path) -> bool:
    raw       = vid_dir / "raw.mp4"
    bbox_file = vid_dir / "bbox.txt"
    frames_dir = vid_dir / "frames"
    audio     = vid_dir / "audio.wav"

    if not raw.exists():
        log.warning("  raw.mp4 missing — skipping")
        return False

    bbox = read_bbox(bbox_file)
    if bbox is None:
        log.warning("  bbox.txt missing or invalid — skipping")
        return False

    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    frames_dir.mkdir(exist_ok=True)

    # ── Extract frames via OpenCV (faster than ffmpeg for sparse sampling) ──
    cap     = cv2.VideoCapture(str(raw))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, int(fps / FRAMES_FPS))
    n_frames = total_f // step

    log.info("  extracting ~%d frames...", n_frames)

    frame_idx = 0
    saved     = 0

    while frame_idx < total_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # Crop to bbox
        crop = frame[y:y+h, x:x+w]
        if crop.size > 0:
            # Upscale to 512px wide
            ch, cw = crop.shape[:2]
            if cw < 512:
                scale = 512 / cw
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
            fname = frames_dir / f"frame_{saved+1:06d}.jpg"
            cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        frame_idx += step

    cap.release()
    log.info("  ✓ %d frames saved", saved)

    # ── audio.wav — 16kHz mono PCM ─────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y", "-i", str(raw),
        "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(audio),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log.error("  ffmpeg audio failed: %s", result.stderr.decode()[-200:])
        return False
    log.info("  ✓ audio.wav")

    return True


def main():
    vid_dirs = sorted(d for d in DATASET_DIR.iterdir()
                      if d.is_dir() and (d / "raw.mp4").exists())

    if not vid_dirs:
        log.error("No videos in %s — run assemble.py first", DATASET_DIR)
        sys.exit(1)

    workers = min(4, len(vid_dirs))
    log.info("Extracting frames from %d videos (%d workers)...", len(vid_dirs), workers)
    ok = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_video, d): d for d in vid_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            vid_dir = futures[fut]
            success = fut.result()
            log.info("[%d/%d] %s — %s", i, len(vid_dirs), vid_dir.name,
                     "✓" if success else "✗")
            if success:
                ok += 1

    log.info("Done — %d/%d videos processed", ok, len(vid_dirs))


if __name__ == "__main__":
    main()

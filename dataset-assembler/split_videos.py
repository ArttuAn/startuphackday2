"""
split_videos.py — Split each assembled video into facecam + audio.

For each video in the dataset:
  facecam.mp4   — cropped to the bbox (just the streamer face/body)
  audio.wav     — full mixed audio, 16kHz mono (ready for ML / voice models)

Requires ffmpeg on PATH.

Usage:
    python split_videos.py
"""

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def has_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def read_bbox(bbox_file: Path) -> dict | None:
    try:
        parts = dict(t.split("=") for t in bbox_file.read_text().split())
        return {k: int(v) for k, v in parts.items()}
    except Exception:
        return None


def split_video(vid_dir: Path) -> bool:
    raw       = vid_dir / "raw.mp4"
    bbox_file = vid_dir / "bbox.txt"
    facecam   = vid_dir / "facecam.mp4"
    audio     = vid_dir / "audio.wav"

    if not raw.exists():
        log.warning("  raw.mp4 missing — skipping")
        return False

    bbox = read_bbox(bbox_file)
    if bbox is None:
        log.warning("  bbox.txt missing or invalid — skipping")
        return False

    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # ── facecam.mp4 — crop + upscale to 512px wide ────────────────────────
    scale_filter = f"crop={w}:{h}:{x}:{y},scale=512:-2:flags=lanczos"

    # Try NVIDIA GPU encoding first, fall back to CPU
    def _ffmpeg_cmd(encoder: str, extra: list) -> list:
        return ["ffmpeg", "-y", "-i", str(raw),
                "-vf", scale_filter, "-c:v", encoder, "-an",
                *extra, str(facecam)]

    gpu_cmd = _ffmpeg_cmd("h264_nvenc", ["-preset", "p1", "-cq", "23"])
    cpu_cmd = _ffmpeg_cmd("libx264",    ["-preset", "ultrafast", "-crf", "23"])

    result = subprocess.run(gpu_cmd, capture_output=True)
    if result.returncode != 0:
        result = subprocess.run(cpu_cmd, capture_output=True)
    if result.returncode != 0:
        log.error("  ffmpeg facecam failed: %s", result.stderr.decode()[-200:])
        return False
    log.info("  ✓ facecam.mp4")

    # ── audio.wav — 16kHz mono PCM ─────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y", "-i", str(raw),
        "-vn",                          # no video
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16kHz
        "-sample_fmt", "s16",           # 16-bit PCM
        str(audio),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log.error("  ffmpeg audio failed: %s", result.stderr.decode()[-200:])
        return False
    log.info("  ✓ audio.wav")

    return True


def main():
    if not has_ffmpeg():
        log.error("ffmpeg not found — install it with: winget install ffmpeg")
        sys.exit(1)

    vid_dirs = sorted(d for d in DATASET_DIR.iterdir()
                      if d.is_dir() and (d / "raw.mp4").exists())

    if not vid_dirs:
        log.error("No videos in %s — run assemble.py first", DATASET_DIR)
        sys.exit(1)

    workers = min(4, len(vid_dirs))
    log.info("Splitting %d videos with %d parallel workers...", len(vid_dirs), workers)
    ok = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(split_video, d): d for d in vid_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            vid_dir = futures[fut]
            success = fut.result()
            log.info("[%d/%d] %s — %s", i, len(vid_dirs), vid_dir.name,
                     "✓" if success else "✗")
            if success:
                ok += 1

    log.info("Done — %d/%d videos split", ok, len(vid_dirs))


if __name__ == "__main__":
    main()

"""
separator.py — Use ffmpeg to split a video into facecam + gameplay.

Given a BBox for the facecam region, produces:
  facecam.mp4   — cropped streamer region, with full audio
  gameplay.mp4  — full frame with the facecam area blacked out, with full audio
"""

from __future__ import annotations

import subprocess
import logging
from pathlib import Path

from config import FFMPEG_CRF, FFMPEG_PRESET
from detector import BBox

log = logging.getLogger("separator")


def _run(cmd: list[str], label: str) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as exc:
        log.error("%s failed:\n%s", label, exc.stderr.decode(errors="replace")[:600])
        return False


def extract_facecam(
    video_path: Path,
    bbox: BBox,
    out_path: Path,
) -> bool:
    """
    Crop the facecam region from video_path → out_path.
    Audio is copied as-is (no re-encode).
    """
    if out_path.exists():
        log.debug("facecam already exists: %s", out_path)
        return True

    log.info("Extracting facecam → %s", out_path.name)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", bbox.as_ffmpeg_crop(),
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", str(FFMPEG_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path),
    ]
    return _run(cmd, f"facecam/{out_path.stem}")


def extract_gameplay(
    video_path: Path,
    bbox: BBox,
    out_path: Path,
) -> bool:
    """
    Produce the gameplay view: full frame with the facecam area blacked out.
    Uses ffmpeg's drawbox filter to fill the facecam region with black.
    Audio is copied as-is.
    """
    if out_path.exists():
        log.debug("gameplay already exists: %s", out_path)
        return True

    log.info("Extracting gameplay (facecam masked) → %s", out_path.name)

    # drawbox=x:y:w:h:color:t   — t=fill means solid fill
    drawbox = f"drawbox={bbox.x}:{bbox.y}:{bbox.w}:{bbox.h}:black:fill"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", drawbox,
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", str(FFMPEG_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path),
    ]
    return _run(cmd, f"gameplay/{out_path.stem}")


def separate(
    video_path: Path,
    bbox: BBox,
    out_dir: Path,
) -> dict[str, Path | None]:
    """
    Run both extractions for one video.

    Returns:
        { "facecam": Path | None, "gameplay": Path | None }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    facecam_path  = out_dir / "facecam.mp4"
    gameplay_path = out_dir / "gameplay.mp4"

    fc_ok = extract_facecam(video_path,  bbox, facecam_path)
    gp_ok = extract_gameplay(video_path, bbox, gameplay_path)

    return {
        "facecam":  facecam_path  if fc_ok else None,
        "gameplay": gameplay_path if gp_ok else None,
    }

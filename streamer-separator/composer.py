"""
composer.py — Produce a side-by-side preview video.

Layout:
  [ FACECAM (scaled) | GAMEPLAY (scaled) ]

Both panels are scaled to PREVIEW_HEIGHT while preserving aspect ratio,
then stacked horizontally with ffmpeg's hstack filter.
Full audio from the gameplay track is used.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from config import FFMPEG_CRF, FFMPEG_PRESET, PREVIEW_HEIGHT

log = logging.getLogger("composer")


def _run(cmd: list[str], label: str) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as exc:
        log.error("%s failed:\n%s", label, exc.stderr.decode(errors="replace")[:600])
        return False


def make_preview(
    facecam_path: Path,
    gameplay_path: Path,
    out_path: Path,
) -> bool:
    """
    Combine facecam.mp4 and gameplay.mp4 into a side-by-side preview.mp4.

    Both inputs are scaled to PREVIEW_HEIGHT.
    Audio comes from the gameplay track (index 1).

    Returns True on success.
    """
    if out_path.exists():
        log.debug("Preview already exists: %s", out_path)
        return True

    if not facecam_path.exists() or not gameplay_path.exists():
        log.error("Missing input for preview: facecam=%s gameplay=%s",
                  facecam_path.exists(), gameplay_path.exists())
        return False

    log.info("Composing side-by-side preview → %s", out_path.name)

    h = PREVIEW_HEIGHT
    # Scale each panel to h; width auto-calculated to preserve aspect ratio
    # vstack/hstack requires same height — scale both to h
    scale_fc = f"[0:v]scale=-2:{h}[left]"
    scale_gp = f"[1:v]scale=-2:{h}[right]"
    hstack   = "[left][right]hstack=inputs=2[v]"
    filtergraph = f"{scale_fc};{scale_gp};{hstack}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(facecam_path),
        "-i", str(gameplay_path),
        "-filter_complex", filtergraph,
        "-map", "[v]",
        "-map", "1:a",           # audio from gameplay
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", str(FFMPEG_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    return _run(cmd, f"preview/{out_path.stem}")

"""
organizer.py — Build the final dataset layout.

For each accepted video, creates:
  dataset/video_<id>/
    raw.mp4        (symlink or copy from data/raw_videos/<id>/raw.mp4)
    audio.wav      (symlink or copy)
    metadata.json

metadata.json schema:
  {
    "id":         "...",
    "title":      "...",
    "url":        "...",
    "duration":   1234,
    "channel":    "...",
    "view_count": 123456,
    "has_speech": true,
    "speech_ratio": 0.72
  }
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import DATASET_DIR, RAW_VIDEOS_DIR
from logger_setup import get_logger

log = get_logger("organizer")


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink src→dst; fall back to copy if symlinks unsupported."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def organize_video(
    video: dict,
    has_speech: bool,
    speech_ratio: float,
) -> Path | None:
    """
    Build the dataset entry for one video.

    Args:
        video:        metadata dict (must have id, title, url, duration, channel)
        has_speech:   result from vad.check_speech
        speech_ratio: float 0–1 from vad.check_speech

    Returns:
        Path to the dataset/<video_id> directory, or None on error.
    """
    vid_id   = video["id"]
    src_dir  = RAW_VIDEOS_DIR / vid_id
    dst_dir  = DATASET_DIR / f"video_{vid_id}"

    raw_mp4  = src_dir / "raw.mp4"
    audio_wav = src_dir / "audio.wav"

    if not raw_mp4.exists():
        log.warning("Missing raw.mp4 for %s; skipping organizer", vid_id)
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Link / copy media files
    _safe_link_or_copy(raw_mp4,   dst_dir / "raw.mp4")
    if audio_wav.exists():
        _safe_link_or_copy(audio_wav, dst_dir / "audio.wav")

    # Write metadata.json
    meta = {
        "id":           vid_id,
        "title":        video.get("title", ""),
        "url":          video.get("url", ""),
        "duration":     video.get("duration"),
        "channel":      video.get("channel", ""),
        "view_count":   video.get("view_count"),
        "has_speech":   has_speech,
        "speech_ratio": round(speech_ratio, 4),
    }
    meta_path = dst_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info("Organized: %s → %s", vid_id, dst_dir)
    return dst_dir


def organize_batch(
    videos: list[dict],
    speech_results: dict[str, tuple[bool, float]],
    reject_no_speech: bool = True,
) -> list[Path]:
    """
    Organize all accepted videos into the dataset directory.

    Args:
        videos:           filtered video metadata list
        speech_results:   { vid_id: (has_speech, ratio) }
        reject_no_speech: if True, exclude videos that failed VAD

    Returns:
        List of successfully organized dataset directories.
    """
    organized: list[Path] = []

    for v in videos:
        vid_id = v["id"]
        has_speech, ratio = speech_results.get(vid_id, (False, 0.0))

        if reject_no_speech and not has_speech:
            log.info("EXCLUDE %s — failed VAD (speech=%.1f%%)", vid_id, ratio * 100)
            continue

        path = organize_video(v, has_speech, ratio)
        if path:
            organized.append(path)

    log.info("Dataset complete: %d videos organized in %s", len(organized), DATASET_DIR)
    return organized

"""
filter.py — Metadata-based filtering of discovered videos.

Applies duration floors, keyword scoring, and deduplication against
the on-disk seen-IDs list.
"""

from __future__ import annotations

from pathlib import Path

from config import (
    MIN_DURATION_SECONDS,
    PREFER_DURATION_SECONDS,
    PREFERRED_KEYWORDS,
    SEEN_IDS_FILE,
)
from logger_setup import get_logger

log = get_logger("filter")


# ── Seen-ID persistence ────────────────────────────────────────────────────

def load_seen_ids() -> set[str]:
    if not SEEN_IDS_FILE.exists():
        return set()
    return set(SEEN_IDS_FILE.read_text().splitlines())


def save_seen_id(vid_id: str) -> None:
    with SEEN_IDS_FILE.open("a") as f:
        f.write(vid_id + "\n")


# ── Individual video scoring ───────────────────────────────────────────────

def _keyword_score(title: str) -> int:
    """Return count of preferred keywords found in lower-cased title."""
    title_lower = title.lower()
    return sum(1 for kw in PREFERRED_KEYWORDS if kw in title_lower)


def _reject_reason(video: dict) -> str | None:
    """
    Return a rejection reason string, or None if the video should be kept.
    """
    duration = video.get("duration")

    # Hard floor — reject if duration unknown or too short
    if duration is None:
        return "duration unknown"
    if duration < MIN_DURATION_SECONDS:
        return f"too short ({duration//60} min < {MIN_DURATION_SECONDS//60} min)"

    return None  # passed


# ── Main filter function ───────────────────────────────────────────────────

def filter_videos(
    videos: list[dict],
    min_duration: int | None = None,
    max_videos: int | None = None,
) -> list[dict]:
    """
    Filter and rank a list of video metadata dicts.

    Args:
        videos:       raw list from search.py
        min_duration: override MIN_DURATION_SECONDS (seconds)
        max_videos:   cap the returned list length

    Returns:
        Filtered, ranked list ready for downloading.
    """
    floor = min_duration if min_duration is not None else MIN_DURATION_SECONDS
    seen_ids = load_seen_ids()

    accepted: list[dict] = []
    rejected_count = 0

    for v in videos:
        vid_id = v.get("id", "")

        # Dedup against already-processed IDs
        if vid_id in seen_ids:
            log.debug("SKIP (already seen): %s", vid_id)
            rejected_count += 1
            continue

        # Apply hard filter
        reason = _reject_reason(v)
        if reason:
            log.info("REJECT %s — %s — %s", vid_id, reason, v.get("title", ""))
            rejected_count += 1
            continue

        # Override min_duration if caller supplied one
        if v["duration"] < floor:
            log.info(
                "REJECT %s — below caller min_duration (%ds) — %s",
                vid_id, floor, v.get("title", ""),
            )
            rejected_count += 1
            continue

        # Attach preference flags for ranking
        v["_preferred"]     = v["duration"] >= PREFER_DURATION_SECONDS
        v["_keyword_score"] = _keyword_score(v.get("title", ""))

        accepted.append(v)

    # Sort: preferred duration first, then keyword score descending
    accepted.sort(key=lambda v: (not v["_preferred"], -v["_keyword_score"]))

    log.info(
        "Filter result: %d accepted, %d rejected (from %d total)",
        len(accepted), rejected_count, len(videos),
    )

    if max_videos:
        accepted = accepted[:max_videos]
        log.info("Capped to max_videos=%d → %d videos", max_videos, len(accepted))

    return accepted

"""
search.py — YouTube / Twitch VOD discovery via yt-dlp.

Returns a list of dicts:
  { id, url, title, duration, channel, view_count, webpage_url }
"""

from __future__ import annotations

import time
import yt_dlp

from config import DEFAULT_QUERIES, REQUEST_DELAY_SECONDS
from logger_setup import get_logger

log = get_logger("search")


# ── yt-dlp options for metadata-only extraction ────────────────────────────
_YDL_OPTS = {
    "quiet":            True,
    "no_warnings":      True,
    "extract_flat":     True,      # fast — no full extraction
    "skip_download":    True,
    "ignoreerrors":     True,
}


def _search_query(query: str, max_results: int) -> list[dict]:
    """Run a single yt-dlp ytsearch query and return raw entries."""
    url = f"ytsearch{max_results}:{query}"
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            log.warning("Search failed for %r: %s", query, exc)
            return []

    entries = info.get("entries") or []
    results = []
    for e in entries:
        if not e:
            continue
        vid_id = e.get("id") or e.get("url", "").split("v=")[-1]
        results.append({
            "id":          vid_id,
            "url":         e.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
            "title":       e.get("title", ""),
            "duration":    e.get("duration"),          # seconds or None
            "channel":     e.get("uploader") or e.get("channel", ""),
            "view_count":  e.get("view_count"),
        })
    return results


def fetch_video_metadata(url: str) -> dict | None:
    """
    Fetch full metadata for a single video URL.
    Used to fill in duration / channel when flat extraction returned None.
    """
    opts = {**_YDL_OPTS, "extract_flat": False}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            log.debug("Metadata fetch failed for %s: %s", url, exc)
            return None

    if not info:
        return None
    return {
        "id":         info.get("id", ""),
        "url":        info.get("webpage_url", url),
        "title":      info.get("title", ""),
        "duration":   info.get("duration"),
        "channel":    info.get("uploader") or info.get("channel", ""),
        "view_count": info.get("view_count"),
    }


def search_videos(
    queries: list[str] | None = None,
    max_per_query: int = 50,
) -> list[dict]:
    """
    Run all queries, deduplicate by video ID, and return merged results.

    Args:
        queries:       list of search strings; defaults to DEFAULT_QUERIES.
        max_per_query: how many results to request per query string.

    Returns:
        Deduplicated list of video metadata dicts.
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    seen: set[str] = set()
    all_results: list[dict] = []

    for q in queries:
        log.info("Searching: %r  (max %d results)", q, max_per_query)
        results = _search_query(q, max_per_query)
        log.info("  → %d raw results", len(results))

        for r in results:
            vid_id = r["id"]
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)

            # If duration is missing, do a full metadata fetch
            if r["duration"] is None:
                log.debug("  Fetching full metadata for %s", vid_id)
                full = fetch_video_metadata(r["url"])
                if full:
                    r.update(full)

            all_results.append(r)

        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Total unique videos found: %d", len(all_results))
    return all_results

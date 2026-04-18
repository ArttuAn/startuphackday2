"""
search.py — YouTube / Twitch VOD discovery via yt-dlp.

Returns a list of dicts:
  { id, url, title, duration, channel, view_count, webpage_url }
"""

from __future__ import annotations

import time
import yt_dlp

from config import DEFAULT_QUERIES, SHORTS_QUERIES, REQUEST_DELAY_SECONDS
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


def search_channel(
    channel: str,
    game_filter: str = "signalis",
    max_results: int = 200,
) -> list[dict]:
    """
    Fetch all videos from a YouTube channel and filter by game title.

    Args:
        channel:     channel handle or URL, e.g. "@lyonclips" or full URL
        game_filter: case-insensitive substring that must appear in the title
        max_results: max videos to pull from the channel (yt-dlp flat extract)

    Returns:
        List of video metadata dicts tagged with is_short where applicable.
    """
    # Normalise to a full URL
    if channel.startswith("http"):
        base_url = channel.rstrip("/")
    else:
        handle = channel.lstrip("@")
        base_url = f"https://www.youtube.com/@{handle}"

    # Use the channel's own search to let YouTube do the filtering —
    # avoids title-language mismatches (e.g. German titles without "signalis")
    if game_filter:
        tab_urls = [f"{base_url}/search?query={game_filter}"]
        log.info("Using channel search URL for game filter: %s", game_filter)
    else:
        tab_urls = [f"{base_url}/videos", f"{base_url}/shorts"]

    opts = {**_YDL_OPTS, "playlistend": max_results}
    seen: set[str] = set()
    all_results: list[dict] = []

    for tab_url in tab_urls:
        log.info("Fetching: %s", tab_url)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(tab_url, download=False)
        except Exception as exc:
            log.warning("Could not fetch %s: %s", tab_url, exc)
            continue

        entries = (info or {}).get("entries") or []
        log.info("  → %d raw entries", len(entries))

        for e in entries:
            if not e:
                continue
            vid_id = e.get("id") or ""
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)
            duration = e.get("duration")
            all_results.append({
                "id":        vid_id,
                "url":       e.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
                "title":     e.get("title", ""),
                "duration":  duration,
                "channel":   e.get("uploader") or e.get("channel", channel),
                "view_count": e.get("view_count"),
                "is_short":  duration is not None and duration <= 60,
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Channel '%s' — %d videos matching '%s'", channel, len(all_results), game_filter)
    return all_results


def search_shorts(
    queries: list[str] | None = None,
    max_per_query: int = 50,
) -> list[dict]:
    """
    Search for YouTube Shorts and tag each result with is_short=True.

    Shorts are regular YouTube videos <= 60 s. yt-dlp finds them via normal
    ytsearch — we just use hashtag-heavy queries and tag the results so the
    filter can apply a different duration floor.
    """
    if queries is None:
        queries = SHORTS_QUERIES

    seen: set[str] = set()
    all_results: list[dict] = []

    for q in queries:
        log.info("Searching shorts: %r  (max %d results)", q, max_per_query)
        results = _search_query(q, max_per_query)
        log.info("  → %d raw results", len(results))

        for r in results:
            vid_id = r["id"]
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)

            if r["duration"] is None:
                full = fetch_video_metadata(r["url"])
                if full:
                    r.update(full)

            r["is_short"] = True
            all_results.append(r)

        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Total unique shorts found: %d", len(all_results))
    return all_results


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

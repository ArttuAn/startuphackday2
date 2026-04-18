"""
downloader.py — Parallel yt-dlp downloader with retry, dedup, and progress.

Downloads best video+audio for each video in the list, storing raw mp4s in
RAW_VIDEOS_DIR.  Writes video IDs to SEEN_IDS_FILE on success so they are
skipped on future runs.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yt_dlp

from config import (
    DOWNLOAD_RETRIES,
    MAX_CONCURRENT_DOWNLOADS,
    RAW_VIDEOS_DIR,
    REQUEST_DELAY_SECONDS,
    SEEN_IDS_FILE,
)
from filter import save_seen_id
from logger_setup import get_logger

log = get_logger("downloader")


# ── yt-dlp options ─────────────────────────────────────────────────────────

import shutil as _shutil

def _has_ffmpeg() -> bool:
    return _shutil.which("ffmpeg") is not None


def _ydl_opts(out_dir: Path, vid_id: str) -> dict:
    """
    Build yt-dlp options for lowest-quality mp4 output.

    Uses vid_id (not "raw") as the intermediate filename so that yt-dlp's
    per-stream temp files (e.g. <id>.f299.mp4, <id>.f140.m4a) never collide
    with the final raw.mp4.  We rename to raw.mp4 after the merge.

    With ffmpeg:    worstvideo[mp4] + worstaudio[m4a] → merged mp4
    Without ffmpeg: worst single-file mp4 (no merge needed)
    """
    # Use video ID as stem so intermediate files are e.g. "abc123.f299.mp4"
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")

    if _has_ffmpeg():
        fmt = (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=720]+bestaudio"
            "/best[height<=720][ext=mp4]"
            "/best[height<=720]"
            "/best[ext=mp4]/best"
        )
        opts = {
            "format":              fmt,
            "merge_output_format": "mp4",
        }
    else:
        fmt = "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best"
        opts = {"format": fmt}

    return {
        **opts,
        "outtmpl":              out_tmpl,
        "retries":              DOWNLOAD_RETRIES,
        "quiet":                True,
        "no_warnings":          True,
        "ignoreerrors":         False,
        "noprogress":           True,
        "age_limit":            0,
    }


# ── Subtitle download (separate, non-fatal) ────────────────────────────────

def _download_subtitles(out_dir: Path, url: str) -> bool:
    """
    Try to fetch auto-generated English captions (.en.vtt).
    Returns True if successful, False if rate-limited or unavailable.
    Never raises — subtitle failure must not block video download.
    """
    opts = {
        "skip_download":     True,
        "writesubtitles":    True,
        "writeautomaticsub": True,
        "subtitleslangs":    ["en"],
        "subtitlesformat":   "vtt",
        "outtmpl":           str(out_dir / "%(id)s.%(ext)s"),
        "quiet":             True,
        "no_warnings":       True,
        "ignoreerrors":      True,   # 429 / unavailable = skip, not fail
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return True
    except Exception:
        return False


# ── Single-video download ──────────────────────────────────────────────────

# Extensions we consider "complete" video files
_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def _find_raw_file(out_dir: Path) -> Path | None:
    """Return the first complete video file in out_dir, ignoring .part files."""
    for f in out_dir.iterdir():
        if f.suffix in _VIDEO_EXTS and ".part" not in f.name:
            return f
    return None


def _cleanup_partial(out_dir: Path, vid_id: str) -> None:
    """Delete all partial/temp files left by a failed or interrupted download."""
    for f in out_dir.iterdir():
        # Remove .part files, yt-dlp temp fragments, and incomplete video files
        # Keep raw.mp4, audio.wav, *.vtt (those are complete outputs)
        is_complete = f.name in ("raw.mp4", "audio.wav") or f.suffix == ".vtt"
        if not is_complete:
            try:
                f.unlink()
                log.debug("  cleaned up: %s", f.name)
            except Exception:
                pass


def download_video(video: dict) -> tuple[str, bool, str]:
    """
    Download one video.

    Returns:
        (video_id, success: bool, message: str)
    """
    vid_id  = video["id"]
    url     = video["url"]
    out_dir = RAW_VIDEOS_DIR / vid_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Already downloaded?
    if (out_dir / "raw.mp4").exists():
        log.info("SKIP (file exists): %s", vid_id)
        return vid_id, True, "already downloaded"

    # Clean up any stale temp files from a previous interrupted run
    for stale in out_dir.glob(f"{vid_id}.*"):
        stale.unlink(missing_ok=True)

    log.info("Downloading: %s  — %s", vid_id, video.get("title", ""))
    if not _has_ffmpeg():
        log.warning("ffmpeg not found — using single-file mp4 fallback for %s", vid_id)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(_ydl_opts(out_dir, vid_id)) as ydl:
                ydl.download([url])

            raw = _find_raw_file(out_dir)
            if raw is None:
                raise FileNotFoundError("No video file found after download")

            target = out_dir / "raw.mp4"
            if raw != target:
                raw.rename(target)

            save_seen_id(vid_id)

            # Subtitles — best-effort, never fails the download
            sub_ok = _download_subtitles(out_dir, url)
            if not sub_ok:
                log.info("  subtitles unavailable or rate-limited for %s (skipped)", vid_id)

            log.info("OK: %s", vid_id)
            return vid_id, True, "downloaded"

        except KeyboardInterrupt:
            log.warning("Interrupted — cleaning up partial files for %s", vid_id)
            _cleanup_partial(out_dir, vid_id)
            raise   # re-raise so the batch loop can exit cleanly

        except Exception as exc:
            msg = str(exc)
            _cleanup_partial(out_dir, vid_id)

            # Age-restricted — retrying won't help, skip immediately
            if "Sign in to confirm your age" in msg or "age" in msg.lower() and "sign in" in msg.lower():
                log.warning("SKIP %s — age-restricted (no cookies)", vid_id)
                return vid_id, False, "age-restricted"

            log.warning("Attempt %d/%d failed for %s: %s", attempt, DOWNLOAD_RETRIES, vid_id, exc)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(REQUEST_DELAY_SECONDS * attempt)

    log.error("FAILED after %d attempts: %s", DOWNLOAD_RETRIES, vid_id)
    return vid_id, False, "max retries exceeded"


# ── Batch download ─────────────────────────────────────────────────────────

def download_batch(
    videos: list[dict],
    max_workers: int = MAX_CONCURRENT_DOWNLOADS,
) -> dict[str, bool]:
    """
    Download all videos in parallel.

    Args:
        videos:      list of filtered video metadata dicts
        max_workers: concurrent download threads

    Returns:
        { video_id: success_bool }
    """
    results: dict[str, bool] = {}

    log.info("Starting batch download: %d videos, %d workers", len(videos), max_workers)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(download_video, v): v["id"] for v in videos}

            for fut in as_completed(futures):
                try:
                    vid_id, success, msg = fut.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error("Unexpected error in download worker: %s", exc)
                    continue
                results[vid_id] = success
                status = "✓" if success else "✗"
                log.info("[%s] %s — %s", status, vid_id, msg)
                time.sleep(REQUEST_DELAY_SECONDS)

    except KeyboardInterrupt:
        log.warning("Download interrupted by user — already completed videos are safe.")
        pool.shutdown(wait=False, cancel_futures=True)

    succeeded = sum(v for v in results.values())
    log.info("Batch complete: %d/%d succeeded", succeeded, len(videos))
    return results

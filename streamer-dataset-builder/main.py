#!/usr/bin/env python3
"""
main.py — Streamer Reaction Dataset Builder CLI

Usage examples:
  # Search only (no download)
  python main.py --query "Signalis playthrough facecam" --max_videos 20

  # Search + download
  python main.py --max_videos 100 --download

  # Search + download + process (audio, frames, VAD, organize)
  python main.py --max_videos 100 --download --process

  # Download then process an already-searched set (resume)
  python main.py --download --process

  # Custom query, 30 min minimum, facecam extraction
  python main.py -q "Signalis stream facecam" --min_duration 1800 \\
      --download --process --facecam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import DEFAULT_QUERIES, REQUEST_DELAY_SECONDS
from logger_setup import get_logger
from search import search_videos, search_shorts, search_channel
from filter import filter_videos
from downloader import download_batch
from processor import process_video
from vad import check_speech
from organizer import organize_batch

log = get_logger("main")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streamer Reaction Dataset Builder — Signalis edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "-q", "--query",
        action="append",
        dest="queries",
        metavar="QUERY",
        help="Search query (can be repeated; defaults to built-in Signalis queries)",
    )
    p.add_argument(
        "--max_videos",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of videos to collect after filtering (default: 50)",
    )
    p.add_argument(
        "--max_per_query",
        type=int,
        default=50,
        metavar="N",
        help="Results to fetch per search query (default: 50)",
    )
    p.add_argument(
        "--min_duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Minimum video duration in seconds (default: 600 = 10 min)",
    )

    # Action flags
    p.add_argument(
        "--download",
        action="store_true",
        help="Download filtered videos via yt-dlp",
    )
    p.add_argument(
        "--process",
        action="store_true",
        help="Extract audio, sample frames, run VAD, and organize dataset",
    )
    p.add_argument(
        "--facecam",
        action="store_true",
        help="(Optional) Run facecam detection/crop during processing (requires opencv-python)",
    )
    p.add_argument(
        "--no_shorts",
        action="store_true",
        help="Skip YouTube Shorts search (only collect long-form videos)",
    )
    p.add_argument(
        "--shorts_only",
        action="store_true",
        help="Only scrape YouTube Shorts, skip long-form video search",
    )
    p.add_argument(
        "--channel",
        metavar="HANDLE",
        default="@LyonClips",
        help="Channel to scrape from (default: @lyonclips)",
    )
    p.add_argument(
        "--game",
        metavar="TITLE",
        default="signalis",
        help="Game title filter (default: signalis)",
    )
    p.add_argument(
        "--no_speech_filter",
        action="store_true",
        help="Keep videos even if VAD detects insufficient speech",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help="Parallel download workers (default: 3)",
    )

    return p.parse_args()


# ── Pipeline stages ────────────────────────────────────────────────────────

def stage_search(args: argparse.Namespace) -> list[dict]:
    log.info("═══ STAGE 1: SEARCH ═══")

    log.info("Channel mode — fetching from %s (game filter: %s)", args.channel, args.game)
    all_videos = search_channel(
        channel=args.channel,
        game_filter=args.game,
        max_results=args.max_per_query,
    )

    # Shorts first, long-form second
    shorts   = [v for v in all_videos if v.get("is_short")]
    long     = [v for v in all_videos if not v.get("is_short")]
    videos   = shorts + long
    log.info("Priority order — shorts: %d, long-form: %d", len(shorts), len(long))

    return videos


def stage_filter(args: argparse.Namespace, videos: list[dict]) -> list[dict]:
    log.info("═══ STAGE 2: FILTER ═══")
    filtered = filter_videos(
        videos,
        min_duration=args.min_duration,
        max_videos=args.max_videos,
    )
    log.info("%d videos passed filtering", len(filtered))
    return filtered


def stage_download(args: argparse.Namespace, filtered: list[dict]) -> dict[str, bool]:
    log.info("═══ STAGE 3: DOWNLOAD ═══")

    shorts   = [v for v in filtered if v.get("is_short")]
    long     = [v for v in filtered if not v.get("is_short")]

    results: dict[str, bool] = {}

    if shorts:
        log.info("Phase 1: downloading %d shorts first...", len(shorts))
        results.update(download_batch(shorts, max_workers=args.workers))

    if long:
        log.info("Phase 2: downloading %d long-form videos...", len(long))
        results.update(download_batch(long, max_workers=args.workers))

    succeeded = [vid_id for vid_id, ok in results.items() if ok]
    log.info("Downloaded %d/%d videos", len(succeeded), len(filtered))
    return results


def stage_process(
    args: argparse.Namespace,
    filtered: list[dict],
    download_results: dict[str, bool] | None,
) -> dict[str, tuple[bool, float]]:
    log.info("═══ STAGE 4: PROCESS ═══")
    speech_results: dict[str, tuple[bool, float]] = {}

    for v in filtered:
        vid_id = v["id"]

        # Skip videos that failed to download
        if download_results and not download_results.get(vid_id, False):
            log.info("SKIP processing (download failed): %s", vid_id)
            continue

        log.info("Processing: %s", vid_id)
        process_video(vid_id, do_frames=True, do_facecam=args.facecam)
        has_speech, ratio = check_speech(vid_id)
        speech_results[vid_id] = (has_speech, ratio)
        time.sleep(REQUEST_DELAY_SECONDS)

    return speech_results


def stage_organize(
    args: argparse.Namespace,
    filtered: list[dict],
    speech_results: dict[str, tuple[bool, float]],
) -> None:
    log.info("═══ STAGE 5: ORGANIZE ═══")
    organized = organize_batch(
        filtered,
        speech_results,
        reject_no_speech=not args.no_speech_filter,
    )
    log.info("Final dataset: %d entries", len(organized))


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    log.info("Streamer Reaction Dataset Builder — starting")
    log.info("Config: max_videos=%d  download=%s  process=%s  workers=%d",
             args.max_videos, args.download, args.process, args.workers)

    # 1. Search
    raw_videos = stage_search(args)
    if not raw_videos:
        log.error("No videos found. Check your queries or network connectivity.")
        sys.exit(1)

    # 2. Filter
    filtered = stage_filter(args, raw_videos)
    if not filtered:
        log.error("No videos passed filtering. Try relaxing --min_duration.")
        sys.exit(1)

    download_results: dict[str, bool] | None = None

    # 3. Download (optional)
    if args.download:
        download_results = stage_download(args, filtered)

    # 4+5. Process + Organize (optional)
    if args.process:
        speech_results = stage_process(args, filtered, download_results)
        stage_organize(args, filtered, speech_results)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

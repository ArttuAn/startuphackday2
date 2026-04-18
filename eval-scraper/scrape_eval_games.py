"""
scrape_eval_games.py — Download gameplay footage for cross-game evaluation.

Downloads pure gameplay videos (no streamer facecam needed) for games
similar to Signalis, to evaluate how well the reaction model generalises.

Default games (similar atmosphere/genre to Signalis):
  - little_nightmares
  - hollow_knight

Output structure:
  eval-scraper/data/
    little_nightmares/
      <video_id>/
        gameplay.mp4
    hollow_knight/
      <video_id>/
        gameplay.mp4

Usage:
    python scrape_eval_games.py
    python scrape_eval_games.py --games "little nightmares" "soma"
    python scrape_eval_games.py --max_per_game 3 --max_duration 600
    python scrape_eval_games.py --games "hollow knight" --max_per_game 5
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import yt_dlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"

DEFAULT_GAMES  = ["little nightmares", "hollow knight"]
SEARCH_QUERY   = "{game} gameplay no commentary"
MAX_PER_GAME   = 3
MAX_DURATION   = 900   # seconds
MIN_DURATION   = 60    # seconds


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def search_game(game: str, max_results: int,
                min_dur: int, max_dur: int) -> list[dict]:
    query = SEARCH_QUERY.format(game=game)
    url   = f"ytsearch{max_results * 3}:{query}"

    ydl_opts = {
        "quiet":         True,
        "extract_flat":  "in_playlist",
        "skip_download": True,
        "ignoreerrors":  True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = (info.get("entries") or []) if info else []
    results = []

    for e in entries:
        if not e:
            continue
        duration = e.get("duration")
        if duration is None or duration < min_dur or duration > max_dur:
            continue
        results.append({
            "id":       e.get("id", ""),
            "title":    e.get("title", ""),
            "duration": duration,
        })
        if len(results) >= max_results:
            break

    return results


def download_video(video: dict, out_dir: Path) -> bool:
    vid_id   = video["id"]
    vid_dir  = out_dir / vid_id
    gameplay = vid_dir / "gameplay.mp4"

    if gameplay.exists():
        log.info("    already downloaded — skipping")
        return True

    vid_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format":              "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl":             str(vid_dir / "gameplay.%(ext)s"),
        "quiet":               True,
        "no_warnings":         True,
        "ignoreerrors":        True,
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }

    url = f"https://www.youtube.com/watch?v={vid_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ret = ydl.download([url])
        for part in vid_dir.glob("*.part"):
            part.unlink(missing_ok=True)
        if ret != 0:
            _cleanup_dir(vid_dir)
            return False
        mp4s = list(vid_dir.glob("*.mp4"))
        if mp4s and not gameplay.exists():
            mp4s[0].rename(gameplay)
        if not gameplay.exists():
            _cleanup_dir(vid_dir)
            return False
        return True
    except Exception as e:
        log.warning("    download error: %s", e)
        _cleanup_dir(vid_dir)
        return False


def _cleanup_dir(d: Path):
    """Remove a directory if it contains no mp4 files."""
    if d.exists() and not list(d.glob("*.mp4")):
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def scrape_game(game: str, max_videos: int,
                min_dur: int, max_dur: int) -> dict:
    safe_name = sanitize_name(game)
    out_dir   = DATA_DIR / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [d for d in out_dir.iterdir()
                if d.is_dir() and (d / "gameplay.mp4").exists()]
    if len(existing) >= max_videos:
        log.info("  Already have %d/%d videos — skipping", len(existing), max_videos)
        return {"game": game, "downloaded": 0, "already_had": len(existing)}

    still_needed = max_videos - len(existing)
    log.info("  Searching: %s", SEARCH_QUERY.format(game=game))
    videos = search_game(game, max_results=still_needed + 5,
                         min_dur=min_dur, max_dur=max_dur)

    done_ids = {d.name for d in out_dir.iterdir()
                if d.is_dir() and (d / "gameplay.mp4").exists()}
    videos   = [v for v in videos if v["id"] not in done_ids]

    if not videos:
        log.warning("  No new videos found for '%s'", game)
        return {"game": game, "downloaded": 0, "already_had": len(existing)}

    log.info("  Found %d candidates — downloading up to %d", len(videos), still_needed)
    downloaded = 0

    for v in videos[:still_needed]:
        log.info("  [%d/%d] %s (%.0fs)", downloaded + 1, still_needed,
                 v["title"][:60], v["duration"])
        if download_video(v, out_dir):
            downloaded += 1
            log.info("    ✓ saved → %s/gameplay.mp4", v["id"])
        else:
            log.warning("    ✗ failed")

    return {"game": game, "downloaded": downloaded, "already_had": len(existing)}


def main():
    parser = argparse.ArgumentParser(description="Scrape gameplay footage for cross-game evaluation")
    parser.add_argument("--games",        nargs="+", default=DEFAULT_GAMES, metavar="GAME")
    parser.add_argument("--max_per_game", type=int,  default=MAX_PER_GAME)
    parser.add_argument("--max_duration", type=int,  default=MAX_DURATION)
    parser.add_argument("--min_duration", type=int,  default=MIN_DURATION)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", DATA_DIR)
    log.info("Games:  %s", args.games)
    log.info("Max per game: %d | Duration: %d–%ds\n",
             args.max_per_game, args.min_duration, args.max_duration)

    total = 0
    for game in args.games:
        log.info("━━━ %s ━━━", game.upper())
        result = scrape_game(game, args.max_per_game,
                             min_dur=args.min_duration,
                             max_dur=args.max_duration)
        log.info("  → downloaded: %d, already had: %d\n",
                 result["downloaded"], result["already_had"])
        total += result["downloaded"]

    log.info("Done — %d new videos → %s", total, DATA_DIR)
    log.info("To evaluate: cd ../model && python evaluate.py --cross_game_dir %s", DATA_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
main.py — Streamer Separator CLI

Scans the dataset folder for raw.mp4 files, auto-detects the facecam region
in each, and produces per-video output:

  output/
    <video_id>/
      facecam.mp4    — cropped streamer face/body (with audio)
      gameplay.mp4   — full frame with facecam blacked out (with audio)
      preview.mp4    — side-by-side [ facecam | gameplay ] (with audio)
      bbox.txt       — detected bounding box for reference

Usage examples:
  # Process all videos in the default dataset folder
  python main.py

  # Process a single video file directly
  python main.py --video /path/to/raw.mp4

  # Override the dataset folder
  python main.py --dataset /path/to/dataset

  # Skip detection and use a fixed region (x y w h)
  python main.py --bbox 1600 800 320 240

  # Only generate the side-by-side preview (skip if already done)
  python main.py --preview_only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import RAW_VIDEOS_DIR, OUTPUT_DIR
from detector import BBox, detect_facecam_region
from separator import separate
from composer import make_preview

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_videos(raw_videos_dir: Path) -> list[tuple[str, Path]]:
    """Return (video_id, raw.mp4 path) for every subfolder in raw_videos_dir."""
    videos = []
    for entry in sorted(raw_videos_dir.iterdir()):
        raw = entry / "raw.mp4"
        if entry.is_dir() and raw.exists():
            videos.append((entry.name, raw))
    return videos


def _save_bbox(bbox: BBox, out_dir: Path) -> None:
    txt = out_dir / "bbox.txt"
    txt.write_text(f"x={bbox.x} y={bbox.y} w={bbox.w} h={bbox.h}\n")


def _load_bbox(out_dir: Path) -> BBox | None:
    txt = out_dir / "bbox.txt"
    if not txt.exists():
        return None
    parts = {}
    for token in txt.read_text().split():
        k, v = token.split("=")
        parts[k] = int(v)
    return BBox(**parts)


# ── Single-video pipeline ──────────────────────────────────────────────────

def process_video(
    vid_id: str,
    video_path: Path,
    out_dir: Path,
    forced_bbox: BBox | None = None,
    preview_only: bool = False,
) -> bool:
    """
    Full pipeline for one video.
    Returns True if at least the preview was produced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("━━━ %s ━━━", vid_id)

    facecam_path  = out_dir / "facecam.mp4"
    gameplay_path = out_dir / "gameplay.mp4"
    preview_path  = out_dir / "preview.mp4"

    # ── 1. Detect (or load cached) bounding box ────────────────────────────
    if not preview_only:
        if forced_bbox:
            bbox = forced_bbox
            log.info("Using forced bbox: %s", bbox)
        else:
            # Try loading cached detection from a previous run
            bbox = _load_bbox(out_dir)
            if bbox:
                log.info("Loaded cached bbox: %s", bbox)
            else:
                bbox = detect_facecam_region(video_path)
                if bbox is None:
                    log.warning("No facecam detected for %s — skipping", vid_id)
                    return False
                _save_bbox(bbox, out_dir)

        # ── 2. Separate ────────────────────────────────────────────────────
        result = separate(video_path, bbox, out_dir)
        if not result["facecam"] or not result["gameplay"]:
            log.error("Separation failed for %s", vid_id)
            return False
    else:
        # preview_only: expect facecam.mp4 + gameplay.mp4 already exist
        if not facecam_path.exists() or not gameplay_path.exists():
            log.warning("preview_only: missing facecam/gameplay for %s — run without --preview_only first", vid_id)
            return False

    # ── 3. Compose side-by-side preview ───────────────────────────────────
    ok = make_preview(facecam_path, gameplay_path, preview_path)
    if ok:
        log.info("✓ Preview ready: %s", preview_path)
    return ok


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streamer Separator — auto-detect and split facecam from gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",      type=Path, default=RAW_VIDEOS_DIR,
                   help=f"raw_videos folder to scan (default: {RAW_VIDEOS_DIR})")
    p.add_argument("--video",        type=Path, default=None,
                   help="Process a single video file instead of the whole dataset")
    p.add_argument("--output",       type=Path, default=OUTPUT_DIR,
                   help=f"Output root folder (default: {OUTPUT_DIR})")
    p.add_argument("--bbox",         type=int, nargs=4, metavar=("X","Y","W","H"),
                   help="Skip detection and use a fixed bounding box: X Y W H")
    p.add_argument("--preview_only", action="store_true",
                   help="Only compose previews; skip detection + separation")
    p.add_argument("--no_mediapipe", action="store_true",
                   help="Force OpenCV Haar cascade (skip MediaPipe)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    forced_bbox = BBox(*args.bbox) if args.bbox else None

    # Build list of (id, path) to process
    if args.video:
        if not args.video.exists():
            log.error("Video file not found: %s", args.video)
            sys.exit(1)
        vid_id = args.video.parent.name or args.video.stem
        jobs = [(vid_id, args.video)]
    else:
        if not args.dataset.exists():
            log.error("Dataset folder not found: %s", args.dataset)
            sys.exit(1)
        jobs = _find_videos(args.dataset)  # args.dataset is the raw_videos dir
        if not jobs:
            log.error("No raw.mp4 files found in %s", args.dataset)
            sys.exit(1)

    log.info("Videos to process: %d", len(jobs))
    success = fail = 0

    for vid_id, video_path in jobs:
        out_dir = args.output / vid_id
        ok = process_video(
            vid_id=vid_id,
            video_path=video_path,
            out_dir=out_dir,
            forced_bbox=forced_bbox,
            preview_only=args.preview_only,
        )
        if ok:
            success += 1
        else:
            fail += 1

    log.info("Done — %d succeeded, %d failed", success, fail)
    log.info("Results in: %s", args.output)


if __name__ == "__main__":
    main()

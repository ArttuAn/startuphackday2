#!/usr/bin/env python3
"""
main.py — Streamer Separator (6-block pipeline)

┌──────────────────────────────────────────────────────────────┐
│  Block 1  SCRAPER           (external — already done)        │
│  Block 2  FACECAM DETECTOR  detector.py                      │
│  Block 3  VOICE SEPARATOR   voice_separator.py  (Demucs)     │
│  Block 4  LANDMARK EXTRACT  landmark_extractor.py (MP+DeepF) │
│  Block 5  CLIP CUTTER       clip_cutter.py                   │
│  Block 6  DATASET BUILDER   clip_extractor + clip_writer     │
└──────────────────────────────────────────────────────────────┘

Output per video:
  output/<video_id>/
    bbox.txt
    voice.wav / game_audio.wav     (Block 3)
    landmarks.json                 (Block 4)
    clip_001/
      gameplay/frame_XXXX.jpg ...  (Block 6)
      facecam/ frame_XXXX.jpg ...  (Block 6)
      audio.wav                    (Block 6, mixed)
      voice.wav                    (Block 6, streamer only)
    clip_001.json                  (Block 6)

Usage:
  python main.py                       # process all videos
  python main.py --video raw.mp4       # single video
  python main.py --bbox X Y W H        # fixed facecam box
  python main.py --report              # diagnose empty folders
  python main.py --rescan              # delete stale caches & redo
  python main.py --skip_voice          # skip Block 3 (no Demucs needed)
  python main.py --skip_landmarks      # skip Block 4 (faster, no MP needed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import OUTPUT_DIR, RAW_VIDEOS_DIR, EMOTION_CONFIDENCE_MIN
from detector import BBox, detect_facecam_region
from voice_separator import separate_voice
from landmark_extractor import extract_landmarks, load_landmarks
from emotion_scanner import scan_emotions
from clip_cutter import find_clip_windows
from clip_extractor import extract_clip
from clip_writer import build_clip_json, write_clip_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_videos(raw_videos_dir: Path) -> list[tuple[str, Path]]:
    return [
        (d.name, d / "raw.mp4")
        for d in sorted(raw_videos_dir.iterdir())
        if d.is_dir() and (d / "raw.mp4").exists()
    ]

def _video_duration(video_path: Path) -> float:
    import cv2
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps

def _save_bbox(bbox: BBox, out_dir: Path) -> None:
    (out_dir / "bbox.txt").write_text(f"x={bbox.x} y={bbox.y} w={bbox.w} h={bbox.h}\n")

def _load_bbox(out_dir: Path) -> BBox | None:
    txt = out_dir / "bbox.txt"
    if not txt.exists():
        return None
    parts = dict(token.split("=") for token in txt.read_text().split())
    return BBox(**{k: int(v) for k, v in parts.items()})

def _save_timeline(timeline, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "emotion_timeline.json").write_text(
        json.dumps([s.to_dict() for s in timeline], indent=2)
    )

def _load_timeline(out_dir: Path):
    from emotion_scanner import EmotionSample
    path = out_dir / "emotion_timeline.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not data:
        return None
    return [EmotionSample(t=d["t"], emotion=d["emotion"], scores=d["scores"]) for d in data]


# ── Report mode ────────────────────────────────────────────────────────────

def report(output_dir: Path, jobs: list[tuple[str, Path]]) -> None:
    print(f"\n{'VIDEO ID':<25} {'B2':^4} {'B3':^4} {'B4':^4} {'TL':^4} {'REACT':^6} {'CLIPS':^5}  STATUS")
    print("─" * 75)
    for vid_id, _ in jobs:
        out_dir       = output_dir / vid_id
        has_bbox      = (out_dir / "bbox.txt").exists()
        has_voice     = (out_dir / "voice.wav").exists()
        has_landmarks = (out_dir / "landmarks.json").exists()
        tl_path       = out_dir / "emotion_timeline.json"
        has_timeline  = tl_path.exists()
        n_reactions   = 0
        n_clips       = len(list(out_dir.glob("clip_*.json"))) if out_dir.exists() else 0

        if has_timeline:
            try:
                data        = json.loads(tl_path.read_text())
                n_reactions = sum(
                    1 for d in data
                    if d.get("emotion") not in ("neutral",)
                    and max(d.get("scores", {}).values() or [0]) >= EMOTION_CONFIDENCE_MIN
                )
            except Exception:
                pass

        def c(v): return "✓" if v else "✗"
        if n_clips > 0:
            status = f"✓ {n_clips} clips"
        elif has_timeline and n_reactions == 0:
            status = "⚠ scanned — no reactions (lower EMOTION_CONFIDENCE_MIN?)"
        elif has_timeline:
            status = "⚠ reactions found but not extracted yet"
        elif has_bbox:
            status = "⚠ bbox ok — scan not run yet"
        else:
            status = "✗ no facecam detected"

        print(f"{vid_id:<25} {c(has_bbox):^4} {c(has_voice):^4} {c(has_landmarks):^4}"
              f" {c(has_timeline):^4} {n_reactions:^6} {n_clips:^5}  {status}")
    print()


# ── Single-video pipeline ──────────────────────────────────────────────────

def process_video(
    vid_id:          str,
    video_path:      Path,
    out_dir:         Path,
    forced_bbox:     BBox | None = None,
    rescan:          bool = False,
    skip_voice:      bool = False,
    skip_landmarks:  bool = False,
    scan_only:       bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("━━━ %s ━━━", vid_id)

    # ── Block 2: Facecam detection ─────────────────────────────────────────
    if forced_bbox:
        bbox = forced_bbox
    else:
        bbox = _load_bbox(out_dir)
        if bbox:
            log.info("[B2] Cached bbox: %s", bbox)
        else:
            log.info("[B2] Detecting facecam region...")
            bbox = detect_facecam_region(video_path)
            if bbox is None:
                log.warning("✗ [B2] No facecam detected — skipping %s", vid_id)
                return 0
            _save_bbox(bbox, out_dir)
            log.info("[B2] ✓ bbox saved: %s", bbox)

    # ── Block 3: Voice separation ──────────────────────────────────────────
    voice_path = None
    if not skip_voice:
        log.info("[B3] Separating voice...")
        result     = separate_voice(video_path, out_dir)
        voice_path = result.get("voice")
        if voice_path:
            log.info("[B3] ✓ voice.wav ready")
        else:
            log.warning("[B3] ⚠ Voice separation failed — continuing without it")
    else:
        log.info("[B3] Skipped (--skip_voice)")
        existing = out_dir / "voice.wav"
        if existing.exists():
            voice_path = existing

    # ── Block 4: Landmark extraction ──────────────────────────────────────
    landmarks = None
    if not skip_landmarks:
        if rescan and (out_dir / "landmarks.json").exists():
            (out_dir / "landmarks.json").unlink()
        log.info("[B4] Extracting landmarks...")
        lm_path   = extract_landmarks(video_path, bbox, out_dir)
        landmarks = load_landmarks(out_dir) if lm_path else None
        if landmarks:
            log.info("[B4] ✓ %d landmark frames loaded", len(landmarks))
        else:
            log.warning("[B4] ⚠ Landmarks unavailable — falling back to emotion scan")
    else:
        log.info("[B4] Skipped (--skip_landmarks)")
        landmarks = load_landmarks(out_dir)

    # ── Block 5: Clip cutting (uses landmarks if available, else emotion scan)
    if landmarks:
        timeline_for_cutting = landmarks
    else:
        if rescan and (out_dir / "emotion_timeline.json").exists():
            (out_dir / "emotion_timeline.json").unlink()
        timeline_for_cutting = _load_timeline(out_dir)
        if not timeline_for_cutting:
            log.info("[B5] Running emotion scan (no landmarks available)...")
            timeline_for_cutting = scan_emotions(video_path, bbox)
            if not timeline_for_cutting:
                log.warning("✗ Empty timeline — skipping %s", vid_id)
                return 0
            _save_timeline(timeline_for_cutting, out_dir)

    # Also keep coarse emotion timeline for clip JSON
    coarse_timeline = _load_timeline(out_dir)
    if coarse_timeline is None:
        coarse_timeline = timeline_for_cutting

    reactions = sum(1 for s in timeline_for_cutting
                    if (s.is_reaction() if hasattr(s, "is_reaction") else
                        s.get("emotion", "neutral") not in ("neutral",)))
    log.info("[B5] %d reaction frames in timeline", reactions)

    if scan_only:
        return 0

    if reactions == 0:
        log.warning("✗ [B5] No reactions — no clips will be cut for %s", vid_id)
        return 0

    duration = _video_duration(video_path)
    windows  = find_clip_windows(timeline_for_cutting, duration)
    if not windows:
        log.warning("✗ [B5] No clip windows found for %s", vid_id)
        return 0

    # ── Block 6: Dataset builder ───────────────────────────────────────────
    written = 0
    for window in windows:
        clip_id  = f"clip_{window.index:03d}"
        json_out = out_dir / f"{clip_id}.json"

        if json_out.exists():
            log.info("[B6] SKIP (exists): %s", clip_id)
            written += 1
            continue

        clip_dir = out_dir / clip_id
        paths    = extract_clip(video_path, bbox, window, clip_dir, voice_path)
        data     = build_clip_json(vid_id, window, paths, coarse_timeline, landmarks)
        write_clip_json(data, out_dir)
        written += 1

    log.info("✓ %s — %d clips written", vid_id, written)
    return written


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streamer Separator — 6-block reaction clip pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",         type=Path, default=RAW_VIDEOS_DIR)
    p.add_argument("--video",           type=Path, default=None)
    p.add_argument("--output",          type=Path, default=OUTPUT_DIR)
    p.add_argument("--bbox",            type=int,  nargs=4, metavar=("X","Y","W","H"))
    p.add_argument("--rescan",          action="store_true",
                   help="Delete cached landmarks/timelines and redo")
    p.add_argument("--report",          action="store_true",
                   help="Print status table for all videos")
    p.add_argument("--scan_only",       action="store_true",
                   help="Run blocks 2–4 only; skip clip extraction")
    p.add_argument("--skip_voice",      action="store_true",
                   help="Skip Block 3 (voice separation)")
    p.add_argument("--skip_landmarks",  action="store_true",
                   help="Skip Block 4 (landmark extraction)")
    p.add_argument("--no_mediapipe",    action="store_true",
                   help="Force Haar cascade in Block 2")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    forced = BBox(*args.bbox) if args.bbox else None

    if args.video:
        if not args.video.exists():
            log.error("File not found: %s", args.video)
            sys.exit(1)
        jobs = [(args.video.parent.name or args.video.stem, args.video)]
    else:
        if not args.dataset.exists():
            log.error("Dataset folder not found: %s", args.dataset)
            sys.exit(1)
        jobs = _find_videos(args.dataset)
        if not jobs:
            log.error("No raw.mp4 files found in %s", args.dataset)
            sys.exit(1)

    log.info("Videos to process: %d", len(jobs))

    if args.report:
        report(args.output, jobs)
        return

    total_clips = 0
    for vid_id, video_path in jobs:
        total_clips += process_video(
            vid_id=vid_id,
            video_path=video_path,
            out_dir=args.output / vid_id,
            forced_bbox=forced,
            rescan=args.rescan,
            skip_voice=args.skip_voice,
            skip_landmarks=args.skip_landmarks,
            scan_only=args.scan_only,
        )

    log.info("Pipeline complete — %d total clips across %d videos", total_clips, len(jobs))


if __name__ == "__main__":
    main()

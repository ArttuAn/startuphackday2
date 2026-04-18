"""
extract_clips.py — Extract paired (gameplay, facecam) clips from each video.

For each video in the dataset that has emotions.json + bbox.txt + raw.mp4,
this script produces:

  dataset/<video_id>/
      facecam.mp4          — full video cropped to facecam bbox
      gameplay.mp4         — full video with facecam region blacked out
      clips/
          <t>_<emotion>/
              facecam.mp4  — CLIP_DURATION clip, cropped to facecam
              gameplay.mp4 — CLIP_DURATION clip, facecam blacked out
              meta.json    — emotion, scores, audio, caption at this moment

Clips are extracted at every reaction timestamp in emotions.json.
A sample of neutral moments is also included for training balance.

Config knobs at the top of this file.

Usage:
    python extract_clips.py
    python extract_clips.py --skip_full   # skip full-video extraction, clips only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
CLIP_DURATION       = 5      # seconds per clip
NEUTRAL_SAMPLE_RATE = 5      # keep 1 in N neutral frames (for training balance)
MIN_REACTION_SCORE  = 0      # any non-neutral/non-none emotion qualifies

FFMPEG_VIDEO_OPTS   = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
FFMPEG_AUDIO_OPTS   = ["-c:a", "aac", "-b:a", "128k"]


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_bbox(bbox_file: Path) -> tuple[int, int, int, int] | None:
    """Parse bbox.txt → (x, y, w, h)."""
    try:
        parts = dict(p.split("=") for p in bbox_file.read_text().split())
        return int(parts["x"]), int(parts["y"]), int(parts["w"]), int(parts["h"])
    except Exception as e:
        log.warning("  could not parse bbox.txt: %s", e)
        return None


def ffmpeg_run(cmd: list[str], label: str) -> bool:
    """Run an ffmpeg command, return True on success."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"] + cmd,
        capture_output=True,
    )
    if result.returncode != 0:
        log.warning("  ffmpeg failed (%s): %s", label, result.stderr.decode()[:300])
        return False
    return True


# ── Full-video extraction ──────────────────────────────────────────────────

def extract_full_facecam(raw: Path, bbox: tuple, out: Path) -> bool:
    """Crop the entire video to the facecam bbox → facecam.mp4."""
    if out.exists():
        log.info("  facecam.mp4 already exists — skipping")
        return True
    x, y, w, h = bbox
    return ffmpeg_run(
        ["-i", str(raw),
         "-vf", f"crop={w}:{h}:{x}:{y}",
         *FFMPEG_VIDEO_OPTS, *FFMPEG_AUDIO_OPTS,
         str(out)],
        "full facecam",
    )


def extract_full_gameplay(raw: Path, bbox: tuple, out: Path) -> bool:
    """Black out the facecam region in the full video → gameplay.mp4."""
    if out.exists():
        log.info("  gameplay.mp4 already exists — skipping")
        return True
    x, y, w, h = bbox
    return ffmpeg_run(
        ["-i", str(raw),
         "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill",
         *FFMPEG_VIDEO_OPTS, *FFMPEG_AUDIO_OPTS,
         str(out)],
        "full gameplay",
    )


# ── Clip extraction ────────────────────────────────────────────────────────

def extract_clip(raw: Path, bbox: tuple, t: float, clip_dir: Path) -> bool:
    """
    Extract a CLIP_DURATION clip centred on timestamp t.

    Produces:
        clip_dir/facecam.mp4  — cropped to bbox
        clip_dir/gameplay.mp4 — bbox blacked out
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = bbox
    # Start half a clip before the emotion timestamp, clamped to 0
    start = max(0.0, t - CLIP_DURATION / 2)

    fc_out = clip_dir / "facecam.mp4"
    gp_out = clip_dir / "gameplay.mp4"

    if fc_out.exists() and gp_out.exists():
        return True  # already extracted

    ok = True
    if not fc_out.exists():
        ok &= ffmpeg_run(
            ["-ss", str(start), "-t", str(CLIP_DURATION),
             "-i", str(raw),
             "-vf", f"crop={w}:{h}:{x}:{y}",
             *FFMPEG_VIDEO_OPTS, *FFMPEG_AUDIO_OPTS,
             str(fc_out)],
            f"facecam clip t={t}",
        )
    if not gp_out.exists():
        ok &= ffmpeg_run(
            ["-ss", str(start), "-t", str(CLIP_DURATION),
             "-i", str(raw),
             "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill",
             *FFMPEG_VIDEO_OPTS, *FFMPEG_AUDIO_OPTS,
             str(gp_out)],
            f"gameplay clip t={t}",
        )
    return ok


# ── Per-video processor ────────────────────────────────────────────────────

def process_video(vid_dir: Path, skip_full: bool = False) -> dict:
    raw        = vid_dir / "raw.mp4"
    bbox_file  = vid_dir / "bbox.txt"
    emo_file   = vid_dir / "emotions.json"

    if not raw.exists():
        return {"error": "no raw.mp4"}
    if not bbox_file.exists():
        return {"error": "no bbox.txt"}
    if not emo_file.exists():
        return {"error": "no emotions.json — run classify_emotions first"}

    bbox = parse_bbox(bbox_file)
    if bbox is None:
        return {"error": "bbox parse failed"}

    timeline = json.loads(emo_file.read_text())
    if not timeline:
        return {"error": "empty emotions.json"}

    # ── Full-video extraction ──────────────────────────────────────────────
    if not skip_full:
        log.info("  extracting full facecam.mp4...")
        extract_full_facecam(raw, bbox, vid_dir / "facecam.mp4")
        log.info("  extracting full gameplay.mp4...")
        extract_full_gameplay(raw, bbox, vid_dir / "gameplay.mp4")

    # ── Clip extraction ────────────────────────────────────────────────────
    clips_dir = vid_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    reaction_clips = 0
    neutral_clips  = 0
    neutral_idx    = 0

    for entry in timeline:
        t       = entry["t"]
        emotion = entry.get("emotion", "none")

        is_reaction = emotion not in ("neutral", "none")

        # Always extract reaction moments; sample neutral for balance
        if is_reaction:
            include = True
        else:
            neutral_idx += 1
            include = (neutral_idx % NEUTRAL_SAMPLE_RATE == 0)

        if not include:
            continue

        clip_name = f"{int(t):06d}s_{emotion}"
        clip_dir  = clips_dir / clip_name

        ok = extract_clip(raw, bbox, t, clip_dir)
        if ok:
            # Write metadata alongside the clip
            meta = {k: entry[k] for k in entry if k != "scores"}
            meta["scores"]    = entry.get("scores", {})
            meta["clip_start"] = max(0.0, t - CLIP_DURATION / 2)
            meta["clip_end"]   = max(0.0, t - CLIP_DURATION / 2) + CLIP_DURATION
            (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            if is_reaction:
                reaction_clips += 1
            else:
                neutral_clips += 1

    return {
        "reaction_clips": reaction_clips,
        "neutral_clips":  neutral_clips,
        "total_clips":    reaction_clips + neutral_clips,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract paired gameplay/facecam clips")
    parser.add_argument("--with_full", action="store_true",
                        help="Also extract full-video facecam.mp4/gameplay.mp4 (slow — off by default)")
    args = parser.parse_args()

    vid_dirs = sorted(
        d for d in DATASET_DIR.iterdir()
        if d.is_dir() and (d / "raw.mp4").exists() and (d / "emotions.json").exists()
    )

    if not vid_dirs:
        log.error("No videos with emotions.json found in %s", DATASET_DIR)
        sys.exit(1)

    log.info("Extracting clips for %d videos (clip_duration=%ds, neutral_rate=1/%d)",
             len(vid_dirs), CLIP_DURATION, NEUTRAL_SAMPLE_RATE)
    if not args.with_full:
        log.info("Clips only — skipping full-video extraction (pass --with_full to enable)")

    total_reaction = 0
    total_neutral  = 0

    for i, vid_dir in enumerate(vid_dirs, 1):
        log.info("[%d/%d] %s", i, len(vid_dirs), vid_dir.name)
        result = process_video(vid_dir, skip_full=not args.with_full)

        if "error" in result:
            log.warning("  ✗ %s", result["error"])
            continue

        log.info("  ✓ %d reaction clips, %d neutral clips",
                 result["reaction_clips"], result["neutral_clips"])
        total_reaction += result["reaction_clips"]
        total_neutral  += result["neutral_clips"]

    log.info("Done — %d reaction clips + %d neutral clips = %d total training pairs",
             total_reaction, total_neutral, total_reaction + total_neutral)


if __name__ == "__main__":
    main()

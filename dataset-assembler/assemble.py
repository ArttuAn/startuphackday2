"""
assemble.py — Combine separator bbox results with raw videos into one dataset.

Reads bboxes.json from the separator to find which videos had a face detected,
then copies raw.mp4 + audio.wav + bbox.txt + facecam.jpg into a clean dataset
folder.

Usage:
    python assemble.py

Output:
    dataset/<video_id>/
        raw.mp4         copy of the source video
        audio.wav       copy of the extracted audio (needed for emotion audio features)
        bbox.txt        x=.. y=.. w=.. h=..
        facecam.jpg     cropped face screenshot
"""

import json
import logging
import shutil
import sys
from pathlib import Path

from config import DATASET_DIR, RAW_VIDEOS_DIR, SEPARATOR_OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    bboxes_file = SEPARATOR_OUTPUT_DIR / "bboxes.json"
    if not bboxes_file.exists():
        log.error("bboxes.json not found at %s — run detect_bboxes.py first", bboxes_file)
        sys.exit(1)

    bboxes = json.loads(bboxes_file.read_text())
    log.info("Found %d videos with detected faces", len(bboxes))

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for i, (vid_id, bbox) in enumerate(bboxes.items(), 1):
        log.info("[%d/%d] %s", i, len(bboxes), vid_id)

        out_dir = DATASET_DIR / vid_id

        raw_video = RAW_VIDEOS_DIR / vid_id / "raw.mp4"
        sep_dir   = SEPARATOR_OUTPUT_DIR / vid_id

        if not raw_video.exists():
            log.warning("  raw.mp4 not found — skipping")
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy raw video
        shutil.copy2(raw_video, out_dir / "raw.mp4")

        # Copy audio (required for audio emotion features in classify_emotions.py)
        audio_src = RAW_VIDEOS_DIR / vid_id / "audio.wav"
        if audio_src.exists():
            shutil.copy2(audio_src, out_dir / "audio.wav")
        else:
            log.warning("  audio.wav not found in raw_videos — audio features will be skipped")

        # Copy bbox.txt
        bbox_src = sep_dir / "bbox.txt"
        if bbox_src.exists():
            shutil.copy2(bbox_src, out_dir / "bbox.txt")
        else:
            # Write from bboxes.json if file missing
            (out_dir / "bbox.txt").write_text(
                f"x={bbox['x']} y={bbox['y']} w={bbox['w']} h={bbox['h']}"
            )

        # Copy facecam screenshot
        facecam_src = sep_dir / "facecam.jpg"
        if facecam_src.exists():
            shutil.copy2(facecam_src, out_dir / "facecam.jpg")

        log.info("  ✓ assembled")
        copied += 1

    log.info("Done — %d assembled, %d skipped", copied, skipped)
    log.info("Dataset: %s", DATASET_DIR)


if __name__ == "__main__":
    main()

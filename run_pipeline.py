"""
run_pipeline.py — Full pipeline orchestrator.

Runs all steps in order:
  1. streamer-dataset-builder  — scrape & download YouTube videos
  2. face-validator             — remove low-quality / no-face videos
  3. streamer-separator         — detect facecam bboxes
  4. dataset-assembler          — assemble dataset
  5. dataset-assembler          — classify emotions (face + audio)
  6. dataset-assembler          — extract paired gameplay/facecam clips
  7. model                      — train emotion predictor (EfficientNet-B0)
  8. model                      — build retrieval index
  9. model                      — evaluate (emotion consistency + FID)

Usage:
    python run_pipeline.py                     # full pipeline, 50 videos
    python run_pipeline.py --max_videos 25     # limit to 25 videos
    python run_pipeline.py --skip_scrape
    python run_pipeline.py --skip_scrape --skip_validate --skip_bbox
    python run_pipeline.py --skip_scrape --skip_validate --skip_bbox --skip_assemble --skip_emotions
    python run_pipeline.py --skip_scrape --skip_validate --skip_bbox --skip_assemble --skip_emotions --skip_clips
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent


def build_steps(max_videos: int, shorts_only: bool = False) -> list[dict]:
    scrape_args = ["--max_videos", str(max_videos), "--download"]
    if shorts_only:
        scrape_args.append("--shorts_only")
    return [
        {
            "name":   "1. Scrape & download videos",
            "script": BASE / "streamer-dataset-builder" / "main.py",
            "args":   scrape_args,
            "flag":   "skip_scrape",
        },
        {
            "name":   "2. Validate face quality (presence, brightness, VTuber filter)",
            "script": BASE / "face-validator" / "validate.py",
            "args":   [],
            "flag":   "skip_validate",
        },
        {
            "name":   "3. Detect facecam bboxes",
            "script": BASE / "streamer-separator" / "detect_bboxes.py",
            "args":   [],
            "flag":   "skip_bbox",
        },
        {
            "name":   "4. Assemble dataset",
            "script": BASE / "dataset-assembler" / "assemble.py",
            "args":   [],
            "flag":   "skip_assemble",
        },
        {
            "name":   "5. Classify emotions + remove emotionless streamers",
            "script": BASE / "dataset-assembler" / "classify_emotions.py",
            "args":   [],
            "flag":   "skip_emotions",
        },
        {
            "name":   "6. Extract paired gameplay/facecam clips",
            "script": BASE / "dataset-assembler" / "extract_clips.py",
            "args":   [],          # clips only by default; add "--with_full" for full videos
            "flag":   "skip_clips",
        },
        {
            "name":   "7. Train emotion predictor (EfficientNet-B0)",
            "script": BASE / "model" / "train.py",
            "args":   [],
            "flag":   "skip_train",
        },
        {
            "name":   "8. Build retrieval index",
            "script": BASE / "model" / "generate.py",
            "args":   ["--build_index"],
            "flag":   "skip_index",
        },
        {
            "name":   "9. Evaluate (emotion consistency + FID)",
            "script": BASE / "model" / "evaluate.py",
            "args":   ["--skip_fid",
                       "--cross_game_dir", str(BASE / "eval-scraper" / "data")],
            "flag":   "skip_eval",
        },
    ]


def run_step(step: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"  {step['name']}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(step["script"])] + step["args"]
    result = subprocess.run(cmd, cwd=step["script"].parent)

    if result.returncode != 0:
        print(f"\n✗ Step failed: {step['name']}")
        return False

    print(f"\n✓ Done: {step['name']}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full streamer dataset pipeline")
    parser.add_argument("--max_videos", type=int, default=50,
                        help="Max videos to scrape and download (default: 50)")
    parser.add_argument("--shorts_only", action="store_true",
                        help="Only scrape YouTube Shorts")
    steps = build_steps(50)  # placeholder; rebuilt after parsing
    for step in steps:
        parser.add_argument(f"--{step['flag']}", action="store_true",
                            help=f"Skip: {step['name']}")
    args = parser.parse_args()

    steps = build_steps(args.max_videos, shorts_only=args.shorts_only)

    print(f"Starting pipeline (max_videos={args.max_videos})...\n")
    failed = False

    for step in steps:
        if getattr(args, step["flag"], False):
            print(f"  ⏭  Skipping: {step['name']}")
            continue

        if not run_step(step):
            print(f"\nPipeline stopped at: {step['name']}")
            print("Fix the error and re-run with earlier steps skipped.")
            failed = True
            break

    if not failed:
        print("\n" + "="*60)
        print("  ✓ Pipeline complete!")
        print("="*60)


if __name__ == "__main__":
    main()

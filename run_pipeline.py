"""
run_pipeline.py — Full pipeline orchestrator.

Runs all steps in order:
  1. streamer-dataset-builder  — scrape & download YouTube videos
  2. streamer-separator         — detect facecam bboxes
  3. dataset-assembler          — assemble dataset
  4. dataset-assembler          — split into facecam.mp4
  5. dataset-assembler          — classify emotions

Usage:
    python run_pipeline.py

Skip steps you don't need:
    python run_pipeline.py --skip_scrape
    python run_pipeline.py --skip_scrape --skip_bbox
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent

STEPS = [
    {
        "name":   "1. Scrape & download videos",
        "script": BASE / "streamer-dataset-builder" / "main.py",
        "args":   ["--max_videos", "50", "--download"],
        "flag":   "skip_scrape",
    },
    {
        "name":   "2. Detect facecam bboxes",
        "script": BASE / "streamer-separator" / "detect_bboxes.py",
        "args":   [],
        "flag":   "skip_bbox",
    },
    {
        "name":   "3. Assemble dataset",
        "script": BASE / "dataset-assembler" / "assemble.py",
        "args":   [],
        "flag":   "skip_assemble",
    },
    {
        "name":   "4. Split videos → facecam.mp4",
        "script": BASE / "dataset-assembler" / "split_videos.py",
        "args":   [],
        "flag":   "skip_split",
    },
    {
        "name":   "5. Classify emotions",
        "script": BASE / "dataset-assembler" / "classify_emotions.py",
        "args":   [],
        "flag":   "skip_emotions",
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
    for step in STEPS:
        parser.add_argument(f"--{step['flag']}", action="store_true",
                            help=f"Skip: {step['name']}")
    args = parser.parse_args()

    print("Starting pipeline...\n")
    failed = False

    for step in STEPS:
        if getattr(args, step["flag"]):
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

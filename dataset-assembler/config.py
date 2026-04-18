"""
config.py — Paths for the dataset assembler.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Inputs ─────────────────────────────────────────────────────────────────

# Raw videos from the scraper
RAW_VIDEOS_DIR = Path(r"C:\Users\Admin\OneDrive\Desktop\startuphackYT\streamer-dataset-builder\data\raw_videos")

# Separator output (bboxes.json + per-video bbox.txt / facecam.jpg)
SEPARATOR_OUTPUT_DIR = Path(r"C:\Users\Admin\OneDrive\Desktop\startuphackYT\streamer-separator\output")

# ── Output ─────────────────────────────────────────────────────────────────

# Final assembled dataset
DATASET_DIR = BASE_DIR / "dataset"

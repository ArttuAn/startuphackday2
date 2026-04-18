"""
config.py — Configuration for the Streamer Separator app.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Where the raw downloaded videos live
# Structure: RAW_VIDEOS_DIR/<video_id>/raw.mp4
RAW_VIDEOS_DIR = Path(r"C:\Users\Admin\OneDrive\Desktop\startuphackYT\streamer-dataset-builder\data\raw_videos")

# ── Detection settings ─────────────────────────────────────────────────────
# How many frames to sample for finding the facecam region
DETECTION_SAMPLE_FRAMES = 60

# Minimum fraction of sampled frames that must contain a face
# for the region to be considered valid
MIN_FACE_HIT_RATE = 0.25

# Expand the detected face box to include the body (multipliers)
BODY_WIDTH_SCALE  = 2.2   # widen to capture shoulders
BODY_HEIGHT_SCALE = 3.5   # extend down to capture torso

# Snap detected region to the nearest screen corner if it's within this
# fraction of the frame width/height from that corner
CORNER_SNAP_THRESHOLD = 0.35

# ── Output settings ────────────────────────────────────────────────────────
# Side-by-side preview: height to scale both panels to (keeps aspect ratio)
PREVIEW_HEIGHT = 720

# ffmpeg encoding preset (ultrafast = fastest encode, larger file)
FFMPEG_PRESET  = "fast"
FFMPEG_CRF     = 28       # quality: lower = better (18–28 typical)

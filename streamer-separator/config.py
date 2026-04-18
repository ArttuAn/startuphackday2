"""
config.py — Configuration for the Streamer Separator app.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
OUTPUT_DIR     = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Where the raw downloaded videos live
# Structure: RAW_VIDEOS_DIR/<video_id>/raw.mp4
RAW_VIDEOS_DIR = Path(r"C:\Users\Admin\OneDrive\Desktop\startuphackYT\streamer-dataset-builder\data\raw_videos")

# ── Facecam detection settings ─────────────────────────────────────────────
DETECTION_SAMPLE_FRAMES = 60      # frames sampled to locate the facecam box
MIN_FACE_HIT_RATE       = 0.05    # min fraction of frames that must have a face
BODY_WIDTH_SCALE        = 2.2     # expand face box → body (horizontal)
BODY_HEIGHT_SCALE       = 3.5     # expand face box → body (vertical)
CORNER_SNAP_THRESHOLD   = 0.35    # snap box to corner if this close to edge

# ── Emotion scanning settings ──────────────────────────────────────────────
# Frames per second to analyse for emotion (lower = faster, less precise)
# 0.2 = 1 frame every 5 seconds — fast enough to catch reactions, ~180 frames/hr
SCAN_FPS               = 0.2

# Emotions that count as a "reaction" (non-neutral)
REACTION_EMOTIONS      = {"fear", "surprise", "angry", "disgust", "sad", "happy"}

# Minimum emotion score (0–100) for a frame to count as a reaction
# (derived from MediaPipe blendshapes — lower = more sensitive)
EMOTION_CONFIDENCE_MIN = 25.0

# ── Clip cutting settings ──────────────────────────────────────────────────
PRE_BUFFER_SEC  = 3.0    # seconds before first reaction frame in a cluster
POST_BUFFER_SEC = 5.0    # seconds after  last  reaction frame in a cluster
MERGE_GAP_SEC   = 8.0    # merge two events if they are closer than this
MIN_CLIP_SEC    = 4.0    # drop clips shorter than this
MAX_CLIP_SEC    = 60.0   # cap runaway clips at this length

# ── Extraction settings ────────────────────────────────────────────────────
FRAMES_FPS      = 1      # frame rate for saved gameplay/facecam frames (jpg)
AUDIO_RATE      = 16000  # Hz, mono PCM wav

# ── ffmpeg settings ────────────────────────────────────────────────────────
FFMPEG_PRESET   = "fast"
FFMPEG_CRF      = 28

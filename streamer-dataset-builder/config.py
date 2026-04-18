"""
config.py — Central configuration for the Streamer Reaction Dataset Builder.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_VIDEOS_DIR  = DATA_DIR / "raw_videos"
FRAMES_DIR      = DATA_DIR / "frames"
FACECAM_DIR     = DATA_DIR / "facecam"
DATASET_DIR     = BASE_DIR / "dataset"
LOGS_DIR        = BASE_DIR / "logs"

for _d in [RAW_VIDEOS_DIR, FRAMES_DIR, FACECAM_DIR, DATASET_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Search defaults ────────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "Signalis playthrough facecam",
    "Signalis full gameplay stream",
    "Signalis streamer reaction",
    "Signalis twitch vod",
]

# ── Filter thresholds ──────────────────────────────────────────────────────
MIN_DURATION_SECONDS   = 10 * 60   # 10 min hard floor
PREFER_DURATION_SECONDS = 30 * 60  # 30 min preference floor

PREFERRED_KEYWORDS = {"facecam", "stream", "vod", "playthrough", "let's play",
                      "lets play", "reaction", "horror"}

# ── Download settings ──────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 3
DOWNLOAD_RETRIES         = 3
REQUEST_DELAY_SECONDS    = 2        # polite delay between requests

# ── Processing settings ────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE   = 16000        # Hz, mono
FRAMES_PER_SECOND   = 1           # frames to sample for facecam detection

# ── VAD settings ──────────────────────────────────────────────────────────
VAD_SPEECH_THRESHOLD = 0.05        # reject if speech < 5 % of duration

# ── Dedup ─────────────────────────────────────────────────────────────────
SEEN_IDS_FILE = DATA_DIR / "seen_ids.txt"

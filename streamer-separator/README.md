# Streamer Separator

Takes downloaded streamer videos and extracts reaction clips with paired gameplay footage, separated voice audio, face landmarks, and emotion labels — ready for AI training.

## Setup

```bash
# Requires Python 3.11 or 3.12 (not 3.13+)
conda create -n streamer python=3.12 -y
conda activate streamer
pip install -r requirements.txt
winget install ffmpeg
```

## Usage

```bash
# Full pipeline (all 6 blocks)
python main.py

# Skip heavy blocks while testing
python main.py --skip_voice --skip_landmarks

# Diagnose why folders are empty
python main.py --report

# Clear stale cache and reprocess
python main.py --rescan
```

## Pipeline

| Block | File | Input → Output |
|-------|------|----------------|
| 1 | *(scraper)* | — → `raw.mp4` |
| 2 | `detector.py` | `raw.mp4` → `bbox.txt` |
| 3 | `voice_separator.py` | `raw.mp4` → `voice.wav` + `game_audio.wav` |
| 4 | `landmark_extractor.py` | facecam crop → `landmarks.json` (468 keypoints + head pose + emotion) |
| 5 | `clip_cutter.py` | landmarks → reaction windows with 0.75s gameplay offset |
| 6 | `clip_extractor` + `clip_writer` | windows → frames + audio + `clip_XXX.json` |

## Output

```
output/<video_id>/
    bbox.txt
    voice.wav / game_audio.wav
    landmarks.json
    clip_001/
        gameplay/frame_0001.jpg ...   # full frame at 1fps
        facecam/ frame_0001.jpg ...   # cropped streamer region
        audio.wav                     # mixed audio slice
        voice.wav                     # streamer voice only
    clip_001.json
```

Each `clip_XXX.json` contains:

```json
{
  "video_id": "abc123",
  "clip_id": "clip_001",
  "t_start": 120.0,
  "t_end": 128.0,
  "t_gameplay": 111.25,
  "gameplay_frames": "clip_001/gameplay",
  "facecam_frames": "clip_001/facecam",
  "audio": "clip_001/audio.wav",
  "voice": "clip_001/voice.wav",
  "landmarks": [{"frame": 3600, "t": 120.0, "landmarks": [[x,y,z], ...], "head_pose": {...}, "emotion": "fear", "emotion_scores": {...}}],
  "emotion_timeline": [{"t": 120.0, "emotion": "fear", "scores": {...}}]
}
```

## Options

| Flag | Description |
|------|-------------|
| `--report` | Print status table showing which blocks completed per video |
| `--rescan` | Delete cached landmarks/timelines and reprocess |
| `--skip_voice` | Skip Block 3 — use if Demucs not available |
| `--skip_landmarks` | Skip Block 4 — faster, falls back to emotion scan |
| `--bbox X Y W H` | Override facecam detection with a fixed region |
| `--video path` | Process a single video file |

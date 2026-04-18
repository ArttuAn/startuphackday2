# Streamer Dataset Builder

Scrapes YouTube and Twitch for Signalis streamer videos, filters them, downloads the best candidates, and structures them into a training dataset.

## Setup

```bash
pip install -r requirements.txt
# Also requires ffmpeg on PATH: winget install ffmpeg
```

## Usage

```bash
# Search + download + process 50 videos
python main.py --max_videos 50 --download --process

# Custom query, minimum 30 min, 100 videos
python main.py -q "Signalis stream facecam" --min_duration 1800 --max_videos 100 --download --process
```

## Pipeline

| Step | What it does |
|------|-------------|
| Search | Queries YouTube via yt-dlp for Signalis streams/playthroughs |
| Filter | Rejects short clips (<10 min), ranks by facecam keywords and duration |
| Download | Parallel yt-dlp downloads (lowest quality mp4, 3 workers) |
| Process | Extracts 16 kHz mono audio, samples 1 fps frames, runs VAD |
| Organize | Writes `dataset/video_<id>/raw.mp4 + audio.wav + metadata.json` |

## Output

```
data/raw_videos/<video_id>/
    raw.mp4
    audio.wav
dataset/video_<id>/
    raw.mp4          # symlink
    audio.wav        # symlink
    metadata.json    # title, url, duration, channel, speech ratio
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max_videos` | 50 | Max videos to collect |
| `--min_duration` | 600 | Min duration in seconds |
| `--download` | off | Run download step |
| `--process` | off | Run processing + VAD step |
| `--workers` | 3 | Parallel download threads |

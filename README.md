# Streamer Reaction Model

An end-to-end pipeline that scrapes streamer gameplay footage, extracts rich emotional signals from face and audio, trains a model to predict streamer reactions from gameplay visuals, and evaluates realism on unseen games.

Built for the StartupHack AI challenge: *train an AI model that infers synthetic streamer reactions to video games.*

---

## Pipeline overview

```
YouTube/Twitch (Signalis VODs)
        │
        ▼
1. streamer-dataset-builder   scrape & download videos
        │
        ▼
2. face-validator             remove low-quality / no-face / VTuber videos
        │
        ▼
3. streamer-separator         detect facecam bounding box per video
        │
        ▼
4. dataset-assembler/assemble.py       organise raw.mp4 + bbox + audio
        │
        ▼
5. dataset-assembler/classify_emotions.py   sample 1 frame / 5s → face + audio signals
        │
        ▼
6. dataset-assembler/extract_clips.py  cut paired gameplay / facecam clips
        │
        ▼
7. model/train.py             train SigLIP + MLP emotion predictor
        │
        ▼
8. model/generate.py          build cosine-similarity retrieval index
        │
        ▼
9. model/evaluate.py          emotion consistency + FID on cross-game footage
```

---

## What signals are captured per clip

Each `meta.json` (written by step 6) contains:

| Field | Description |
|---|---|
| `emotion` | Dominant emotion label (10 classes) |
| `scores` | Per-emotion scores 0–100 |
| `arousal` | Facial activation intensity 0–1 |
| `valence` | Affective tone −1 (unpleasant) … +1 (pleasant) |
| `surprise` | Sudden widening of eyes/brows/mouth 0–1 |
| `tension` | Brow furrow + lip press + squint 0–1 |
| `blendshapes` | All 52 raw MediaPipe expression coefficients |
| `head_pose` | `{yaw, pitch, roll}` in degrees |
| `head_motion` | Nose-tip displacement since previous 5 s sample |
| `audio.loudness` | RMS energy 0–100 |
| `audio.pitch_mean` | Mean F0 in Hz |
| `audio.pitch_var` | Pitch variability |
| `audio.speech` | Fraction of window with speech activity |
| `audio.audio_emotion` | `silence / quiet / speaking / excited / laughing / gasp / shouting` |
| `caption` | Nearest auto-caption text |

**Emotion classes:** neutral, happy, excited, sad, angry, fear, surprise, disgust, contempt, confused

---

## Model

- **Encoder:** SigLIP `google/siglip-base-patch16-224` (frozen) → 768-dim patch embeddings
- **Head:** `Linear(768→256) → GELU → Dropout → Linear(256→10) → Softmax`
- **Loss:** Class-weighted focal loss (handles imbalanced emotion distribution)
- **Input:** 4 evenly-spaced frames from a 5 s gameplay clip
- **Output:** Soft probability distribution over 10 emotion classes
- **Trainable params:** ~200 K (head only)

### Generation (step 8–9)

Given new gameplay footage the model predicts an emotion distribution, then retrieves the nearest-matching facecam clip from the index using cosine similarity on that distribution vector. The retrieved clip is composited onto the gameplay frame via ffmpeg.

---

## Quick start

### Requirements

```bash
pip install -r requirements.txt
# System: ffmpeg  (winget install ffmpeg  /  apt install ffmpeg)
```

### Run the full pipeline

```bash
python run_pipeline.py                         # 50 videos, mediapipe backend
python run_pipeline.py --max_videos 25
python run_pipeline.py --emotion_backend deepface
```

### Skip steps you've already done

```bash
# Re-run only classification + clips + training
python run_pipeline.py --skip_scrape --skip_validate --skip_bbox --skip_assemble

# Re-run only training, index, evaluation
python run_pipeline.py --skip_scrape --skip_validate --skip_bbox --skip_assemble --skip_emotions --skip_clips

# Re-run only steps 7–9
python run_pipeline.py --skip_scrape --skip_validate --skip_bbox --skip_assemble --skip_emotions --skip_clips
```

### All flags

| Flag | Skips |
|---|---|
| `--skip_scrape` | Step 1 — scrape & download |
| `--skip_validate` | Step 2 — face validation |
| `--skip_bbox` | Step 3 — bbox detection |
| `--skip_assemble` | Step 4 — dataset assembly |
| `--skip_emotions` | Step 5 — emotion classification |
| `--skip_clips` | Step 6 — clip extraction |
| `--skip_train` | Step 7 — model training |
| `--skip_index` | Step 8 — retrieval index |
| `--skip_eval` | Step 9 — evaluation |
| `--emotion_backend` | `mediapipe` (default) / `fer` / `deepface` |
| `--max_videos N` | Limit scraping to N videos |
| `--shorts_only` | Scrape only YouTube Shorts |

---

## Emotion backends

| Backend | Install | Notes |
|---|---|---|
| `mediapipe` | included in requirements.txt | 52 blendshape coefficients → custom 10-class mapping. Fast, no GPU needed. |
| `fer` | `pip install fer` | CNN trained on FER2013. 7 standard classes. |
| `deepface` | `pip install deepface tf-keras` | Ensemble model. Most accurate, slowest. |

---

## WandB tracking

Pass your API key to track training:

```bash
python model/train.py --wandb_key YOUR_KEY
```

Logged: per-batch focal loss, per-epoch top-1/top-3/macro-F1, confusion matrix, dataset split artifact, gameplay+facecam frame pairs, class weights.

---

## Cross-game evaluation

Step 9 evaluates emotion consistency on footage from *Little Nightmares* and *Hollow Knight* (scraped separately by `eval-scraper/scrape_eval_games.py`). FID is available but skipped by default (`--skip_fid`).

---

## Directory structure

```
run_pipeline.py                 — pipeline orchestrator
requirements.txt
streamer-dataset-builder/       — YouTube scraper
face-validator/                 — face presence + quality filter
streamer-separator/             — facecam bbox detection
dataset-assembler/
    assemble.py
    classify_emotions.py        — face + audio signal extraction
    extract_clips.py            — paired clip cutter
    dataset/                    — per-video data (gitignored)
model/
    train.py                    — SigLIP + MLP training
    generate.py                 — retrieval index + compositing
    evaluate.py                 — emotion consistency + FID
    dataset.py                  — PyTorch dataset loader
    checkpoints/                — saved model weights
eval-scraper/
    scrape_eval_games.py        — scraper for cross-game eval footage
```

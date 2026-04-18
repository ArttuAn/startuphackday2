"""
classify_emotions.py — Classify emotion every second of each facecam.mp4.

Uses MediaPipe FaceLandmarker blendshapes (52 expression coefficients) to
score 10 emotions per frame. No TensorFlow required.

10 emotions:
    neutral, happy, excited, sad, angry, fear, surprise, disgust, contempt, confused

Output per video:
    dataset/<video_id>/emotions.json   — [{t, emotion, scores}, ...]

Usage:
    python classify_emotions.py
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path

import cv2
from tqdm import tqdm

from config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# MediaPipe model (shared with separator if already downloaded)
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

# Also check separator folder for existing model download
_SEPARATOR_MODEL = Path(__file__).parent.parent / "streamer-separator" / "face_landmarker.task"


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    if _SEPARATOR_MODEL.exists():
        log.info("Using model from streamer-separator folder")
        return _SEPARATOR_MODEL
    log.info("Downloading MediaPipe FaceLandmarker model (~30 MB)...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    log.info("Model downloaded.")
    return _MODEL_PATH


# ── 10-emotion blendshape mapping ──────────────────────────────────────────

def blendshapes_to_emotions(bs: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Map MediaPipe blendshape scores (0–1) to 10 emotion scores (0–100)."""

    # Component signals
    brow_up      = (bs.get("browInnerUp", 0) +
                    bs.get("browOuterUpLeft", 0) +
                    bs.get("browOuterUpRight", 0)) / 3
    brow_down    = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    brow_asym    = abs(bs.get("browDownLeft", 0) - bs.get("browDownRight", 0))
    eye_wide     = (bs.get("eyeWideLeft", 0)   + bs.get("eyeWideRight", 0)) / 2
    eye_squint   = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
    smile        = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    smile_asym   = abs(bs.get("mouthSmileLeft", 0) - bs.get("mouthSmileRight", 0))
    frown        = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    mouth_open   = bs.get("jawOpen", 0)
    mouth_str    = (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2
    nose_sneer   = (bs.get("noseSneerLeft", 0)  + bs.get("noseSneerRight", 0)) / 2
    mouth_press  = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
    cheek_puff   = bs.get("cheekPuff", 0)

    scores = {
        "neutral":  round(max(0.0, 100 - brow_up*60 - eye_wide*40 - smile*40
                               - frown*30 - brow_down*30 - nose_sneer*40), 2),
        "happy":    round(min(100, (smile*0.7 + mouth_open*0.2 + cheek_puff*0.1) * 100), 2),
        "excited":  round(min(100, (smile*0.5 + mouth_open*0.3 + brow_up*0.2) * 100), 2),
        "sad":      round(min(100, (frown*0.5 + brow_down*0.3 + (1-smile)*0.2) * 100), 2),
        "angry":    round(min(100, (brow_down*0.5 + mouth_press*0.3 + (1-smile)*0.2) * 100), 2),
        "fear":     round(min(100, (brow_up*0.4 + eye_wide*0.3 + mouth_str*0.3) * 100), 2),
        "surprise": round(min(100, (brow_up*0.35 + eye_wide*0.35 + mouth_open*0.3) * 100), 2),
        "disgust":  round(min(100, (nose_sneer*0.6 + eye_squint*0.3 + frown*0.1) * 100), 2),
        "contempt": round(min(100, (smile_asym*0.6 + mouth_press*0.2 + eye_squint*0.2) * 100), 2),
        "confused": round(min(100, (brow_asym*0.5  + brow_down*0.3  + (1-mouth_open)*0.2) * 100), 2),
    }

    dominant = max(scores, key=lambda k: scores[k])
    return dominant, scores


# ── Per-video classifier ───────────────────────────────────────────────────

def classify_video(facecam_path: Path, out_path: Path) -> bool:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.2,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap     = cv2.VideoCapture(str(facecam_path))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, int(fps * 60))     # 1 frame per minute (PROTOTYPE — change to int(fps) for 1/sec)
    n_frames = total_f // step

    # Folder for per-frame screenshots
    frames_dir = out_path.parent / "emotion_frames"
    frames_dir.mkdir(exist_ok=True)

    timeline = []
    frame_idx = 0

    with tqdm(total=n_frames, unit="s", desc=facecam_path.parent.name,
              dynamic_ncols=True) as pbar:
        while frame_idx < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            t = round(frame_idx / fps, 2)
            emotion, scores = "neutral", {"neutral": 100.0}

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                emotion, scores = blendshapes_to_emotions(bs)
            else:
                emotion, scores = "none", {}

            timeline.append({"t": t, "emotion": emotion, "scores": scores})

            # Save high-quality screenshot — upscale to at least 512px wide
            frame_name = f"{int(t):06d}s_{emotion}.jpg"
            fh, fw = frame.shape[:2]
            if fw < 512:
                scale = 512 / fw
                frame_out = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                       interpolation=cv2.INTER_LANCZOS4)
            else:
                frame_out = frame
            cv2.imwrite(str(frames_dir / frame_name), frame_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            pbar.set_postfix(t=f"{t:.0f}s", emotion=emotion)
            pbar.update(1)
            frame_idx += step

    cap.release()
    landmarker.close()

    out_path.write_text(json.dumps(timeline, indent=2))
    no_face   = sum(1 for e in timeline if e["emotion"] == "none")
    reactions = sum(1 for e in timeline if e["emotion"] not in ("neutral", "none"))
    log.info("  ✓ %d frames — %d reactions, %d no-face (%.0f%%)",
             len(timeline), reactions, no_face,
             100 * no_face / len(timeline) if timeline else 0)
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    try:
        import mediapipe
    except ImportError:
        log.error("mediapipe not installed — run: pip install mediapipe")
        sys.exit(1)

    vid_dirs = sorted(d for d in DATASET_DIR.iterdir()
                      if d.is_dir() and (d / "facecam.mp4").exists())

    if not vid_dirs:
        log.error("No facecam.mp4 files found — run split_videos.py first")
        sys.exit(1)

    log.warning("⚠️  PROTOTYPE MODE: sampling 1 frame/minute. Change step to int(fps) in classify_video() for 1 frame/second before production.")
    log.info("Classifying emotions for %d videos...", len(vid_dirs))
    ok = 0

    for i, vid_dir in enumerate(vid_dirs, 1):
        log.info("[%d/%d] %s", i, len(vid_dirs), vid_dir.name)
        out_path = vid_dir / "emotions.json"


        if classify_video(vid_dir / "facecam.mp4", out_path):
            ok += 1

    log.info("Done — %d/%d videos classified", ok, len(vid_dirs))


if __name__ == "__main__":
    main()

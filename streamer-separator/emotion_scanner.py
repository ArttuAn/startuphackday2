"""
emotion_scanner.py — Scan a full video and build a raw emotion timeline.

Uses MediaPipe FaceLandmarker blendshapes (52 expression coefficients)
to infer emotion — no TensorFlow/DeepFace required.

Blendshape-to-emotion mapping:
  fear/surprise → browInnerUp + eyeWideLeft/Right high
  happy         → mouthSmileLeft/Right high
  sad           → browDownLeft/Right + mouthFrownLeft/Right high
  angry         → browDownLeft/Right high, no smile
  neutral       → everything below threshold
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import EMOTION_CONFIDENCE_MIN, REACTION_EMOTIONS, SCAN_FPS
from detector import BBox

log = logging.getLogger("emotion_scanner")

# MediaPipe FaceLandmarker model (downloaded once on first run)
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        log.info("Downloading MediaPipe FaceLandmarker model (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        log.info("Model downloaded.")
    return _MODEL_PATH


# ── Blendshape → emotion ───────────────────────────────────────────────────

def _blendshapes_to_emotion(bs: dict[str, float]) -> tuple[str, dict[str, float]]:
    """
    Map 52 MediaPipe blendshape scores to a dominant emotion + score dict.
    All scores are 0.0–1.0.
    """
    brow_up    = (bs.get("browInnerUp", 0) +
                  bs.get("browOuterUpLeft", 0) +
                  bs.get("browOuterUpRight", 0)) / 3
    eye_wide   = (bs.get("eyeWideLeft",  0) + bs.get("eyeWideRight", 0)) / 2
    brow_down  = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    smile      = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown      = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    mouth_open = bs.get("jawOpen", 0)
    mouth_str  = (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2

    scores = {
        "fear":     round(min(1.0, brow_up * 0.5 + eye_wide * 0.3 + mouth_str * 0.2) * 100, 2),
        "surprise": round(min(1.0, brow_up * 0.4 + eye_wide * 0.4 + mouth_open * 0.2) * 100, 2),
        "happy":    round(min(1.0, smile * 0.8 + mouth_open * 0.2) * 100, 2),
        "sad":      round(min(1.0, frown * 0.5 + brow_down * 0.3 + (1 - smile) * 0.2) * 100, 2),
        "angry":    round(min(1.0, brow_down * 0.6 + (1 - smile) * 0.4) * 100, 2),
        "neutral":  round(max(0.0, 100 - brow_up * 60 - eye_wide * 40 - smile * 40 - frown * 30), 2),
    }
    dominant = max(scores, key=lambda k: scores[k])
    return dominant, scores


# ── EmotionSample ──────────────────────────────────────────────────────────

@dataclass
class EmotionSample:
    t:       float
    emotion: str
    scores:  dict[str, float] = field(default_factory=dict)

    def is_reaction(self) -> bool:
        return (
            self.emotion in REACTION_EMOTIONS
            and self.scores.get(self.emotion, 0.0) >= EMOTION_CONFIDENCE_MIN
        )

    def to_dict(self) -> dict:
        return {
            "t":       round(self.t, 3),
            "emotion": self.emotion,
            "scores":  {k: round(v, 2) for k, v in self.scores.items()},
        }


# ── Main scanner ───────────────────────────────────────────────────────────

def scan_emotions(video_path: Path, bbox: BBox) -> list[EmotionSample]:
    """
    Scan the full video and return an emotion timeline using MediaPipe blendshapes.
    Results are cached by main.py — this only runs once per video.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model()

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.3,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step     = max(1, int(fps / SCAN_FPS))
    n_frames = total_f // step

    log.info("Scanning %.0f min video @ %.2f fps → %d frames",
             (total_f / fps) / 60, SCAN_FPS, n_frames)

    timeline: list[EmotionSample] = []
    frame_idx = 0

    with tqdm(total=n_frames, unit="frame", desc="Emotion scan", dynamic_ncols=True) as pbar:
        while frame_idx < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            t = round(frame_idx / fps, 3)
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            crop = frame[y:y+h, x:x+w]

            emotion, scores = "neutral", {"neutral": 100.0}

            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                if result.face_blendshapes:
                    bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                    emotion, scores = _blendshapes_to_emotion(bs)

            sample = EmotionSample(t=t, emotion=emotion, scores=scores)
            timeline.append(sample)
            pbar.set_postfix(t=f"{t:.0f}s", emotion=emotion)
            pbar.update(1)
            frame_idx += step

    cap.release()
    landmarker.close()

    reactions = sum(1 for s in timeline if s.is_reaction())
    log.info("Scan complete — %d samples, %d reactions (%.0f%%)",
             len(timeline), reactions,
             100 * reactions / len(timeline) if timeline else 0)
    return timeline

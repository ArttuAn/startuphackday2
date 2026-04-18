"""
Block 4 — landmark_extractor.py

Extracts per-frame facial structure from the facecam crop using MediaPipe
FaceLandmarker — no TensorFlow required.

Per frame:
  - 478 3D face landmarks (normalised 0–1)
  - 52 blendshape expression coefficients
  - Head pose: pitch, yaw, roll (degrees)
  - Derived emotion label + scores (from blendshapes)

Output: landmarks.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import SCAN_FPS
from detector import BBox
from emotion_scanner import _blendshapes_to_emotion, _ensure_model

log = logging.getLogger("landmark_extractor")

LANDMARK_FPS = SCAN_FPS


# ── Head pose from landmarks ───────────────────────────────────────────────

_NOSE_TIP   = 1
_CHIN       = 199
_LEFT_EYE   = 33
_RIGHT_EYE  = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH= 291

def _estimate_head_pose(
    landmarks: list,
    frame_w: int,
    frame_h: int,
) -> dict[str, float]:
    model_points = np.array([
        [0.0,    0.0,    0.0   ],
        [0.0,   -330.0, -65.0  ],
        [-225.0, 170.0, -135.0 ],
        [225.0,  170.0, -135.0 ],
        [-150.0,-150.0, -125.0 ],
        [150.0, -150.0, -125.0 ],
    ], dtype=np.float64)

    idxs = [_NOSE_TIP, _CHIN, _LEFT_EYE, _RIGHT_EYE, _LEFT_MOUTH, _RIGHT_MOUTH]
    image_points = np.array(
        [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in idxs],
        dtype=np.float64,
    )

    focal = float(frame_w)
    cam   = np.array([[focal,0,frame_w/2],[0,focal,frame_h/2],[0,0,1]], dtype=np.float64)
    dist  = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(model_points, image_points, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rmat[2,1], rmat[2,2]))
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))
        roll  = math.degrees(math.atan2( rmat[1,0], rmat[0,0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))
        roll  = 0.0

    return {"pitch": round(pitch,2), "yaw": round(yaw,2), "roll": round(roll,2)}


# ── Public API ─────────────────────────────────────────────────────────────

def extract_landmarks(
    video_path: Path,
    bbox:       BBox,
    out_dir:    Path,
    force:      bool = False,
) -> Path | None:
    """
    Extract per-frame landmarks, blendshapes, head pose, and emotion.
    Writes landmarks.json. Returns path or None on failure.
    """
    out_path = out_dir / "landmarks.json"
    if not force and out_path.exists():
        log.info("Landmarks already exist for %s — skipping", out_dir.name)
        return out_path

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

    out_dir.mkdir(parents=True, exist_ok=True)
    cap     = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, int(src_fps / LANDMARK_FPS))
    n_frames= total_f // step

    log.info("Extracting landmarks from %s (~%d frames)", out_dir.name, n_frames)

    records   = []
    frame_idx = 0

    with tqdm(total=n_frames, unit="frame", desc="Landmarks", dynamic_ncols=True) as pbar:
        while frame_idx < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            t = round(frame_idx / src_fps, 3)
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            crop = frame[y:y+h, x:x+w]

            record: dict = {"frame": frame_idx, "t": t,
                            "landmarks": [], "blendshapes": {},
                            "head_pose": {"pitch":0.0,"yaw":0.0,"roll":0.0},
                            "emotion": "neutral",
                            "emotion_scores": {"neutral": 100.0}}

            if crop.size > 0:
                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    record["landmarks"] = [[round(p.x,5), round(p.y,5), round(p.z,5)]
                                           for p in lm]
                    record["head_pose"] = _estimate_head_pose(lm, w, h)

                if result.face_blendshapes:
                    bs = {c.category_name: round(c.score, 4)
                          for c in result.face_blendshapes[0]}
                    record["blendshapes"] = bs
                    emotion, scores       = _blendshapes_to_emotion(bs)
                    record["emotion"]        = emotion
                    record["emotion_scores"] = scores

            records.append(record)
            pbar.set_postfix(t=f"{t:.0f}s", emotion=record["emotion"],
                             lm="✓" if record["landmarks"] else "✗")
            frame_idx += step
            pbar.update(1)

    cap.release()
    landmarker.close()

    out_path.write_text(json.dumps(records, indent=2))
    log.info("✓ landmarks.json: %d frames, %d with face",
             len(records), sum(1 for r in records if r["landmarks"]))
    return out_path


def load_landmarks(out_dir: Path) -> list[dict] | None:
    path = out_dir / "landmarks.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

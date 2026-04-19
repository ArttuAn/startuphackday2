"""
classify_emotions.py — Classify emotion every minute (prototype) from face + audio.

Face: MediaPipe FaceLandmarker blendshapes → 10 emotion scores
Audio: 5-second window around each timestamp → loudness, pitch, speech energy

Output per video:
    dataset/<video_id>/emotions.json   — [{t, emotion, scores, audio}, ...]
    dataset/<video_id>/emotion_frames/ — face screenshot per timestamp

After classification, each video is checked for emotional expressiveness:
    MIN_REACTION_RATE — fraction of frames that must be non-neutral/non-none.
    Videos below this threshold are deleted from the dataset (too boring to
    train on — streamer barely reacts).

⚠️  PROTOTYPE MODE: 1 frame/minute. Change step to int(fps) for 1 frame/second.

Usage:
    python classify_emotions.py
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import DATASET_DIR
from parse_captions import load_captions, nearest_caption


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

_MODEL_URL       = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_PATH      = Path(__file__).parent / "face_landmarker.task"
_SEPARATOR_MODEL = Path(__file__).parent.parent / "streamer-separator" / "face_landmarker.task"

AUDIO_CLIP_SEC = 5  # seconds of audio to analyse around each timestamp


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    if _SEPARATOR_MODEL.exists():
        log.info("Using model from streamer-separator folder")
        return _SEPARATOR_MODEL
    log.info("Downloading MediaPipe FaceLandmarker model (~30 MB)...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


# ── Audio loader ───────────────────────────────────────────────────────────

def load_audio(wav_path: Path) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Load mono 16-bit PCM wav. Returns (samples float32, sample_rate)."""
    try:
        import wave, array
        with wave.open(str(wav_path), "rb") as wf:
            sr        = wf.getframerate()
            n_frames  = wf.getnframes()
            raw       = wf.readframes(n_frames)
            samples   = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr
    except Exception as e:
        log.warning("  Could not load audio: %s", e)
        return None, None


# ── Audio feature extraction ───────────────────────────────────────────────

def analyse_audio_window(samples: np.ndarray, sr: int, t: float) -> dict:
    """
    Extract features from a AUDIO_CLIP_SEC window centred on t.

    Returns dict with:
        loudness      — RMS energy 0–100 (higher = louder)
        pitch_mean    — mean F0 estimate in Hz (0 if no speech)
        pitch_var     — pitch variability (higher = more expressive)
        speech        — fraction of window with speech activity (0–1)
        audio_emotion — coarse label: silent / quiet / speaking / excited / shouting
    """
    half   = int(sr * AUDIO_CLIP_SEC / 2)
    centre = int(t * sr)
    start  = max(0, centre - half)
    end    = min(len(samples), centre + half)
    clip   = samples[start:end]

    if clip.size == 0:
        return {"loudness": 0, "pitch_mean": 0, "pitch_var": 0,
                "speech": 0, "audio_emotion": "silent"}

    # ── Loudness (RMS → 0–100) ─────────────────────────────────────────────
    rms      = float(np.sqrt(np.mean(clip ** 2)))
    loudness = round(min(100.0, rms * 500), 2)   # scale: 0.2 RMS ≈ 100

    # ── Speech activity (zero-crossing rate heuristic) ────────────────────
    frame_len  = sr // 10   # 100 ms frames
    zcrs, rms_frames = [], []
    for i in range(0, len(clip) - frame_len, frame_len):
        f = clip[i:i+frame_len]
        zcrs.append(float(np.mean(np.abs(np.diff(np.sign(f))))) / 2)
        rms_frames.append(float(np.sqrt(np.mean(f ** 2))))

    rms_arr  = np.array(rms_frames)
    zcr_arr  = np.array(zcrs)
    speech_threshold = max(0.005, np.mean(rms_arr) * 0.5)
    speech_frames    = rms_arr > speech_threshold
    speech_ratio     = round(float(np.mean(speech_frames)), 3)

    # ── Pitch estimate (autocorrelation on speech frames) ─────────────────
    pitch_values = []
    for i, is_speech in enumerate(speech_frames):
        if not is_speech:
            continue
        f = clip[i*frame_len:(i+1)*frame_len]
        # Autocorrelation
        corr = np.correlate(f, f, mode="full")
        corr = corr[len(corr)//2:]
        # Look for first peak in 80–400 Hz range
        lo = int(sr / 400)
        hi = int(sr / 80)
        if hi < len(corr):
            peak = np.argmax(corr[lo:hi]) + lo
            if corr[peak] > 0.1:
                pitch_values.append(sr / peak)

    pitch_mean = round(float(np.mean(pitch_values)) if pitch_values else 0, 1)
    pitch_var  = round(float(np.std(pitch_values))  if pitch_values else 0, 1)

    # ── Coarse audio emotion label ─────────────────────────────────────────
    if loudness < 5:
        audio_emotion = "silence"
    elif loudness < 20 and speech_ratio < 0.3:
        audio_emotion = "quiet"
    elif loudness > 45 and pitch_var > 55 and speech_ratio > 0.4:
        audio_emotion = "laughing"      # rhythmic energy + high pitch variability
    elif loudness > 45 and speech_ratio < 0.25:
        audio_emotion = "gasp"          # energy burst with little sustained speech
    elif loudness > 70:
        audio_emotion = "shouting"
    elif loudness > 40 and pitch_var > 30:
        audio_emotion = "excited"
    else:
        audio_emotion = "speaking"

    return {
        "loudness":     loudness,
        "pitch_mean":   pitch_mean,
        "pitch_var":    pitch_var,
        "speech":       speech_ratio,
        "audio_emotion": audio_emotion,
    }


# ── Arousal / valence / head pose helpers ─────────────────────────────────

def compute_arousal_valence(bs: dict) -> tuple[float, float]:
    """
    Derive arousal (0–1) and valence (−1..+1) from MediaPipe blendshapes.

    Arousal  — how activated/intense the face is (0 = neutral rest).
    Valence  — affective tone (+1 = very pleasant, −1 = very unpleasant).
    """
    eye_wide      = (bs.get("eyeWideLeft",0)         + bs.get("eyeWideRight",0)) / 2
    brow_up       = (bs.get("browInnerUp",0)          +
                     bs.get("browOuterUpLeft",0)       +
                     bs.get("browOuterUpRight",0)) / 3
    brow_down     = (bs.get("browDownLeft",0)         + bs.get("browDownRight",0)) / 2
    jaw_open      = bs.get("jawOpen", 0)
    smile         = (bs.get("mouthSmileLeft",0)       + bs.get("mouthSmileRight",0)) / 2
    frown         = (bs.get("mouthFrownLeft",0)       + bs.get("mouthFrownRight",0)) / 2
    nose_sneer    = (bs.get("noseSneerLeft",0)        + bs.get("noseSneerRight",0)) / 2
    mouth_stretch = (bs.get("mouthStretchLeft",0)     + bs.get("mouthStretchRight",0)) / 2

    arousal = min(1.0,
                  eye_wide * 0.20 + brow_up * 0.20 + brow_down * 0.15 +
                  jaw_open * 0.25 + mouth_stretch * 0.20)

    valence = smile * 0.6 - frown * 0.3 - nose_sneer * 0.2 - brow_down * 0.1
    valence = max(-1.0, min(1.0, valence * 2.0))

    return round(arousal, 3), round(valence, 3)


def compute_surprise_tension(bs: dict) -> tuple[float, float]:
    """
    Surprise (0–1): sudden widening of eyes + brows + mouth.
    Tension  (0–1): compressed, braced face — brow furrow + lip press + squint.
    """
    eye_wide   = (bs.get("eyeWideLeft",0)   + bs.get("eyeWideRight",0)) / 2
    brow_up    = (bs.get("browInnerUp",0)   + bs.get("browOuterUpLeft",0) +
                  bs.get("browOuterUpRight",0)) / 3
    jaw_open   = bs.get("jawOpen", 0)
    brow_down  = (bs.get("browDownLeft",0)  + bs.get("browDownRight",0)) / 2
    mouth_press = (bs.get("mouthPressLeft",0) + bs.get("mouthPressRight",0)) / 2
    eye_squint = (bs.get("eyeSquintLeft",0) + bs.get("eyeSquintRight",0)) / 2

    surprise = min(1.0, eye_wide * 0.35 + brow_up * 0.35 + jaw_open * 0.30)
    tension  = min(1.0, brow_down * 0.40 + mouth_press * 0.35 + eye_squint * 0.25)

    return round(surprise, 3), round(tension, 3)


def arousal_valence_from_scores(scores: dict) -> tuple[float, float]:
    """
    Fallback for FER / DeepFace backends — estimate arousal/valence
    from the discrete emotion probability scores (0–100).
    """
    total = sum(scores.values())
    if total < 1e-6:
        return 0.0, 0.0
    n = {k: v / 100.0 for k, v in scores.items()}
    arousal = round(1.0 - n.get("neutral", 0), 3)
    positive = n.get("happy", 0) + n.get("excited", 0) * 0.7 + n.get("surprise", 0) * 0.3
    negative = (n.get("sad", 0) + n.get("angry", 0) + n.get("fear", 0) +
                n.get("disgust", 0) + n.get("contempt", 0) * 0.5)
    valence  = round(max(-1.0, min(1.0, (positive - negative) * 2.0)), 3)
    return arousal, valence


def extract_head_pose(mat) -> dict:
    """
    Extract yaw / pitch / roll (degrees) from a 4×4 facial-transformation matrix
    returned by MediaPipe FaceLandmarker.
    """
    import math
    try:
        pitch = math.atan2(-mat[2][0],
                           math.sqrt(mat[2][1] ** 2 + mat[2][2] ** 2))
        yaw   = math.atan2(mat[1][0], mat[0][0])
        roll  = math.atan2(mat[2][1], mat[2][2])
        return {
            "yaw":   round(math.degrees(yaw),   1),
            "pitch": round(math.degrees(pitch), 1),
            "roll":  round(math.degrees(roll),  1),
        }
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


# ── Alternative emotion backends ──────────────────────────────────────────

def fer_emotions(face_crop_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
    """
    FER backend — proper CNN trained on FER2013.
    pip install fer
    Returns (dominant_emotion, scores_0_to_100).
    Emotions: angry, disgust, fear, happy, sad, surprise, neutral
    """
    try:
        from fer import FER
    except ImportError:
        raise ImportError("pip install fer")

    detector = FER(mtcnn=False)   # mtcnn=False = faster, uses OpenCV face detect
    result   = detector.detect_emotions(face_crop_bgr)

    if not result:
        return "none", {}

    emotions = result[0]["emotions"]   # dict e.g. {"happy": 0.92, "neutral": 0.05 ...}
    scores   = {k: round(v * 100, 2) for k, v in emotions.items()}
    dominant = max(scores, key=lambda k: scores[k])
    return dominant, scores


def deepface_emotions(face_crop_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
    """
    DeepFace backend — ensemble of pretrained models.
    pip install deepface
    Returns (dominant_emotion, scores_0_to_100).
    Emotions: angry, disgust, fear, happy, sad, surprise, neutral
    """
    try:
        from deepface import DeepFace
    except ImportError:
        raise ImportError("pip install deepface")

    try:
        result = DeepFace.analyze(
            face_crop_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        emotions = result["emotion"]   # already 0–100
        scores   = {k: round(float(v), 2) for k, v in emotions.items()}

        # Post-process: suppress angry/sad when happy score is significant
        happy = scores.get("happy", 0)
        if happy > 20:
            suppress = happy / 100.0
            for label in ("angry", "sad", "disgust"):
                if label in scores:
                    scores[label] = round(scores[label] * (1.0 - suppress), 2)

        # Recompute dominant after post-processing
        dominant = max(scores, key=lambda k: scores[k])
        return dominant, scores
    except Exception as e:
        log.debug("deepface error: %s", e)
        return "none", {}


# ── 10-emotion blendshape mapping ──────────────────────────────────────────

def blendshapes_to_emotions(bs: dict[str, float]) -> tuple[str, dict[str, float]]:
    brow_up     = (bs.get("browInnerUp", 0) +
                   bs.get("browOuterUpLeft", 0) +
                   bs.get("browOuterUpRight", 0)) / 3
    brow_down   = (bs.get("browDownLeft", 0)  + bs.get("browDownRight", 0)) / 2
    brow_asym   = abs(bs.get("browDownLeft", 0) - bs.get("browDownRight", 0))
    eye_wide    = (bs.get("eyeWideLeft", 0)   + bs.get("eyeWideRight", 0)) / 2
    eye_squint  = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
    smile       = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    smile_asym  = abs(bs.get("mouthSmileLeft", 0) - bs.get("mouthSmileRight", 0))
    frown       = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    mouth_open  = bs.get("jawOpen", 0)
    mouth_str   = (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2
    nose_sneer  = (bs.get("noseSneerLeft", 0)  + bs.get("noseSneerRight", 0)) / 2
    mouth_press = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
    cheek_puff  = bs.get("cheekPuff", 0)

    # Angry guard: suppress if face is smiling or mouth is open (surprise/happy)
    angry_raw = brow_down * 0.5 + (nose_sneer + mouth_press) * 0.5
    angry_val = angry_raw * max(0.0, 1.0 - smile * 3.0) * max(0.0, 1.0 - mouth_open * 2.0)

    # Sad guard: suppress if smiling
    sad_raw = frown * 0.6 + brow_down * 0.4
    sad_val = sad_raw * max(0.0, 1.0 - smile * 3.0)

    scores = {
        "neutral":  round(max(0.0, 100 - brow_up*60 - eye_wide*40 - smile*40
                              - frown*30 - brow_down*30 - nose_sneer*40), 2),
        "happy":    round(min(100, (smile*0.7 + mouth_open*0.2 + cheek_puff*0.1) * 100), 2),
        "excited":  round(min(100, (smile*0.5 + mouth_open*0.3 + brow_up*0.2) * 100), 2),
        "sad":      round(min(100, sad_val * 100), 2),
        "angry":    round(min(100, angry_val * 100), 2),
        "fear":     round(min(100, (brow_up*0.4 + eye_wide*0.3 + mouth_str*0.3) * 100), 2),
        "surprise": round(min(100, (brow_up*0.35 + eye_wide*0.35 + mouth_open*0.3) * 100), 2),
        "disgust":  round(min(100, (nose_sneer*0.6 + eye_squint*0.3 + frown*0.1) * 100), 2),
        "contempt": round(min(100, (smile_asym*0.6 + mouth_press*0.2 + eye_squint*0.2) * 100), 2),
        "confused": round(min(100, (brow_asym*0.5  + brow_down*0.3  + (1-mouth_open)*0.2) * 100), 2),
    }
    # Pick dominant: if any reaction emotion exceeds threshold, prefer it over neutral.
    # (Neutral starts at 100 and always wins argmax — use threshold instead.)
    REACTION_THRESHOLD = 20.0
    reaction_scores = {k: v for k, v in scores.items() if k != "neutral"}
    best_reaction = max(reaction_scores, key=lambda k: reaction_scores[k])
    if reaction_scores[best_reaction] >= REACTION_THRESHOLD:
        dominant = best_reaction
    else:
        dominant = "neutral"
    return dominant, scores


# ── Per-video classifier ───────────────────────────────────────────────────

def classify_video(vid_dir: Path, out_path: Path,
                   backend: str = "mediapipe") -> bool:
    """
    backend: "mediapipe" (blendshapes) | "fer" (FER2013 CNN) | "deepface"
    """
    if backend == "mediapipe":
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

    facecam_path = vid_dir / "facecam.mp4" if (vid_dir / "facecam.mp4").exists() else vid_dir / "raw.mp4"
    audio_path   = vid_dir / "audio.wav"

    # Read bbox so we can crop the facecam region out of raw.mp4 per-frame.
    # Without this, MediaPipe runs on the full stream frame where the face is
    # a tiny overlay — blendshapes are unreliable and reactions all read neutral.
    bbox_crop: tuple[int, int, int, int] | None = None
    bbox_file = vid_dir / "bbox.txt"
    if bbox_file.exists() and facecam_path.name == "raw.mp4":
        try:
            parts = dict(p.split("=") for p in bbox_file.read_text().split())
            bbox_crop = (int(parts["x"]), int(parts["y"]),
                         int(parts["w"]), int(parts["h"]))
            log.info("  bbox crop: x=%d y=%d w=%d h=%d", *bbox_crop)
        except Exception as e:
            log.warning("  could not parse bbox.txt (%s) — running on full frame", e)

    if backend == "mediapipe":
        model_path = _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.2,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    else:
        landmarker = None   # not used for fer/deepface

    # Load captions (VTT auto-captions downloaded alongside video)
    captions = load_captions(vid_dir)
    has_captions = len(captions) > 0

    # Load audio — extract from raw.mp4 via ffmpeg if audio.wav is missing
    if not audio_path.exists():
        raw_mp4 = vid_dir / "raw.mp4"
        if raw_mp4.exists():
            import subprocess
            log.info("  audio.wav missing — extracting from raw.mp4 via ffmpeg...")
            result = subprocess.run(
                ["ffmpeg", "-i", str(raw_mp4),
                 "-ar", "16000", "-ac", "1", "-y", str(audio_path),
                 "-loglevel", "error"],
                capture_output=True,
            )
            if result.returncode != 0:
                log.warning("  ffmpeg extraction failed: %s", result.stderr.decode()[:200])

    audio_samples, audio_sr = load_audio(audio_path) if audio_path.exists() else (None, None)
    if audio_samples is None:
        log.warning("  no audio available — skipping audio features")

    cap      = cv2.VideoCapture(str(facecam_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step     = max(1, int(fps * 5))    # sample every 5 seconds
    n_frames = total_f // step

    timeline      = []
    frame_idx     = 0
    prev_nose_pos = None   # (x, y) normalised — for head motion energy

    with tqdm(total=n_frames, unit="s", desc=vid_dir.name, dynamic_ncols=True) as pbar:
        while frame_idx < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            t = round(frame_idx / fps, 2)

            # ── Crop to facecam region (avoids running MediaPipe on tiny overlay) ──
            full_frame = frame.copy()   # keep for re-detection fallback
            if bbox_crop is not None:
                bx, by, bw, bh = bbox_crop
                fh_frame, fw_frame = frame.shape[:2]
                bx = max(0, min(bx, fw_frame - 1))
                by = max(0, min(by, fh_frame - 1))
                bw = max(1, min(bw, fw_frame - bx))
                bh = max(1, min(bh, fh_frame - by))
                crop = frame[by:by + bh, bx:bx + bw]

                # ── Dynamic bbox re-detection if crop is too dark ────────
                # Streamer sometimes moves their facecam during long streams.
                # If mean brightness < 15 (nearly black), scan the full frame
                # for a face and update bbox_crop for all future frames.
                if crop.mean() < 15:
                    gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
                    faces = _face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                    if len(faces) > 0:
                        fx, fy, fw2, fh2 = faces[0]
                        # Expand bbox to ~2× face size for full facecam region
                        pad = int(max(fw2, fh2) * 0.6)
                        nx  = max(0, fx - pad)
                        ny  = max(0, fy - pad)
                        nw  = min(fw_frame - nx, fw2 + pad * 2)
                        nh  = min(fh_frame - ny, fh2 + pad * 2)
                        bbox_crop = (nx, ny, nw, nh)
                        log.info("  t=%.0fs: facecam moved — updated bbox to "
                                 "x=%d y=%d w=%d h=%d", t, *bbox_crop)
                        crop = full_frame[ny:ny + nh, nx:nx + nw]

                frame = crop

            # ── Upscale small crops — MediaPipe needs ≥256px for reliable landmarks ──
            MIN_MP_SIZE = 256
            fh_c, fw_c = frame.shape[:2]
            if fw_c < MIN_MP_SIZE or fh_c < MIN_MP_SIZE:
                scale = MIN_MP_SIZE / min(fw_c, fh_c)
                frame = cv2.resize(
                    frame,
                    (int(fw_c * scale), int(fh_c * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # ── Face emotion + richer signals ─────────────────────────────
            emotion, scores  = "none", {}
            blendshapes_dict = {}
            arousal, valence = 0.0, 0.0
            surprise, tension = 0.0, 0.0
            head_pose        = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            head_motion      = 0.0   # displacement of nose tip (normalised px)

            if backend == "mediapipe":
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.face_blendshapes:
                    bs = {c.category_name: c.score
                          for c in result.face_blendshapes[0]}
                    emotion, scores          = blendshapes_to_emotions(bs)
                    arousal, valence         = compute_arousal_valence(bs)
                    surprise, tension        = compute_surprise_tension(bs)
                    blendshapes_dict         = {k: round(v, 4) for k, v in bs.items()}

                # Head pose from transformation matrix
                if result.facial_transformation_matrixes:
                    head_pose = extract_head_pose(
                        result.facial_transformation_matrixes[0])

                # Head motion: nose tip displacement since last sample
                if result.face_landmarks:
                    nose = result.face_landmarks[0][1]   # index 1 = nose tip
                    nose_pos = (nose.x, nose.y)
                    if prev_nose_pos is not None:
                        head_motion = round(
                            ((nose_pos[0] - prev_nose_pos[0]) ** 2 +
                             (nose_pos[1] - prev_nose_pos[1]) ** 2) ** 0.5, 4)
                    prev_nose_pos = nose_pos

            elif backend == "fer":
                emotion, scores  = fer_emotions(frame)
                arousal, valence = arousal_valence_from_scores(scores)
            elif backend == "deepface":
                emotion, scores  = deepface_emotions(frame)
                arousal, valence = arousal_valence_from_scores(scores)

            # ── Audio analysis ─────────────────────────────────────────────
            audio_info = {}
            if audio_samples is not None:
                audio_info = analyse_audio_window(audio_samples, audio_sr, t)

            # ── Nearest caption event ──────────────────────────────────────
            caption_entry = nearest_caption(captions, t) if has_captions else {}

            timeline.append({
                "t":              t,
                "emotion":        emotion,
                "scores":         scores,
                # ── dimensional affect ────────────────────────────────────
                "arousal":        arousal,        # 0–1  (intensity)
                "valence":        valence,        # −1..+1 (pleasant vs unpleasant)
                "surprise":       surprise,       # 0–1
                "tension":        tension,        # 0–1
                # ── head motion ───────────────────────────────────────────
                "head_pose":      head_pose,      # {yaw, pitch, roll} degrees
                "head_motion":    head_motion,    # nose-tip displacement (norm. coords)
                # ── raw expression embedding ──────────────────────────────
                "blendshapes":    blendshapes_dict,   # 52 MediaPipe coefficients
                # ── audio ─────────────────────────────────────────────────
                "audio":          audio_info,
                "caption":        caption_entry.get("text", ""),
                "caption_events": caption_entry.get("events", []),
            })

            pbar.set_postfix(t=f"{t:.0f}s", face=emotion,
                             audio=audio_info.get("audio_emotion", "-"))
            pbar.update(1)
            frame_idx += step

    cap.release()
    if landmarker is not None:
        landmarker.close()

    out_path.write_text(json.dumps(timeline, indent=2))
    no_face   = sum(1 for e in timeline if e["emotion"] == "none")
    reactions = sum(1 for e in timeline if e["emotion"] not in ("neutral", "none"))
    reaction_rate = reactions / len(timeline) if timeline else 0.0
    cap_events = sum(1 for e in timeline if e.get("caption_events"))
    log.info("  ✓ %d entries — reactions=%.0f%%, no-face=%d, audio=%s, caption_events=%d",
             len(timeline), reaction_rate * 100, no_face,
             "yes" if audio_samples is not None else "no",
             cap_events)
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classify emotions in streamer videos")
    parser.add_argument(
        "--backend", default="mediapipe",
        choices=["mediapipe", "fer", "deepface"],
        help=(
            "Emotion detection backend:\n"
            "  mediapipe — MediaPipe blendshapes (default, no extra install)\n"
            "  fer        — FER2013 CNN (pip install fer)  ← recommended\n"
            "  deepface   — DeepFace ensemble (pip install deepface)  ← most accurate"
        ),
    )
    args = parser.parse_args()

    if args.backend == "mediapipe":
        try:
            import mediapipe
        except ImportError:
            log.error("pip install mediapipe")
            sys.exit(1)
    elif args.backend == "fer":
        try:
            from fer import FER
        except ImportError:
            log.error("pip install fer")
            sys.exit(1)
    elif args.backend == "deepface":
        try:
            from deepface import DeepFace
        except ImportError:
            log.error("pip install deepface")
            sys.exit(1)

    vid_dirs = sorted(d for d in DATASET_DIR.iterdir()
                      if d.is_dir() and (
                          (d / "facecam.mp4").exists() or (d / "raw.mp4").exists()
                      ))

    if not vid_dirs:
        log.error("No videos found in %s — run assemble.py first", DATASET_DIR)
        sys.exit(1)

    log.info("Backend: %s | Classifying %d videos (1 frame/5s)...",
             args.backend, len(vid_dirs))

    ok = 0
    for i, vid_dir in enumerate(vid_dirs, 1):
        log.info("[%d/%d] %s", i, len(vid_dirs), vid_dir.name)
        result = classify_video(vid_dir, vid_dir / "emotions.json",
                                backend=args.backend)
        if result:
            ok += 1

    log.info("Done — %d/%d videos classified successfully.", ok, len(vid_dirs))


if __name__ == "__main__":
    main()

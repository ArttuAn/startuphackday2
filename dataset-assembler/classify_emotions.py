"""
classify_emotions.py — Classify emotion every minute (prototype) from face + audio.

Face: MediaPipe FaceLandmarker blendshapes → 10 emotion scores
Audio: 5-second window around each timestamp → loudness, pitch, speech energy

Output per video:
    dataset/<video_id>/emotions.json   — [{t, emotion, scores, audio}, ...]
    dataset/<video_id>/emotion_frames/ — face screenshot per timestamp

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

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
        audio_emotion = "silent"
    elif loudness < 20 and speech_ratio < 0.3:
        audio_emotion = "quiet"
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

def classify_video(vid_dir: Path, out_path: Path) -> bool:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    facecam_path = vid_dir / "facecam.mp4" if (vid_dir / "facecam.mp4").exists() else vid_dir / "raw.mp4"
    audio_path   = vid_dir / "audio.wav"

    model_path = _ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.2,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Load audio once
    audio_samples, audio_sr = load_audio(audio_path) if audio_path.exists() else (None, None)
    if audio_samples is None:
        log.warning("  no audio.wav found — skipping audio analysis")

    cap      = cv2.VideoCapture(str(facecam_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step     = max(1, int(fps * 60))   # ⚠️ PROTOTYPE: 1/min — change to int(fps) for 1/sec
    n_frames = total_f // step

    frames_dir = out_path.parent / "emotion_frames"
    frames_dir.mkdir(exist_ok=True)

    timeline  = []
    frame_idx = 0

    with tqdm(total=n_frames, unit="s", desc=vid_dir.name, dynamic_ncols=True) as pbar:
        while frame_idx < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            t = round(frame_idx / fps, 2)

            # ── Face emotion ───────────────────────────────────────────────
            emotion, scores = "none", {}
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                emotion, scores = blendshapes_to_emotions(bs)

            # ── Audio analysis ─────────────────────────────────────────────
            audio_info = {}
            if audio_samples is not None:
                audio_info = analyse_audio_window(audio_samples, audio_sr, t)

            timeline.append({
                "t":       t,
                "emotion": emotion,
                "scores":  scores,
                "audio":   audio_info,
            })

            # ── Screenshot ────────────────────────────────────────────────
            frame_name = f"{int(t):06d}s_{emotion}.jpg"
            fh, fw = frame.shape[:2]
            if fw < 512:
                scale     = 512 / fw
                frame_out = cv2.resize(frame, (int(fw*scale), int(fh*scale)),
                                       interpolation=cv2.INTER_LANCZOS4)
            else:
                frame_out = frame
            cv2.imwrite(str(frames_dir / frame_name), frame_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            pbar.set_postfix(t=f"{t:.0f}s", face=emotion,
                             audio=audio_info.get("audio_emotion", "-"))
            pbar.update(1)
            frame_idx += step

    cap.release()
    landmarker.close()

    out_path.write_text(json.dumps(timeline, indent=2))
    no_face = sum(1 for e in timeline if e["emotion"] == "none")
    log.info("  ✓ %d entries — %d no-face, audio=%s",
             len(timeline), no_face,
             "yes" if audio_samples is not None else "no")
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    try:
        import mediapipe
    except ImportError:
        log.error("mediapipe not installed — run: pip install mediapipe")
        sys.exit(1)

    vid_dirs = sorted(d for d in DATASET_DIR.iterdir()
                      if d.is_dir() and (
                          (d / "facecam.mp4").exists() or (d / "raw.mp4").exists()
                      ))

    if not vid_dirs:
        log.error("No videos found in %s — run assemble.py first", DATASET_DIR)
        sys.exit(1)

    log.warning("⚠️  PROTOTYPE MODE: 1 frame/minute. Change step to int(fps) for 1/sec.")
    log.info("Classifying emotions for %d videos...", len(vid_dirs))
    ok = 0

    for i, vid_dir in enumerate(vid_dirs, 1):
        log.info("[%d/%d] %s", i, len(vid_dirs), vid_dir.name)
        if classify_video(vid_dir, vid_dir / "emotions.json"):
            ok += 1

    log.info("Done — %d/%d videos classified", ok, len(vid_dirs))


if __name__ == "__main__":
    main()

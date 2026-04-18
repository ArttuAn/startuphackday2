"""
vad.py — Voice Activity Detection quality filter.

Uses webrtcvad (Google's WebRTC VAD) on the extracted 16 kHz mono WAV.
Falls back to a simple energy-based heuristic if webrtcvad is unavailable.

Returns True if the video has sufficient speech, False otherwise.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from config import AUDIO_SAMPLE_RATE, RAW_VIDEOS_DIR, VAD_SPEECH_THRESHOLD
from logger_setup import get_logger

log = get_logger("vad")

_FRAME_MS    = 30          # WebRTC VAD requires 10, 20, or 30 ms frames
_AGGRESSIVENESS = 2        # 0 (least) – 3 (most aggressive filtering)


# ── WebRTC VAD ─────────────────────────────────────────────────────────────

def _webrtcvad_speech_ratio(wav_path: Path) -> float:
    """Return fraction of 30 ms frames classified as speech."""
    import webrtcvad
    vad = webrtcvad.Vad(_AGGRESSIVENESS)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()  # bytes per sample
        n_frames    = wf.getnframes()
        raw_audio   = wf.readframes(n_frames)

    # WebRTC VAD requires mono 16-bit at 8/16/32/48 kHz
    if n_channels != 1 or sample_width != 2:
        log.warning("Unexpected WAV format (channels=%d width=%d); using heuristic", n_channels, sample_width)
        return _energy_speech_ratio_from_bytes(raw_audio, sample_rate)

    # Frame size in bytes
    frame_bytes = int(sample_rate * _FRAME_MS / 1000) * sample_width
    total = 0
    speech = 0

    offset = 0
    while offset + frame_bytes <= len(raw_audio):
        frame = raw_audio[offset: offset + frame_bytes]
        total += 1
        try:
            if vad.is_speech(frame, sample_rate):
                speech += 1
        except Exception:
            pass
        offset += frame_bytes

    return speech / total if total else 0.0


# ── Energy heuristic fallback ─────────────────────────────────────────────

def _energy_speech_ratio_from_bytes(raw: bytes, sample_rate: int) -> float:
    """Rough speech ratio based on RMS energy threshold."""
    n_samples = len(raw) // 2
    samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])

    frame_size = sample_rate // 10  # 100 ms frames
    threshold  = 500                 # empirical silence floor (16-bit PCM)

    total = speech = 0
    for i in range(0, len(samples) - frame_size, frame_size):
        chunk = samples[i: i + frame_size]
        rms   = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        total += 1
        if rms > threshold:
            speech += 1

    return speech / total if total else 0.0


# ── Public API ─────────────────────────────────────────────────────────────

def check_speech(vid_id: str) -> tuple[bool, float]:
    """
    Run VAD on the extracted audio for vid_id.

    Returns:
        (has_speech: bool, ratio: float)
        has_speech is True when ratio >= VAD_SPEECH_THRESHOLD.
    """
    wav_path = RAW_VIDEOS_DIR / vid_id / "audio.wav"

    if not wav_path.exists():
        log.warning("No audio file for %s; cannot run VAD", vid_id)
        return False, 0.0

    try:
        import webrtcvad  # noqa: F401
        ratio = _webrtcvad_speech_ratio(wav_path)
        method = "webrtcvad"
    except ImportError:
        log.debug("webrtcvad not installed; falling back to energy heuristic")
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sr  = wf.getframerate()
        ratio  = _energy_speech_ratio_from_bytes(raw, sr)
        method = "energy"

    has_speech = ratio >= VAD_SPEECH_THRESHOLD
    log.info(
        "VAD [%s] %s: speech=%.1f%%  → %s",
        method, vid_id, ratio * 100, "PASS" if has_speech else "FAIL",
    )
    return has_speech, ratio

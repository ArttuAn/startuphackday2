"""
Block 3 — voice_separator.py

Splits the raw video audio into:
  voice.wav       — streamer speech only
  game_audio.wav  — everything else (game sounds, music)

Uses Facebook Demucs (htdemucs model, two-stem vocal separation).
Falls back to extracting the raw mixed audio if Demucs is unavailable.

Input:  <out_dir>/../raw.mp4   (or explicit video_path)
Output: <out_dir>/voice.wav
        <out_dir>/game_audio.wav
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("voice_separator")


def _has_demucs() -> bool:
    return shutil.which("demucs") is not None or _demucs_importable()


def _demucs_importable() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_raw_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000) -> bool:
    """Extract mixed audio from video as a fallback."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _run_demucs(video_path: Path, out_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Run Demucs two-stem separation on the audio track.
    Returns (voice_path, game_audio_path) or (None, None) on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Step 1: extract full audio from video → tmp wav
        mixed_wav = tmp_path / "mixed.wav"
        cmd_extract = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(mixed_wav),
        ]
        try:
            subprocess.run(cmd_extract, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            log.error("Audio extraction for Demucs failed: %s", exc)
            return None, None

        # Step 2: run Demucs — produces htdemucs/<stem>/vocals.wav + no_vocals.wav
        cmd_demucs = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "--out", str(tmp_path),
            str(mixed_wav),
        ]
        try:
            subprocess.run(cmd_demucs, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            log.error("Demucs failed: %s", exc.stderr.decode(errors="replace")[:400])
            return None, None

        # Step 3: find output files (Demucs nests under model name folder)
        vocals_candidates    = list(tmp_path.rglob("vocals.wav"))
        no_vocals_candidates = list(tmp_path.rglob("no_vocals.wav"))

        if not vocals_candidates:
            log.error("Demucs vocals.wav not found in output")
            return None, None

        vocals_src    = vocals_candidates[0]
        no_vocals_src = no_vocals_candidates[0] if no_vocals_candidates else None

        # Step 4: downsample to 16 kHz mono and copy to out_dir
        voice_out      = out_dir / "voice.wav"
        game_audio_out = out_dir / "game_audio.wav"

        def _resample(src: Path, dst: Path) -> bool:
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le", str(dst),
            ]
            try:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except subprocess.CalledProcessError:
                return False

        ok1 = _resample(vocals_src, voice_out)
        ok2 = _resample(no_vocals_src, game_audio_out) if no_vocals_src else False

        return (voice_out if ok1 else None,
                game_audio_out if ok2 else None)


# ── Public API ─────────────────────────────────────────────────────────────

def separate_voice(
    video_path: Path,
    out_dir:    Path,
    force:      bool = False,
) -> dict[str, Path | None]:
    """
    Split audio into voice + game tracks.

    Args:
        video_path: path to raw.mp4
        out_dir:    folder to write voice.wav and game_audio.wav
        force:      re-run even if outputs already exist

    Returns:
        { "voice": Path | None, "game_audio": Path | None }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path      = out_dir / "voice.wav"
    game_audio_path = out_dir / "game_audio.wav"

    if not force and voice_path.exists() and game_audio_path.exists():
        log.info("Voice separation already done for %s — skipping", out_dir.name)
        return {"voice": voice_path, "game_audio": game_audio_path}

    if _has_demucs():
        log.info("Running Demucs voice separation for %s...", out_dir.name)
        voice, game = _run_demucs(video_path, out_dir)
        if voice:
            log.info("✓ Voice separated: %s", voice.name)
            return {"voice": voice, "game_audio": game}
        log.warning("Demucs failed — falling back to raw audio")
    else:
        log.warning("Demucs not installed — extracting raw mixed audio (install with: pip install demucs)")

    # Fallback: raw mixed audio
    if _extract_raw_audio(video_path, voice_path):
        log.info("✓ Raw audio extracted (not separated): %s", voice_path.name)
        return {"voice": voice_path, "game_audio": None}

    log.error("✗ Audio extraction failed for %s", video_path)
    return {"voice": None, "game_audio": None}

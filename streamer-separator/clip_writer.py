"""
clip_writer.py — Assemble and write the final clip_XXX.json.

JSON schema:
{
  "video_id":          "0LrgyRuvT1U",
  "clip_id":           "clip_001",

  // Reaction window (streamer face + voice)
  "t_start":           120.0,
  "t_end":             128.0,
  "duration":          8.0,

  // Gameplay trigger window (what caused the reaction)
  "t_gameplay":        111.25,   // = t_start - reaction_delay - gameplay_window

  // File paths (relative to this video's output folder)
  "gameplay_frames":   "clip_001/gameplay",
  "facecam_frames":    "clip_001/facecam",
  "audio":             "clip_001/audio.wav",
  "voice":             "clip_001/voice.wav",      // streamer voice only (Demucs)

  // Per-frame face structure for the reaction window
  "landmarks":  [
    {
      "frame": 3600, "t": 120.0,
      "landmarks": [[x,y,z], ...],
      "head_pose": {"pitch": -5.2, "yaw": 12.1, "roll": 1.3},
      "emotion": "fear",
      "emotion_scores": {"fear": 71.2, "neutral": 18.1, ...}
    },
    ...
  ],

  // Coarse emotion timeline (from scan, same range)
  "emotion_timeline": [
    {"t": 120.0, "emotion": "neutral", "scores": {...}},
    {"t": 120.5, "emotion": "fear",    "scores": {...}},
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from clip_cutter import ClipWindow

log = logging.getLogger("clip_writer")


def build_clip_json(
    video_id:   str,
    window:     ClipWindow,
    paths:      dict,
    timeline:   list,
    landmarks:  list[dict] | None = None,
) -> dict:
    """
    Build the clip dict.
    timeline and landmarks are filtered to the reaction window's time range.
    """
    clip_id = f"clip_{window.index:03d}"

    # Slice coarse emotion timeline to reaction window
    def _t(s) -> float:
        return s.t if hasattr(s, "t") else s["t"]

    def _to_dict(s) -> dict:
        return s.to_dict() if hasattr(s, "to_dict") else s

    emotion_slice = [
        _to_dict(s) for s in timeline
        if window.t_start <= _t(s) <= window.t_end
    ]

    # Slice per-frame landmarks to reaction window
    landmark_slice = []
    if landmarks:
        landmark_slice = [
            r for r in landmarks
            if window.t_start <= r.get("t", 0.0) <= window.t_end
        ]

    return {
        "video_id":        video_id,
        "clip_id":         clip_id,
        "t_start":         round(window.t_start,    3),
        "t_end":           round(window.t_end,      3),
        "duration":        round(window.duration,   3),
        "t_gameplay":      round(window.t_gameplay, 3),
        "gameplay_frames": paths.get("gameplay_frames", ""),
        "facecam_frames":  paths.get("facecam_frames",  ""),
        "audio":           paths.get("audio",           ""),
        "voice":           paths.get("voice",           ""),
        "landmarks":       landmark_slice,
        "emotion_timeline": emotion_slice,
    }


def write_clip_json(clip_data: dict, out_dir: Path) -> Path:
    """Write clip_XXX.json to out_dir. Returns the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{clip_data['clip_id']}.json"
    json_path.write_text(json.dumps(clip_data, indent=2, ensure_ascii=False))
    log.info("Wrote %s  (%d landmarks, %d emotion entries)",
             json_path.name,
             len(clip_data.get("landmarks", [])),
             len(clip_data.get("emotion_timeline", [])))
    return json_path

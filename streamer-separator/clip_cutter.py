"""
clip_cutter.py — Find reaction event windows from an emotion/landmark timeline.

Each ClipWindow has two time ranges:
  - reaction window  [t_start      → t_end]        streamer face + audio
  - gameplay window  [t_gameplay   → t_start]       what triggered the reaction
    (shifted back by REACTION_DELAY_SEC before the reaction starts)

This separation lets the model learn:
  "given this gameplay footage → generate this reaction"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import (
    MAX_CLIP_SEC,
    MERGE_GAP_SEC,
    MIN_CLIP_SEC,
    POST_BUFFER_SEC,
    PRE_BUFFER_SEC,
)

log = logging.getLogger("clip_cutter")

# How far before the reaction the gameplay trigger window starts (seconds)
REACTION_DELAY_SEC  = 0.75
GAMEPLAY_WINDOW_SEC = 5.0    # length of the gameplay trigger window


@dataclass
class ClipWindow:
    index:       int
    t_start:     float   # reaction window start
    t_end:       float   # reaction window end
    t_gameplay:  float   # gameplay trigger window start (before reaction)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def __str__(self) -> str:
        return (
            f"clip_{self.index:03d}  "
            f"gameplay=[{self.t_gameplay:.1f}s→{self.t_start:.1f}s]  "
            f"reaction=[{self.t_start:.1f}s→{self.t_end:.1f}s  ({self.duration:.1f}s)]"
        )


def _is_reaction(sample: dict) -> bool:
    """Works with both EmotionSample objects and raw landmark dicts."""
    from config import EMOTION_CONFIDENCE_MIN, REACTION_EMOTIONS
    if hasattr(sample, "is_reaction"):
        return sample.is_reaction()
    emotion = sample.get("emotion", "neutral")
    score   = sample.get("emotion_scores", {}).get(emotion, 0.0)
    return emotion in REACTION_EMOTIONS and score >= EMOTION_CONFIDENCE_MIN


def find_clip_windows(
    timeline:       list,
    video_duration: float,
) -> list[ClipWindow]:
    """
    Identify reaction + gameplay windows from a timeline.

    Args:
        timeline:       list of EmotionSample or landmark dicts (both supported)
        video_duration: total video length in seconds

    Returns:
        Sorted list of ClipWindow objects.
    """
    # ── 1. Collect reaction timestamps ────────────────────────────────────
    reaction_times = [
        (s.t if hasattr(s, "t") else s["t"])
        for s in timeline
        if _is_reaction(s)
    ]

    if not reaction_times:
        log.warning("No reaction events in timeline")
        return []

    log.info("%d reaction frames found across %.0fs of video",
             len(reaction_times), video_duration)

    # ── 2. Cluster nearby reactions ────────────────────────────────────────
    clusters: list[list[float]] = []
    current:  list[float]       = [reaction_times[0]]

    for t in reaction_times[1:]:
        if t - current[-1] <= MERGE_GAP_SEC:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    clusters.append(current)

    # ── 3. Build windows with buffers + gameplay offset ────────────────────
    windows: list[ClipWindow] = []

    for cluster in clusters:
        # Reaction window
        raw_start = min(cluster) - PRE_BUFFER_SEC
        raw_end   = max(cluster) + POST_BUFFER_SEC
        t_start   = max(0.0, raw_start)
        t_end     = min(video_duration, raw_end)

        # Cap very long clips
        if (t_end - t_start) > MAX_CLIP_SEC:
            mid     = (min(cluster) + max(cluster)) / 2
            t_start = max(0.0,           mid - MAX_CLIP_SEC / 2)
            t_end   = min(video_duration, mid + MAX_CLIP_SEC / 2)

        if (t_end - t_start) < MIN_CLIP_SEC:
            log.debug("Dropped short clip at %.1fs", t_start)
            continue

        # Gameplay trigger window — ends just before the reaction starts,
        # shifted back by REACTION_DELAY_SEC to account for human reaction lag
        t_gameplay = max(0.0, t_start - REACTION_DELAY_SEC - GAMEPLAY_WINDOW_SEC)

        windows.append(ClipWindow(
            index=len(windows) + 1,
            t_start=t_start,
            t_end=t_end,
            t_gameplay=t_gameplay,
        ))

    log.info("Cut %d clips from %d clusters", len(windows), len(clusters))
    for w in windows:
        log.info("  %s", w)

    return windows

"""
parse_captions.py — Parse YouTube VTT caption files into a clean timeline.

YouTube auto-generated captions include bracketed sound events:
    [laughter], [applause], [music], [gasps], [cheering], [sighs] ...

These are gold for reaction modelling — they timestamp emotional vocal events
that face blendshapes alone may miss (e.g. laughing off-screen, audio-only
reactions).

Output format (captions.json):
    [
        {
            "start": 12.5,          # seconds
            "end":   14.0,
            "text":  "oh no no no",
            "events": ["laughter"]  # bracketed sound labels, lowercased, empty if none
        },
        ...
    ]

Usage (standalone):
    python parse_captions.py <video_dir>

    Reads <video_dir>/<id>.en.vtt (or *.vtt) and writes <video_dir>/captions.json.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("parse_captions")

# Bracketed sound events we care about (lowercased match)
SOUND_EVENTS = {
    "laughter", "laughing", "chuckling",
    "applause", "cheering", "clapping",
    "gasps", "gasping",
    "sighs", "sighing",
    "music",
    "screaming", "screams",
    "crying", "sobbing",
    "groaning", "groans",
    "whistling",
    "shouting",
}

# Regex to find [bracketed text] anywhere in a caption line
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
# Strip VTT inline timing tags like <00:00:00.000> and <c>...</c>
_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_time_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm or MM:SS.mmm to float seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def parse_vtt(vtt_path: Path) -> list[dict]:
    """
    Parse a WebVTT file and return a list of caption entries.

    Handles:
      - Standard VTT cue blocks
      - YouTube's inline timing tags (<00:00:00.000><c>word</c>)
      - Duplicate/overlapping cues (deduplication by start time + text)
    """
    text = vtt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    entries: list[dict] = []
    seen: set[tuple] = set()   # (start, text) dedup

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp lines: "00:00:12.500 --> 00:00:14.000 ..."
        if "-->" in line:
            time_part = line.split("-->")
            try:
                start = _vtt_time_to_seconds(time_part[0])
                end   = _vtt_time_to_seconds(time_part[1].split()[0])
            except (ValueError, IndexError):
                i += 1
                continue

            # Collect text lines until blank line or next timestamp
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                raw = lines[i].strip()
                # Strip inline VTT tags, keep plain text
                clean = _TAG_RE.sub("", raw).strip()
                if clean:
                    text_lines.append(clean)
                i += 1

            full_text = " ".join(text_lines).strip()
            if not full_text:
                continue

            # Dedup — YouTube VTT often repeats cues with rolling windows
            key = (round(start, 1), full_text[:80])
            if key in seen:
                continue
            seen.add(key)

            # Extract bracketed sound events
            brackets = [m.group(1).lower().strip() for m in _BRACKET_RE.finditer(full_text)]
            events   = [b for b in brackets if any(ev in b for ev in SOUND_EVENTS)]

            # Clean brackets from display text
            clean_text = _BRACKET_RE.sub("", full_text).strip()

            entries.append({
                "start":  round(start, 3),
                "end":    round(end, 3),
                "text":   clean_text,
                "events": events,
            })
        else:
            i += 1

    entries.sort(key=lambda e: e["start"])
    return entries


def find_vtt_file(vid_dir: Path) -> Path | None:
    """Find the first .vtt caption file in vid_dir."""
    # Prefer English
    for pattern in ["*.en.vtt", "*.en-US.vtt", "*.vtt"]:
        matches = list(vid_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_captions(vid_dir: Path) -> list[dict]:
    """
    Load and parse captions for a video directory.
    Returns empty list if no VTT file found.
    """
    vtt = find_vtt_file(vid_dir)
    if vtt is None:
        return []
    try:
        entries = parse_vtt(vtt)
        log.info("  captions: %d entries, %d sound events from %s",
                 len(entries),
                 sum(1 for e in entries if e["events"]),
                 vtt.name)
        return entries
    except Exception as exc:
        log.warning("  could not parse captions: %s", exc)
        return []


def nearest_caption(captions: list[dict], t: float, window: float = 30.0) -> dict:
    """
    Return the caption entry whose midpoint is closest to time t,
    within ±window seconds. Returns empty dict if none found.
    """
    best     = None
    best_dist = window + 1

    for c in captions:
        mid  = (c["start"] + c["end"]) / 2
        dist = abs(mid - t)
        if dist < best_dist:
            best_dist = dist
            best      = c

    return best if best_dist <= window else {}


# ── Standalone CLI ─────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python parse_captions.py <video_dir>")
        sys.exit(1)

    vid_dir = Path(sys.argv[1])
    entries = load_captions(vid_dir)
    if not entries:
        print("No captions found.")
        sys.exit(0)

    out = vid_dir / "captions.json"
    out.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} entries to {out}")
    events_total = sum(1 for e in entries if e["events"])
    print(f"Sound events found: {events_total}")


if __name__ == "__main__":
    main()

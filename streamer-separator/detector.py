"""
detector.py — Automatically locate the streamer's facecam bounding box.

Strategy:
  1. Sample N frames evenly across the video.
  2. Upscale each frame to at least 640px wide (helps low-res 256x144 videos).
  3. Run MediaPipe FaceLandmarker (Tasks API, MP 0.10+).
     Falls back to OpenCV Haar cascade if model unavailable.
  4. Collect all face bounding boxes and find the most stable cluster
     (median position across frames with detections).
  5. Expand the box to include the body (shoulders + torso).
  6. Optionally snap the region to the nearest screen corner.

Returns a BBox(x, y, w, h) in pixel coordinates, or None if no face found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    BODY_HEIGHT_SCALE,
    BODY_WIDTH_SCALE,
    CORNER_SNAP_THRESHOLD,
    DETECTION_SAMPLE_FRAMES,
    MIN_FACE_HIT_RATE,
)

log = logging.getLogger("detector")

# Minimum width to upscale frames to before detection (helps tiny facecams)
_DETECT_MIN_WIDTH = 640


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def __str__(self) -> str:
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    def as_ffmpeg_crop(self) -> str:
        """Return ffmpeg crop filter string: crop=w:h:x:y"""
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"

    def clamp(self, frame_w: int, frame_h: int) -> "BBox":
        """Ensure box stays within frame boundaries."""
        x = max(0, min(self.x, frame_w - 1))
        y = max(0, min(self.y, frame_h - 1))
        w = max(1, min(self.w, frame_w - x))
        h = max(1, min(self.h, frame_h - y))
        return BBox(x, y, w, h)


# ── Frame upscaling ────────────────────────────────────────────────────────

def _upscale(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Upscale frame so its width is at least _DETECT_MIN_WIDTH.
    Returns (upscaled_frame, scale_factor).  scale_factor < 1.0 means no change.
    """
    h, w = frame.shape[:2]
    if w >= _DETECT_MIN_WIDTH:
        return frame, 1.0
    scale = _DETECT_MIN_WIDTH / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


# ── MediaPipe FaceLandmarker detector (Tasks API, MP 0.10+) ───────────────

_landmarker = None

def _get_landmarker():
    """Lazy-init MediaPipe FaceLandmarker (reuses model downloaded by emotion_scanner)."""
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from emotion_scanner import _ensure_model
        model_path = _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            num_faces=1,
            min_face_detection_confidence=0.2,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return _landmarker
    except Exception as exc:
        log.warning("Could not init MediaPipe FaceLandmarker: %s", exc)
        return None


def _detect_faces_mediapipe(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) face boxes using MediaPipe FaceLandmarker."""
    import mediapipe as mp
    lm = _get_landmarker()
    if lm is None:
        return []

    h, w = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm.detect(mp_img)

    if not result.face_landmarks:
        return []

    boxes = []
    for face_lm in result.face_landmarks:
        xs = [p.x * w for p in face_lm]
        ys = [p.y * h for p in face_lm]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        boxes.append((x1, y1, bw, bh))
    return boxes


# ── OpenCV Haar fallback ───────────────────────────────────────────────────

_haar = None

def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar


def _detect_faces_haar(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# ── Frame sampling ─────────────────────────────────────────────────────────

def _sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    """Extract n evenly-spaced frames from video_path."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


# ── Clustering: find the most stable face region ──────────────────────────

def _dominant_box(
    all_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Return the median bounding box from all detections."""
    if not all_boxes:
        return None
    xs  = sorted(b[0] for b in all_boxes)
    ys  = sorted(b[1] for b in all_boxes)
    ws  = sorted(b[2] for b in all_boxes)
    hs  = sorted(b[3] for b in all_boxes)
    mid = len(all_boxes) // 2
    return xs[mid], ys[mid], ws[mid], hs[mid]


# ── Expand face box → body box ─────────────────────────────────────────────

def _expand_to_body(
    fx: int, fy: int, fw: int, fh: int,
    frame_w: int, frame_h: int,
) -> BBox:
    """Scale face box outward to cover shoulders + torso."""
    cx = fx + fw // 2

    new_w = int(fw * BODY_WIDTH_SCALE)
    new_h = int(fh * BODY_HEIGHT_SCALE)

    # Keep the face centred horizontally; extend downward for the body
    new_x = cx - new_w // 2
    new_y = fy - int(fh * 0.3)   # small upward padding above head

    return BBox(new_x, new_y, new_w, new_h).clamp(frame_w, frame_h)


# ── Corner snapping ────────────────────────────────────────────────────────

def _snap_to_corner(box: BBox, frame_w: int, frame_h: int) -> BBox:
    """
    If the box is near a corner, snap it flush to that corner and extend
    outward so it doesn't clip the overlay's shadow/border.
    """
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2

    snap_x = CORNER_SNAP_THRESHOLD * frame_w
    snap_y = CORNER_SNAP_THRESHOLD * frame_h

    x, y, w, h = box.x, box.y, box.w, box.h

    # Horizontal snap
    if cx < snap_x:
        x = 0
    elif cx > frame_w - snap_x:
        x = frame_w - w

    # Vertical snap
    if cy < snap_y:
        y = 0
    elif cy > frame_h - snap_y:
        y = frame_h - h

    return BBox(x, y, w, h).clamp(frame_w, frame_h)


# ── Public API ─────────────────────────────────────────────────────────────

def detect_facecam_region(video_path: Path, use_mediapipe: bool = True) -> BBox | None:
    """
    Analyse video_path and return the estimated facecam BBox.

    Frames are upscaled to at least 640px wide before detection so that
    tiny facecam overlays on low-res (e.g. 256x144) videos are detectable.
    Detected coordinates are scaled back to native resolution before returning.

    Args:
        video_path:     path to the raw.mp4 file
        use_mediapipe:  try MediaPipe FaceLandmarker first; fall back to Haar

    Returns:
        BBox or None if no stable face region was found.
    """
    if not video_path.exists():
        log.error("Video not found: %s", video_path)
        return None

    cap  = cv2.VideoCapture(str(video_path))
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    log.info("Sampling %d frames from %s (%dx%d)", DETECTION_SAMPLE_FRAMES, video_path.name, fw, fh)
    frames = _sample_frames(video_path, DETECTION_SAMPLE_FRAMES)
    if not frames:
        log.warning("Could not read frames from %s", video_path)
        return None

    # Choose detector — prefer new MediaPipe Tasks API
    detect_fn = _detect_faces_haar  # default fallback
    if use_mediapipe:
        if _get_landmarker() is not None:
            detect_fn = _detect_faces_mediapipe
            log.debug("Using MediaPipe FaceLandmarker detector")
        else:
            log.warning("MediaPipe FaceLandmarker unavailable; using Haar cascade fallback")

    # Collect detections (upscale each frame first)
    all_boxes: list[tuple[int, int, int, int]] = []
    hit_frames = 0

    for frame in frames:
        upscaled, scale = _upscale(frame)
        boxes = detect_fn(upscaled)
        if boxes:
            hit_frames += 1
            # Take the largest face, then scale coordinates back to native res
            largest = max(boxes, key=lambda b: b[2] * b[3])
            if scale != 1.0:
                largest = tuple(int(v / scale) for v in largest)
            all_boxes.append(largest)

    hit_rate = hit_frames / len(frames)
    log.info("Face detected in %.0f%% of sampled frames (%d/%d)", hit_rate * 100, hit_frames, len(frames))

    if hit_rate < MIN_FACE_HIT_RATE:
        log.warning("Hit rate too low (%.0f%% < %.0f%%) — no stable facecam region found",
                    hit_rate * 100, MIN_FACE_HIT_RATE * 100)
        return None

    dominant = _dominant_box(all_boxes)
    if dominant is None:
        return None

    fx, fy, fw_box, fh_box = dominant
    body_box = _expand_to_body(fx, fy, fw_box, fh_box, fw, fh)
    snapped  = _snap_to_corner(body_box, fw, fh)

    log.info("Facecam region: %s", snapped)
    return snapped

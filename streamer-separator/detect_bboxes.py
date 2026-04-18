"""
detect_bboxes.py — Grab one frame per video, find the face with OpenCV, save bbox + crop.

Usage:
    python detect_bboxes.py

Output:
    output/bboxes.json              — all bboxes in one file
    output/<video_id>/
        bbox.txt                    — x=.. y=.. w=.. h=..
        facecam.jpg                 — cropped face region
"""

import json
import logging
import sys
from pathlib import Path

import cv2

from config import RAW_VIDEOS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_in_frame(frame):
    """Upscale small frames, run Haar cascade, return largest (x,y,w,h) or None."""
    h, w = frame.shape[:2]
    scale = max(1.0, 640 / w)
    if scale > 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    if len(faces) == 0:
        return None

    # Largest face = most likely the streamer overlay
    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    # Scale coords back to original resolution
    return (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))


def grab_frame(video_path: Path):
    """Return the middle frame of the video."""
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    video_dirs = sorted(d for d in RAW_VIDEOS_DIR.iterdir()
                        if d.is_dir() and (d / "raw.mp4").exists())

    if not video_dirs:
        log.error("No raw.mp4 files found in %s", RAW_VIDEOS_DIR)
        sys.exit(1)

    log.info("Found %d videos", len(video_dirs))
    results = {}

    for i, vid_dir in enumerate(video_dirs, 1):
        vid_id   = vid_dir.name
        out_dir  = OUTPUT_DIR / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("[%d/%d] %s", i, len(video_dirs), vid_id)

        frame = grab_frame(vid_dir / "raw.mp4")
        if frame is None:
            log.warning("  could not read frame — skipping")
            out_dir.rmdir()
            continue

        bbox = detect_face_in_frame(frame)
        if bbox is None:
            log.warning("  no face detected — skipping")
            out_dir.rmdir()  # remove the folder we just created
            continue

        x, y, w, h = bbox
        results[vid_id] = {"x": x, "y": y, "w": w, "h": h}
        log.info("  ✓ x=%d y=%d w=%d h=%d", x, y, w, h)

        # Save bbox.txt
        (out_dir / "bbox.txt").write_text(f"x={x} y={y} w={w} h={h}")

        # Save cropped face — upscale to at least 512px wide, high quality
        crop = frame[y:y+h, x:x+w]
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            if cw < 512:
                scale = 512 / cw
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(out_dir / "facecam.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save combined JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "bboxes.json").write_text(json.dumps(results, indent=2))

    detected = sum(1 for v in results.values() if v is not None)
    log.info("Done — %d/%d videos have a face bbox", detected, len(results))


if __name__ == "__main__":
    main()

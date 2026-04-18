"""
synthesize.py — Step 10: Synthesize novel streamer reactions via face reenactment.

Given a source portrait (your AI streamer's face), this script:
  1. Predicts emotion from gameplay using the trained model (step 7)
  2. Retrieves the best-matching reaction clip from the index (step 8)
  3. Transfers the driving facecam motion onto the source portrait using
     MediaPipe landmark-based warping (genuine face reenactment)
  4. Composites the animated synthetic face onto gameplay

Output is SYNTHETIC — a new face (not a recording) expressing emotions
learned from real Signalis streamers, reacting to new gameplay footage.

Usage:
    # Synthesize reaction for a single gameplay clip:
    python synthesize.py --source_portrait portraits/streamer.jpg --input gameplay.mp4

    # Batch: synthesize for all eval-game clips:
    python synthesize.py --source_portrait portraits/streamer.jpg --batch_eval

    # Called by run_pipeline.py (step 10):
    python synthesize.py --source_portrait portraits/default.jpg --batch_eval

Dependencies: mediapipe (already in requirements.txt), ffmpeg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import EMOTION_CLASSES, N_EMOTIONS, sample_frames_rgb
from generate import (DEFAULT_CHECKPOINT, DEFAULT_INDEX,
                      load_model_and_processor, predict_emotion_vector,
                      retrieve_clip)

BASE            = Path(__file__).parent
DEFAULT_DATASET = BASE.parent / "dataset-assembler" / "dataset"
DEFAULT_EVAL    = BASE.parent / "eval-scraper" / "data"
PORTRAITS_DIR   = BASE / "portraits"
OUT_DIR         = BASE / "synthesized"

MOTION_SCALE    = 0.65   # dampen driving motion (1.0 = full transfer)
N_LMS           = 468    # MediaPipe Face Mesh landmarks


# ── MediaPipe helpers ────────────────────────────────────────────────────────

def _get_lms(img_bgr: np.ndarray, face_mesh) -> np.ndarray | None:
    """Return (N_LMS, 2) float32 pixel coords, or None if no face detected."""
    h, w = img_bgr.shape[:2]
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    return np.array([(lm.x * w, lm.y * h)
                     for lm in lms[:N_LMS]], dtype=np.float32)


def _build_triangles(lms: np.ndarray, img_shape) -> list[tuple[int, int, int]]:
    """Delaunay triangulation of face landmarks → index triples."""
    h, w = img_shape[:2]
    pts  = np.clip(lms, [1, 1], [w - 2, h - 2])
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for pt in pts:
        subdiv.insert((float(pt[0]), float(pt[1])))

    result = []
    for tri in subdiv.getTriangleList():
        tri_pts = tri.reshape(3, 2)
        indices = []
        for pt in tri_pts:
            dists = np.sum((lms - pt) ** 2, axis=1)
            idx   = int(np.argmin(dists))
            if np.sqrt(dists[idx]) < 3.0:
                indices.append(idx)
        if len(set(indices)) == 3:
            result.append(tuple(indices))
    return result


# ── Per-triangle affine warp ─────────────────────────────────────────────────

def _warp_triangle(src: np.ndarray, dst: np.ndarray,
                   src_pts: np.ndarray, dst_pts: np.ndarray):
    """Warp one triangle from src image into dst image in-place."""
    h, w = src.shape[:2]

    r1   = cv2.boundingRect(src_pts.reshape(1, 3, 2))
    r2   = cv2.boundingRect(dst_pts.reshape(1, 3, 2))

    r1x, r1y = max(r1[0], 0), max(r1[1], 0)
    r1w      = min(r1[0] + r1[2], w) - r1x
    r1h      = min(r1[1] + r1[3], h) - r1y
    r2x, r2y = max(r2[0], 0), max(r2[1], 0)
    r2w      = min(r2[0] + r2[2], w) - r2x
    r2h      = min(r2[1] + r2[3], h) - r2y

    if r1w <= 0 or r1h <= 0 or r2w <= 0 or r2h <= 0:
        return

    src_local = src_pts - np.float32([r1[0], r1[1]])
    dst_local = dst_pts - np.float32([r2[0], r2[1]])

    M = cv2.getAffineTransform(src_local, dst_local)

    src_crop = src[r1y:r1y + r1h, r1x:r1x + r1w]
    warped   = cv2.warpAffine(src_crop, M, (r2[2], r2[3]),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

    mask     = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255)

    dst_roi  = dst[r2y:r2y + r2h, r2x:r2x + r2w]
    ch, cw   = dst_roi.shape[:2]
    warped   = warped[:ch, :cw]
    mask     = mask[:ch, :cw]
    dst_roi[mask > 0] = warped[mask > 0]


# ── Frame-level reenactment ──────────────────────────────────────────────────

def reenact_frame(source_img: np.ndarray,
                  source_lms: np.ndarray,
                  triangles: list,
                  driving_lms: np.ndarray,
                  neutral_lms: np.ndarray) -> np.ndarray:
    """
    Animate source_img by transferring expression motion from driving frame.

    motion   = (driving_lms - neutral_lms) * MOTION_SCALE
    target   = source_lms + motion        (where the source landmarks should go)
    Then: warp source triangles to target positions.
    Feather-blend back so background is unchanged.
    """
    h, w       = source_img.shape[:2]
    motion     = (driving_lms - neutral_lms) * MOTION_SCALE
    target_lms = np.clip(source_lms + motion, [0, 0], [w - 1, h - 1])

    output = source_img.copy()
    for i, j, k in triangles:
        _warp_triangle(source_img, output,
                       source_lms[[i, j, k]],
                       target_lms[[i, j, k]])

    # Feather-blend: restrict warped region to face convex hull
    hull  = cv2.convexHull(target_lms.astype(np.float32))
    mask  = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    mask  = cv2.dilate(mask,  np.ones((15, 15), np.uint8))
    mask  = cv2.GaussianBlur(mask, (21, 21), 8)
    alpha = mask.astype(np.float32)[:, :, None] / 255.0

    return (output * alpha + source_img * (1 - alpha)).astype(np.uint8)


# ── Video reenactment ────────────────────────────────────────────────────────

def reenact_video(source_img: np.ndarray,
                  driving_path: Path,
                  out_path: Path) -> bool:
    """
    Animate source_img for every frame of driving_path.
    Writes synthesized video (with driving audio) to out_path.
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("mediapipe not installed — pip install mediapipe")
        return False

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    # ── Source landmarks + triangulation (computed once) ──────────────────
    source_lms = _get_lms(source_img, face_mesh)
    if source_lms is None:
        print("  ✗ No face detected in source portrait")
        face_mesh.close()
        return False
    triangles = _build_triangles(source_lms, source_img.shape)

    # ── Open driving video ─────────────────────────────────────────────────
    cap   = cv2.VideoCapture(str(driving_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w  = source_img.shape[:2]

    # Determine neutral pose = first frame with detected face
    neutral_lms = source_lms.copy()
    for _ in range(min(30, total)):
        ok, frame = cap.read()
        if not ok:
            break
        lms = _get_lms(frame, face_mesh)
        if lms is not None:
            neutral_lms = lms
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Write synthesized frames to temp dir ──────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_idx = 0
        while True:
            ok, driving_frame = cap.read()
            if not ok:
                break

            # Resize driving frame to source size for landmark matching
            drv_resized = cv2.resize(driving_frame, (w, h))
            drv_lms     = _get_lms(drv_resized, face_mesh)
            if drv_lms is None:
                drv_lms = neutral_lms

            synth = reenact_frame(source_img, source_lms, triangles,
                                  drv_lms, neutral_lms)
            cv2.imwrite(f"{tmpdir}/{frame_idx:05d}.png", synth)
            frame_idx += 1

        cap.release()
        face_mesh.close()

        if frame_idx == 0:
            return False

        # ── ffmpeg: encode frames + copy audio from driving clip ──────────
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", f"{tmpdir}/%05d.png",
            "-i", str(driving_path),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr.decode()[:200]}")
            return False

    return True


# ── Composite synthesized face onto gameplay ─────────────────────────────────

def composite(gameplay_path: Path, synth_face_path: Path,
              out_path: Path, corner: str = "bottom-right",
              face_scale: float = 0.25) -> bool:
    cap = cv2.VideoCapture(str(gameplay_path))
    gw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fw = int(gw * face_scale)
    fh = fw
    positions = {
        "bottom-right": (gw - fw - 10, gh - fh - 10),
        "bottom-left":  (10,            gh - fh - 10),
        "top-right":    (gw - fw - 10, 10),
        "top-left":     (10,            10),
    }
    ox, oy = positions.get(corner, positions["bottom-right"])

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(gameplay_path),
        "-stream_loop", "-1",
        "-i", str(synth_face_path),
        "-filter_complex",
        f"[1:v]scale={fw}:{fh}[face];[0:v][face]overlay={ox}:{oy}:shortest=1[v]",
        "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── Full synthesis pipeline ──────────────────────────────────────────────────

def synthesize_reaction(
    gameplay_path: Path,
    source_portrait: Path,
    out_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    index_path: Path  = DEFAULT_INDEX,
    corner: str       = "bottom-right",
) -> bool:
    """
    End-to-end: gameplay → emotion prediction → retrieve → reenact → composite.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[synth] device={device} | portrait={source_portrait.name}")

    # Load model + index
    model, processor, n_frames = load_model_and_processor(checkpoint, device)
    index = torch.load(index_path, map_location="cpu")

    # Predict emotion from gameplay
    print(f"[synth] predicting emotion for {gameplay_path.name}...")
    pred_vec = predict_emotion_vector(
        model, processor, gameplay_path, n_frames, device)

    top_emotion = EMOTION_CLASSES[pred_vec.argmax().item()]
    print(f"[synth] predicted dominant emotion: {top_emotion}")

    # Retrieve best-matching driving clip
    clip_dir = retrieve_clip(pred_vec, index)
    driving  = clip_dir / "facecam.mp4"
    print(f"[synth] driving clip: {clip_dir.name}")

    if not driving.exists():
        print(f"  ✗ No facecam.mp4 in {clip_dir}")
        return False

    # Load source portrait
    source_img = cv2.imread(str(source_portrait))
    if source_img is None:
        print(f"  ✗ Cannot read source portrait: {source_portrait}")
        return False

    # Face reenactment: source + driving → synthesized face clip
    synth_face = out_path.with_suffix("") / "synth_face.mp4"
    synth_face.parent.mkdir(parents=True, exist_ok=True)
    print(f"[synth] running face reenactment...")
    ok = reenact_video(source_img, driving, synth_face)
    if not ok:
        print("  ✗ Reenactment failed")
        return False

    # Composite synthesized face onto gameplay
    print(f"[synth] compositing → {out_path}")
    ok = composite(gameplay_path, synth_face, out_path, corner=corner)
    if ok:
        print(f"[synth] ✓ saved: {out_path}")
    return ok


# ── Batch mode: synthesize for eval-game clips ──────────────────────────────

def batch_eval(source_portrait: Path,
               eval_dir: Path,
               out_dir: Path,
               checkpoint: Path,
               index_path: Path,
               max_clips: int = 5):
    """Synthesize reactions for the first max_clips eval gameplay clips.
    Falls back to training dataset clips if eval_dir is empty."""
    clips = [
        f for game_dir in sorted(eval_dir.iterdir()) if game_dir.is_dir()
        for f in sorted(game_dir.glob("*.mp4"))
    ] if eval_dir.exists() else []

    if not clips:
        print(f"No eval clips in {eval_dir} — falling back to training dataset clips")
        clips = sorted(DEFAULT_DATASET.glob("*/clips/*/gameplay.mp4"))

    if not clips:
        print("No gameplay clips found anywhere. Run step 6 (extract_clips) first.")
        return

    clips = clips[:max_clips]
    print(f"[synth] batch: {len(clips)} clips")

    for clip in clips:
        out = out_dir / clip.parent.name / (clip.stem + "_synth.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        synthesize_reaction(clip, source_portrait, out, checkpoint, index_path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 10 — Synthesize streamer reactions via face reenactment")
    parser.add_argument("--source_portrait", type=Path,
                        default=PORTRAITS_DIR / "default.jpg",
                        help="Source face image for the synthetic streamer")
    parser.add_argument("--input",      type=Path,
                        help="Single gameplay clip to react to")
    parser.add_argument("--output",     type=Path,
                        default=OUT_DIR / "reaction_synth.mp4")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index",      type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--batch_eval", action="store_true",
                        help="Batch: synthesize for eval-game clips (cross-game demo)")
    parser.add_argument("--eval_dir",   type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    parser.add_argument("--max_clips",  type=int,  default=5)
    parser.add_argument("--corner",     type=str,  default="bottom-right",
                        choices=["bottom-right", "bottom-left",
                                 "top-right",    "top-left"])
    args = parser.parse_args()

    if not args.source_portrait.exists():
        print(f"Portrait not found: {args.source_portrait}")
        print("Place a face image at that path, or pass --source_portrait <path>")
        return

    if args.batch_eval:
        batch_eval(args.source_portrait, args.eval_dir, args.out_dir,
                   args.checkpoint, args.index, args.max_clips)
        return

    if args.input is None:
        parser.error("--input required (or use --batch_eval)")

    synthesize_reaction(
        gameplay_path=args.input,
        source_portrait=args.source_portrait,
        out_path=args.output,
        checkpoint=args.checkpoint,
        index_path=args.index,
        corner=args.corner,
    )


if __name__ == "__main__":
    main()

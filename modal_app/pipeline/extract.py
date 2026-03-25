"""
Video extraction pipeline: SAM2 tracking + ViTPose keypoints + Whisper transcription.
Produces JSONL records for Head 1/2 training (Stream C).
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# SAM2 / Subject tracking
# ---------------------------------------------------------------------------

def track_subject_sam2(
    frames: np.ndarray,      # [T, H, W, 3] uint8 RGB
    bbox_frame0: tuple,      # (x1, y1, x2, y2) on frame 0
) -> list[np.ndarray]:       # list of T bool masks [H, W]
    """
    Track a subject across all frames.
    Tries SAM2 first, falls back to OpenCV-based bbox tracker.
    """
    try:
        return _track_sam2(frames, bbox_frame0)
    except Exception as e:
        print(f"SAM2 unavailable ({e}), falling back to OpenCV tracker")
        return _track_opencv(frames, bbox_frame0)


def _track_sam2(frames, bbox_frame0):
    """SAM2-based tracking (requires sam2 package and model weights)."""
    from sam2.build_sam import build_sam2_video_predictor
    model = build_sam2_video_predictor("sam2_hiera_large.pt")
    inference_state = model.init_state_from_frames(frames)
    x1, y1, x2, y2 = bbox_frame0
    model.add_new_points_or_box(
        inference_state, frame_idx=0, obj_id=1,
        box=np.array([x1, y1, x2, y2]),
    )
    masks = [None] * len(frames)
    for frame_idx, obj_ids, mask_logits in model.propagate_in_video(inference_state):
        masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
    return masks


def _make_opencv_tracker():
    """Create an OpenCV tracker, trying multiple APIs for compatibility."""
    for factory in [
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerMIL_create(),
    ]:
        try:
            return factory()
        except AttributeError:
            continue
    raise RuntimeError("No compatible OpenCV tracker found")


def _track_opencv(frames, bbox_frame0):
    """
    OpenCV tracker fallback. Returns approximate bool masks from tracked bbox.
    """
    x1, y1, x2, y2 = bbox_frame0
    H, W = frames[0].shape[:2]
    tracker = _make_opencv_tracker()
    init_bbox = (x1, y1, x2 - x1, y2 - y1)  # OpenCV uses (x, y, w, h)
    tracker.init(frames[0], init_bbox)

    masks = []
    for frame in frames:
        ok, bbox = tracker.update(frame)
        mask = np.zeros((H, W), dtype=bool)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            mask[max(0,y):min(H,y+h), max(0,x):min(W,x+w)] = True
        masks.append(mask)
    return masks


def extract_bboxes_from_masks(masks: list[np.ndarray]) -> list[Optional[tuple]]:
    """Convert binary masks to tight bounding boxes (x1,y1,x2,y2)."""
    bboxes = []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            bboxes.append(None)
        else:
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return bboxes


# ---------------------------------------------------------------------------
# ViTPose / Keypoint extraction
# ---------------------------------------------------------------------------

# COCO-17 keypoint indices used for K-VRC joint mapping
COCO_JOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
}

# Map COCO keypoints to K-VRC joint names
KVRC_JOINT_MAP = {
    "head":        ["nose", "left_eye", "right_eye"],  # average
    "spine2":      ["left_shoulder", "right_shoulder"],  # midpoint
    "left_arm":    ["left_shoulder"],
    "right_arm":   ["right_shoulder"],
    "left_forearm": ["left_elbow"],
    "right_forearm": ["right_elbow"],
}


def extract_upper_body_keypoints(
    frame: np.ndarray,       # [H, W, 3] uint8 RGB
    bbox: tuple,             # (x1, y1, x2, y2)
) -> dict:                   # {"head": [x, y, conf], "spine2": [...], ...}
    """
    Extract K-VRC joint proxies from a single frame.
    Tries ultralytics YOLO-pose first, falls back to geometric heuristics.
    """
    try:
        return _keypoints_yolo(frame, bbox)
    except Exception as e:
        print(f"YOLO-pose unavailable ({e}), using geometric heuristics")
        return _keypoints_heuristic(frame, bbox)


def _keypoints_yolo(frame, bbox):
    """YOLO-pose based keypoint extraction."""
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")  # downloads on first use
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    results = model(crop, verbose=False)
    if not results or results[0].keypoints is None:
        return _keypoints_heuristic(frame, bbox)
    kps = results[0].keypoints.data[0].cpu().numpy()  # [17, 3] (x, y, conf)
    return _coco_kps_to_kvrc(kps, offset=(x1, y1))


def _keypoints_heuristic(frame, bbox):
    """
    Geometric fallback: estimate joint positions from bounding box proportions.
    These are rough approximations, not real pose estimates.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = x1 + w / 2

    return {
        "head":         [cx,                    y1 + h * 0.1,  0.3],
        "spine2":       [cx,                    y1 + h * 0.35, 0.3],
        "left_arm":     [x1 + w * 0.2,          y1 + h * 0.35, 0.3],
        "right_arm":    [x1 + w * 0.8,          y1 + h * 0.35, 0.3],
        "left_forearm": [x1 + w * 0.1,          y1 + h * 0.5,  0.3],
        "right_forearm":[x1 + w * 0.9,          y1 + h * 0.5,  0.3],
    }


def _coco_kps_to_kvrc(kps: np.ndarray, offset: tuple = (0, 0)) -> dict:
    """Convert COCO-17 keypoints to K-VRC joint dict.
    Only averages keypoints with confidence > 0 to avoid pulling towards origin.
    """
    ox, oy = offset
    result = {}
    for joint_name, source_joints in KVRC_JOINT_MAP.items():
        coords = []
        for src in source_joints:
            idx = COCO_JOINTS.get(src)
            if idx is not None and idx < len(kps) and kps[idx][2] > 0:
                coords.append(kps[idx])
        if coords:
            avg = np.mean(coords, axis=0)
            result[joint_name] = [float(avg[0] + ox), float(avg[1] + oy), float(avg[2])]
        else:
            result[joint_name] = [0.0, 0.0, 0.0]
    return result


# ---------------------------------------------------------------------------
# Whisper transcription + frame alignment
# ---------------------------------------------------------------------------

def transcribe_video(video_path: str) -> list[dict]:
    """
    Transcribe audio from video using Whisper.
    Returns list of word-level segments: [{"word": str, "start": float, "end": float}]
    """
    try:
        import whisper
    except ImportError:
        print("whisper not installed, skipping transcription")
        return []

    model = whisper.load_model("base")
    result = model.transcribe(video_path, word_timestamps=True)
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return words


def words_to_frame_text(words: list[dict], frame_idx: int, fps: float, window_sec: float = 2.0) -> str:
    """Get text spoken within `window_sec` before/at the given frame."""
    t = frame_idx / fps
    t_start = t - window_sec
    window_words = [w["word"] for w in words if t_start <= w["start"] <= t]
    return " ".join(window_words).strip()


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str) -> tuple[np.ndarray, float]:
    """Load video frames as [T, H, W, 3] uint8 RGB array. Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


def run_full_extraction(
    video_path: str,
    bbox_frame0: tuple,
    output_dir: str = "data/raw",
    chunk_frames: int = 60,  # ~2s at 30fps
) -> str:
    """
    Full Stream C extraction pipeline:
    1. Load video frames
    2. Transcribe audio (Whisper)
    3. Track subject (SAM2/OpenCV)
    4. Extract bounding boxes
    5. Extract keypoints per frame (ViTPose/heuristic)
    6. Chunk into 60-frame records with text windows
    7. Write JSONL to output_dir

    Returns path to output JSONL file.
    """
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    print(f"Loading {video_path}...")
    frames, fps = load_video_frames(video_path)
    T = len(frames)
    print(f"  {T} frames at {fps:.1f} fps ({T/fps:.1f}s)")

    print("Transcribing audio...")
    words = transcribe_video(video_path)
    print(f"  {len(words)} words transcribed")

    print("Tracking subject...")
    masks = track_subject_sam2(frames, bbox_frame0)
    bboxes = extract_bboxes_from_masks(masks)

    print("Extracting keypoints...")
    keypoints_per_frame = []
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is None:
            bbox = bbox_frame0  # fallback to initial bbox
        kps = extract_upper_body_keypoints(frame, bbox)
        keypoints_per_frame.append(kps)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{T} frames processed")

    print("Building records...")
    records = []
    for start in range(0, T - chunk_frames + 1, chunk_frames):
        end = start + chunk_frames
        kp_sequence = keypoints_per_frame[start:end]
        text_window = words_to_frame_text(words, end, fps)
        records.append({
            "source": "stream_c",
            "video": Path(video_path).name,
            "frame_start": start,
            "frame_end": end,
            "fps": fps,
            "text_window": text_window,
            "keypoint_sequence": kp_sequence,
        })

    out_path = str(Path(output_dir) / f"{video_name}_stream_c.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    return out_path

import json
import os
import numpy as np
from pathlib import Path

JOINT_ORDER = ["head", "spine2", "left_arm", "right_arm", "left_forearm", "right_forearm"]


def normalize_keypoints(
    kp_seq: np.ndarray,  # [T, 6, 2] raw pixel xy
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> np.ndarray:
    """Normalize to [-1, 1] relative to frame dimensions."""
    normed = kp_seq.copy().astype(np.float32)
    normed[:, :, 0] = (normed[:, :, 0] / frame_w) * 2 - 1
    normed[:, :, 1] = (normed[:, :, 1] / frame_h) * 2 - 1
    return np.clip(normed, -1.0, 1.0)


def kp_dict_to_array(kp_dict: dict) -> np.ndarray:
    """Convert keypoint dict to [6, 2] float array in JOINT_ORDER."""
    arr = np.zeros((6, 2), dtype=np.float32)
    for i, joint in enumerate(JOINT_ORDER):
        try:
            val = kp_dict.get(joint)
            if val is not None and len(val) >= 2:
                arr[i, 0] = float(val[0])
                arr[i, 1] = float(val[1])
        except (TypeError, ValueError, IndexError):
            pass  # keep zeros for malformed joint data
    return arr


def build_dataset(raw_dir: str, processed_dir: str) -> dict:
    """
    Merge all JSONL files in raw_dir into processed tensors.
    Returns dict with numpy arrays ready for training.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    texts, keypoints_list = [], []

    for fname in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("source") != "stream_c":
                    continue
                kp_seq = record.get("keypoint_sequence", [])
                if len(kp_seq) < 60:
                    continue
                kp_arr = np.stack([kp_dict_to_array(kp) for kp in kp_seq[:60]])  # [60, 6, 2]
                kp_arr = normalize_keypoints(kp_arr)
                keypoints_list.append(kp_arr)
                texts.append(record.get("text_window", ""))

    ds = {
        "texts": texts,
        "keypoints": np.stack(keypoints_list) if keypoints_list else np.zeros((0, 60, 6, 2), dtype=np.float32),
    }

    np.save(os.path.join(processed_dir, "keypoints.npy"), ds["keypoints"])
    with open(os.path.join(processed_dir, "texts.json"), "w") as f:
        json.dump(texts, f)

    print(f"Dataset built: {len(texts)} records")
    return ds

"""
Stream A dataset builder: reads logged API records (from STREAM_A_LOG_PATH)
and prepares training data for Head 3 (face) and Head 4 (screen).
"""
import json
import numpy as np
from pathlib import Path

from modal_app.heads import FACE_PARAM_NAMES, MOOD_LABELS

# Map the 6 Claude-labeled expression keys to their indices in FACE_PARAM_NAMES
EXPRESSION_KEY_TO_PARAM_IDX = {
    "brow_raise": FACE_PARAM_NAMES.index("brow_raise"),
    "brow_furrow": FACE_PARAM_NAMES.index("brow_furrow"),
    "eye_squint": FACE_PARAM_NAMES.index("eye_squint"),
    "mouth_open": FACE_PARAM_NAMES.index("mouth_open"),
    "smile_width": FACE_PARAM_NAMES.index("smile_width"),
    "glitch_intensity": FACE_PARAM_NAMES.index("glitch_intensity"),
}


def build_face_screen_dataset(raw_dir: str) -> dict:
    """
    Read all JSONL files in raw_dir, filter for Stream A records (those with
    expression_weights), and build training arrays for Head 3 and Head 4.

    Returns dict with:
      texts: list[str]
      face_targets: np.ndarray [N, 32] float32
      mood_targets: np.ndarray [N] int64
      energy_targets: np.ndarray [N] float32
    """
    texts, face_targets, mood_targets, energy_targets = [], [], [], []

    for fname in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip stream_c records (no face labels)
                if rec.get("source") == "stream_c":
                    continue

                # Require expression_weights to be present
                if not rec.get("expression_weights"):
                    continue

                # Build 32-dim face target (default 0.5 for unlabeled dims)
                face = np.full(32, 0.5, dtype=np.float32)
                for key, idx in EXPRESSION_KEY_TO_PARAM_IDX.items():
                    val = rec["expression_weights"].get(key)
                    if val is not None:
                        try:
                            face[idx] = float(np.clip(float(val), 0.0, 1.0))
                        except (TypeError, ValueError):
                            pass  # keep the 0.5 default for this dimension

                # Mood label
                mood_str = rec.get("screen_mood", "")
                mood_idx = MOOD_LABELS.index(mood_str) if mood_str in MOOD_LABELS else 0

                # Energy
                energy_raw = rec.get("motion_energy")
                energy = float(np.clip(energy_raw if energy_raw is not None else 0.5, 0.0, 1.0))

                # Build context text from conversation window
                context = rec.get("context_window", [])
                text = " [SEP] ".join(
                    f"{h['role']}: {h['text']}"
                    for h in context[-3:]
                    if isinstance(h, dict) and "role" in h and "text" in h
                )
                if not text and rec.get("reply"):
                    text = rec["reply"]

                texts.append(text)
                face_targets.append(face)
                mood_targets.append(mood_idx)
                energy_targets.append(energy)

    return {
        "texts": texts,
        "face_targets": np.stack(face_targets) if face_targets else np.zeros((0, 32), dtype=np.float32),
        "mood_targets": np.array(mood_targets, dtype=np.int64),
        "energy_targets": np.array(energy_targets, dtype=np.float32),
    }

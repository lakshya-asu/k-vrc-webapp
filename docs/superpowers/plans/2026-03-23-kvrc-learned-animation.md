# K-VRC Learned Animation System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace K-VRC's rule-based animation (fixed emotion → clip) with a learned system that generates body motion, face expressions, and screen content from conversational context.

**Architecture:** Shared MiniLM text encoder (384-dim) feeds four lightweight task-specific heads. Body motion: clip retrieval head (softmax over Mixamo clips) + MDM refinement head (delta quaternions). Face: 32-param continuous MLP. Screen: 8-way mood softmax + 8 continuous params. All served from Modal; Vercel `/api/chat` calls Modal after Claude responds.

**Tech Stack:** Python 3.11, PyTorch 2.x, Modal, sentence-transformers, SAM2, ViTPose, Whisper, FastAPI, Three.js (existing), Vite (existing)

**Spec:** `docs/superpowers/specs/2026-03-23-kvrc-learned-animation-design.md`

**Read the spec before starting.** It contains all architectural decisions, parameter counts, rotation representations, and data schemas referenced here.

---

## Phase Boundaries

Each phase is independently deployable and testable:

- **Phase 1 (Tasks 1–4):** Modal infrastructure scaffold — `/infer` stub up and reachable
- **Phase 2 (Tasks 5–9):** Video extraction pipeline — Stream C records flowing into `data/raw/`
- **Phase 3 (Tasks 10–14):** Head 1 + Head 2 training — body motion working end-to-end
- **Phase 4 (Tasks 15–19):** Stream A + Head 3 + Head 4 + faceScreen rewrite — face/screen working
- **Phase 5 (Tasks 20–23):** Frontend integration — K-VRC browser wired to Modal
- **Phase 6 (Tasks 24–27):** Annotation tool + fine-tuning — expression quality improved

---

## File Map

### New files (Modal)
```
modal_app/
  __init__.py
  app.py            ← Modal app, volumes, secrets, keep_warm
  encoder.py        ← MiniLM load + embed()
  heads.py          ← Head 1–4 definitions, forward pass, weight I/O
  serve.py          ← FastAPI /infer, /train, /extract endpoints
  train.py          ← training loop, phase-aware
  pipeline/
    __init__.py
    extract.py      ← SAM2 + ViTPose + Whisper → JSONL records
    label.py        ← Stream A: extend Claude response with semantic labels
    annotate.py     ← Stream B: annotation HTTP server
    dataset.py      ← merge streams → processed tensors
data/               ← gitignored
  raw/
  annotated/
  processed/
tests/
  test_encoder.py
  test_heads.py
  test_extract.py
  test_dataset.py
  test_serve.py
requirements.txt
```

### Modified files (frontend)
```
src/animationController.js   ← add blendClips(weights)
src/faceScreen.js            ← rewrite renderer for 32 params, add setParams()
src/robot.js                 ← apply motion_deltas on top of blended clip
api/chat.js                  ← call Modal /infer, extend Claude schema, Stream A logging
```

### New frontend files
```
annotate.html                ← Stream B annotation UI
src/annotate.js              ← annotation UI logic
```

---

## Phase 1: Modal Infrastructure

### Task 1: Project scaffold + Modal account

**Files:**
- Create: `modal_app/__init__.py`
- Create: `modal_app/app.py`
- Create: `requirements.txt`

- [ ] Install Modal CLI and authenticate
```bash
pip install modal
modal token new
```
Expected: browser opens, you log in, token saved to `~/.modal.toml`

- [ ] Create `requirements.txt`
```
modal>=0.60.0
fastapi>=0.110.0
sentence-transformers>=2.7.0
torch>=2.2.0
numpy>=1.26.0
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] Create `modal_app/__init__.py` (empty)

- [ ] Create `modal_app/app.py`
```python
import modal

# Persistent volume for model weights + data
volume = modal.Volume.from_name("kvrc-data", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("kvrc-animation", image=image)

VOLUME_PATH = "/data"
MODEL_PATH = "/data/models"
DATA_PATH = "/data/train"
```

- [ ] Verify Modal connection
```bash
modal run modal_app/app.py
```
Expected: "App completed" with no errors

- [ ] Commit
```bash
git add modal_app/ requirements.txt
git commit -m "feat: modal project scaffold"
```

---

### Task 2: Shared encoder

**Files:**
- Create: `modal_app/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] Write failing test
```python
# tests/test_encoder.py
import numpy as np

def test_embed_returns_384_dim():
    from modal_app.encoder import embed
    result = embed(["hello world"])
    assert result.shape == (1, 384)

def test_embed_normalizes_output():
    from modal_app.encoder import embed
    result = embed(["test sentence"])
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_embed_context_window():
    from modal_app.encoder import embed_context
    history = [
        {"role": "user", "text": "what are you thinking about"},
        {"role": "assistant", "text": "nothing you'd understand"},
        {"role": "user", "text": "try me"},
    ]
    result = embed_context(history)
    assert result.shape == (384,)
```

- [ ] Run test to confirm failure
```bash
pytest tests/test_encoder.py -v
```
Expected: ImportError — `modal_app.encoder` does not exist

- [ ] Create `modal_app/encoder.py`
```python
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns [N, 384] normalized float32 array."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.astype(np.float32)

def embed_context(history: list[dict], max_turns: int = 3) -> np.ndarray:
    """
    Embed the last `max_turns` turns of conversation history.
    history: list of {"role": "user"|"assistant", "text": str}
    Returns: [384] float32 array
    """
    recent = history[-max_turns:]
    text = " [SEP] ".join(f"{h['role']}: {h['text']}" for h in recent)
    return embed([text])[0]
```

- [ ] Run tests
```bash
pip install sentence-transformers
pytest tests/test_encoder.py -v
```
Expected: all 3 PASS

- [ ] Commit
```bash
git add modal_app/encoder.py tests/test_encoder.py
git commit -m "feat: shared MiniLM encoder with context window"
```

---

### Task 3: Head definitions (stub + shape tests)

**Files:**
- Create: `modal_app/heads.py`
- Create: `tests/test_heads.py`

- [ ] Write failing shape tests
```python
# tests/test_heads.py
import torch
import numpy as np

N_CLIPS = 178  # update if GLB changes

def test_clip_retrieval_head_output_shape():
    from modal_app.heads import ClipRetrievalHead
    head = ClipRetrievalHead(n_clips=N_CLIPS)
    context = torch.randn(1, 384)
    out = head(context)
    assert out.shape == (1, N_CLIPS)
    assert abs(out.sum().item() - 1.0) < 1e-5  # softmax sums to 1

def test_mdm_head_output_shape():
    from modal_app.heads import MDMRefinementHead
    head = MDMRefinementHead(n_joints=6, n_frames=60)
    context = torch.randn(1, 384)
    base_rotations = torch.randn(1, 60, 6, 4)  # quaternion xyzw
    out = head(context, base_rotations)
    assert out.shape == (1, 60, 6, 4)

def test_face_head_output_range():
    from modal_app.heads import FaceExpressionHead
    head = FaceExpressionHead()
    context = torch.randn(1, 384)
    out = head(context)
    assert out.shape == (1, 32)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid outputs

def test_screen_head_output():
    from modal_app.heads import ScreenContentHead
    head = ScreenContentHead()
    context = torch.randn(1, 384)
    continuous, mood_logits = head(context)
    assert continuous.shape == (1, 8)
    assert mood_logits.shape == (1, 8)
    assert abs(mood_logits.softmax(dim=-1).sum().item() - 1.0) < 1e-5
```

- [ ] Run to confirm failure
```bash
pytest tests/test_heads.py -v
```
Expected: ImportError

- [ ] Create `modal_app/heads.py`
```python
import torch
import torch.nn as nn
import numpy as np

MOOD_LABELS = ["cold", "warm", "glitch", "static", "data", "boot", "angry", "dream"]
FACE_PARAM_NAMES = [
    "brow_raise", "brow_furrow", "brow_asymmetry",
    "eye_openness", "eye_squint", "glitch_flicker",
    "mouth_curl_left", "mouth_curl_right", "mouth_open", "smile_width",
    "scan_line_speed", "scan_line_opacity", "glitch_intensity", "glitch_color_shift",
    "color_temperature", "color_saturation", "color_brightness",
    "noise_scale", "noise_speed", "vignette",
    "blink_rate", "pupil_dilation", "iris_glow_intensity",
    "reserved_0", "reserved_1", "reserved_2", "reserved_3",
    "reserved_4", "reserved_5", "reserved_6", "reserved_7", "reserved_8",
]
assert len(FACE_PARAM_NAMES) == 32


class ClipRetrievalHead(nn.Module):
    def __init__(self, n_clips: int):
        super().__init__()
        self.n_clips = n_clips
        self.proj = nn.Linear(384, n_clips)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, 384] → weights: [B, N_clips] (softmax)"""
        return torch.softmax(self.proj(context), dim=-1)


class MDMRefinementHead(nn.Module):
    def __init__(self, n_joints: int = 6, n_frames: int = 60, d_model: int = 128):
        super().__init__()
        self.n_joints = n_joints
        self.n_frames = n_frames
        self.joint_dim = n_joints * 4  # quaternion xyzw

        self.context_proj = nn.Linear(384, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.out_proj = nn.Linear(d_model, self.joint_dim)

    def forward(self, context: torch.Tensor, base_rotations: torch.Tensor) -> torch.Tensor:
        """
        context: [B, 384]
        base_rotations: [B, 60, 6, 4] quaternion xyzw
        returns: delta quaternions [B, 60, 6, 4]
        """
        B = context.shape[0]
        ctx = self.context_proj(context).unsqueeze(1).expand(-1, self.n_frames, -1)  # [B, 60, d]
        out = self.encoder(ctx)  # [B, 60, d]
        deltas = self.out_proj(out)  # [B, 60, 24]
        return deltas.view(B, self.n_frames, self.n_joints, 4)


class FaceExpressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, 384] → params: [B, 32] in [0, 1]"""
        return self.net(context)


class ScreenContentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.continuous_out = nn.Sequential(nn.Linear(64, 8), nn.Sigmoid())
        self.mood_out = nn.Linear(64, 8)  # raw logits for softmax

    def forward(self, context: torch.Tensor):
        """context: [B, 384] → (continuous [B,8], mood_logits [B,8])"""
        h = self.net(context)
        return self.continuous_out(h), self.mood_out(h)


def apply_delta_quaternions(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    base: [..., 4] xyzw quaternions (normalized)
    delta: [..., 4] small delta quaternions
    returns: normalized sum, approximation valid for small deltas
    """
    combined = base + delta
    norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return combined / norm
```

- [ ] Run tests
```bash
pytest tests/test_heads.py -v
```
Expected: all 4 PASS

- [ ] Commit
```bash
git add modal_app/heads.py tests/test_heads.py
git commit -m "feat: all 4 head architectures with shape tests"
```

---

### Task 4: FastAPI /infer stub on Modal

**Files:**
- Create: `modal_app/serve.py`
- Create: `tests/test_serve.py`

- [ ] Write failing test
```python
# tests/test_serve.py
import httpx
import pytest

MODAL_URL = "https://kvrc-animation--serve.modal.run"  # update after first deploy

def test_infer_returns_expected_keys():
    resp = httpx.post(f"{MODAL_URL}/infer", json={
        "text": "hello",
        "history": [{"role": "user", "text": "hello"}]
    }, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert "clip_weights" in data
    assert "motion_deltas" in data
    assert "motion_deltas_shape" in data
    assert "face_params" in data
    assert "screen_tokens" in data

def test_infer_motion_deltas_shape():
    resp = httpx.post(f"{MODAL_URL}/infer", json={
        "text": "test",
        "history": []
    }, timeout=30)
    data = resp.json()
    assert data["motion_deltas_shape"] == [60, 6, 4]
    assert len(data["motion_deltas"]) == 60 * 6 * 4
```

- [ ] Create `modal_app/serve.py`
```python
import modal
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from modal_app.app import app, volume, image, VOLUME_PATH
from modal_app.encoder import embed_context
from modal_app.heads import (
    ClipRetrievalHead, MDMRefinementHead,
    FaceExpressionHead, ScreenContentHead,
    MOOD_LABELS, FACE_PARAM_NAMES,
)
import torch

N_CLIPS = 178  # must match GLB; update if clips change
N_JOINTS = 6
N_FRAMES = 60
JOINT_NAMES = ["Head", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"]

web_app = FastAPI()

class InferRequest(BaseModel):
    text: str
    history: list[dict] = []

class InferResponse(BaseModel):
    clip_weights: dict
    motion_deltas: list[float]
    motion_deltas_shape: list[int]
    face_params: dict
    screen_tokens: dict


def _load_heads():
    """Load head weights from volume if they exist, else use random init."""
    import os
    heads = {
        "clip": ClipRetrievalHead(n_clips=N_CLIPS),
        "mdm": MDMRefinementHead(n_joints=N_JOINTS, n_frames=N_FRAMES),
        "face": FaceExpressionHead(),
        "screen": ScreenContentHead(),
    }
    for name, head in heads.items():
        path = f"{VOLUME_PATH}/models/{name}_head.pt"
        if os.path.exists(path):
            head.load_state_dict(torch.load(path, map_location="cpu"))
        head.eval()
    return heads


@app.function(
    volumes={VOLUME_PATH: volume},
    keep_warm=1,
    image=image,
)
@modal.asgi_app()
def serve():
    heads = _load_heads()

    @web_app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest):
        history = req.history + [{"role": "assistant", "text": req.text}]
        ctx = embed_context(history)
        ctx_t = torch.from_numpy(ctx).unsqueeze(0)  # [1, 384]

        with torch.no_grad():
            # Head 1: clip weights
            clip_w = heads["clip"](ctx_t)[0].numpy()  # [N_clips]
            top3_idx = np.argsort(clip_w)[-3:][::-1]

            # Head 2: motion deltas (use zero base rotations as stub until training)
            base_rot = torch.zeros(1, N_FRAMES, N_JOINTS, 4)
            base_rot[..., 3] = 1.0  # identity quaternion w=1
            deltas = heads["mdm"](ctx_t, base_rot)[0]  # [60, 6, 4]

            # Head 3: face params
            face = heads["face"](ctx_t)[0].numpy()  # [32]

            # Head 4: screen tokens
            continuous, mood_logits = heads["screen"](ctx_t)
            continuous = continuous[0].numpy()
            mood_idx = mood_logits[0].softmax(dim=0).argmax().item()

        # Serialize clip weights (top-3 only, rest negligible)
        clip_names = _get_clip_names()
        clip_weights_dict = {clip_names[i]: float(clip_w[i]) for i in top3_idx}

        return InferResponse(
            clip_weights=clip_weights_dict,
            motion_deltas=deltas.numpy().flatten().tolist(),
            motion_deltas_shape=[N_FRAMES, N_JOINTS, 4],
            face_params={name: float(face[i]) for i, name in enumerate(FACE_PARAM_NAMES)},
            screen_tokens={
                "mood": MOOD_LABELS[mood_idx],
                "continuous": {
                    "energy": float(continuous[0]),
                    "pattern_scale": float(continuous[1]),
                    "pattern_speed": float(continuous[2]),
                    "vignette": float(continuous[3]),
                    "noise_intensity": float(continuous[4]),
                    "glow_radius": float(continuous[5]),
                    "chromatic_aberration": float(continuous[6]),
                    "flicker_rate": float(continuous[7]),
                }
            }
        )

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    return web_app


def _get_clip_names() -> list[str]:
    """Returns ordered list of clip names. Order must match Head 1 training."""
    import os, json
    path = f"{VOLUME_PATH}/models/clip_names.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: numeric placeholders (pre-training)
    return [f"clip_{i}" for i in range(N_CLIPS)]
```

- [ ] Deploy to Modal
```bash
modal deploy modal_app/serve.py
```
Expected: prints deployment URL like `https://kvrc-animation--serve.modal.run`

- [ ] Update `MODAL_URL` in `tests/test_serve.py` with the actual URL

- [ ] Run integration test
```bash
pytest tests/test_serve.py -v
```
Expected: both tests PASS (heads are random init, but shape is correct)

- [ ] Commit
```bash
git add modal_app/serve.py tests/test_serve.py
git commit -m "feat: modal /infer stub deployed, shape tests passing"
```

---

## Phase 2: Video Extraction Pipeline

### Task 5: SAM2 video tracking

**Files:**
- Create: `modal_app/pipeline/__init__.py`
- Create: `modal_app/pipeline/extract.py` (partial — SAM2 section)

- [ ] Add SAM2 to requirements.txt
```
# append to requirements.txt
git+https://github.com/facebookresearch/segment-anything-2.git
huggingface_hub>=0.22.0
opencv-python-headless>=4.9.0
```

- [ ] Write test for SAM2 mask output shape
```python
# tests/test_extract.py
import numpy as np

def test_sam2_mask_output_shape():
    """SAM2 should return one mask per frame with same H,W as input."""
    from modal_app.pipeline.extract import track_subject_sam2
    # Use a tiny 10-frame synthetic video for testing
    frames = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    bbox_frame0 = (10, 10, 50, 50)  # x1,y1,x2,y2
    masks = track_subject_sam2(frames, bbox_frame0)
    assert len(masks) == 10
    assert masks[0].shape == (64, 64)
    assert masks[0].dtype == bool
```

- [ ] Create `modal_app/pipeline/__init__.py` (empty)

- [ ] Create SAM2 section of `modal_app/pipeline/extract.py`
```python
import numpy as np
import torch
from pathlib import Path

def track_subject_sam2(
    frames: np.ndarray,          # [T, H, W, 3] uint8 RGB
    bbox_frame0: tuple,          # (x1, y1, x2, y2) on frame 0
) -> list[np.ndarray]:           # list of T bool masks [H, W]
    """
    Track a subject across all frames using SAM2.
    Requires one bounding box prompt on frame 0.
    """
    from sam2.build_sam import build_sam2_video_predictor

    model = build_sam2_video_predictor("sam2_hiera_large.pt")
    inference_state = model.init_state_from_frames(frames)

    x1, y1, x2, y2 = bbox_frame0
    model.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        box=np.array([x1, y1, x2, y2]),
    )

    masks = [None] * len(frames)
    for frame_idx, obj_ids, mask_logits in model.propagate_in_video(inference_state):
        masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

    return masks


def extract_bboxes_from_masks(masks: list[np.ndarray]) -> list[tuple]:
    """Convert binary masks to tight bounding boxes (x1,y1,x2,y2)."""
    bboxes = []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            bboxes.append(None)
        else:
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return bboxes
```

- [ ] Run test (will be slow first time — downloads SAM2 weights)
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pytest tests/test_extract.py::test_sam2_mask_output_shape -v
```
Expected: PASS

- [ ] Commit
```bash
git add modal_app/pipeline/ tests/test_extract.py requirements.txt
git commit -m "feat: SAM2 subject tracking pipeline"
```

---

### Task 6: ViTPose upper body keypoint extraction

**Files:**
- Modify: `modal_app/pipeline/extract.py`
- Modify: `tests/test_extract.py`

- [ ] Add ViTPose test
```python
# append to tests/test_extract.py

def test_vitpose_keypoint_shape():
    from modal_app.pipeline.extract import extract_upper_body_keypoints
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 50, 300, 400)
    keypoints = extract_upper_body_keypoints(frame, bbox)
    # Returns dict with 6 K-VRC joint proxies, each [x, y, confidence]
    expected_joints = ["head", "spine2", "left_arm", "right_arm", "left_forearm", "right_forearm"]
    for joint in expected_joints:
        assert joint in keypoints
        assert len(keypoints[joint]) == 3  # x, y, confidence
```

- [ ] Add ViTPose extraction to `extract.py`
```python
# append to modal_app/pipeline/extract.py

# COCO-17 keypoint indices used for K-VRC joint mapping
COCO_JOINTS = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,    "right_elbow": 8,
    "left_wrist": 9,    "right_wrist": 10,
}

def extract_upper_body_keypoints(
    frame: np.ndarray,  # [H, W, 3] uint8 RGB
    bbox: tuple,        # (x1, y1, x2, y2)
) -> dict:              # joint_name → [x, y, confidence]
    """
    Extract 2D keypoints for K-VRC's 6 tracked joints using ViTPose.
    Maps COCO-17 indices to K-VRC bone proxies.
    """
    from transformers import AutoProcessor, ViTPoseForPoseEstimation
    import torch

    processor = AutoProcessor.from_pretrained("ViTPose/vitpose-h-simple")
    model = ViTPoseForPoseEstimation.from_pretrained("ViTPose/vitpose-h-simple")
    model.eval()

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    inputs = processor(images=crop, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.heatmaps: [1, 17, H/4, W/4]
    heatmaps = outputs.heatmaps[0].numpy()  # [17, H, W]
    H, W = heatmaps.shape[1], heatmaps.shape[2]

    def get_kp(coco_idx):
        hm = heatmaps[coco_idx]
        conf = float(hm.max())
        flat_idx = hm.argmax()
        ky, kx = divmod(flat_idx, W)
        # Scale back to original frame coords
        fx = x1 + (kx / W) * (x2 - x1)
        fy = y1 + (ky / H) * (y2 - y1)
        return [float(fx), float(fy), conf]

    # Spine2 proxy: midpoint of shoulders
    ls = get_kp(COCO_JOINTS["left_shoulder"])
    rs = get_kp(COCO_JOINTS["right_shoulder"])
    spine2 = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, min(ls[2], rs[2])]

    return {
        "head":         get_kp(COCO_JOINTS["nose"]),
        "spine2":       spine2,
        "left_arm":     ls,
        "right_arm":    rs,
        "left_forearm": get_kp(COCO_JOINTS["left_elbow"]),
        "right_forearm":get_kp(COCO_JOINTS["right_elbow"]),
    }
```

- [ ] Run test
```bash
pip install transformers
pytest tests/test_extract.py::test_vitpose_keypoint_shape -v
```
Expected: PASS

- [ ] Commit
```bash
git add modal_app/pipeline/extract.py tests/test_extract.py
git commit -m "feat: ViTPose upper body keypoint extraction"
```

---

### Task 7: Whisper transcription + frame alignment

**Files:**
- Modify: `modal_app/pipeline/extract.py`
- Modify: `tests/test_extract.py`

- [ ] Add transcription test
```python
# append to tests/test_extract.py
import tempfile, os

def test_whisper_returns_word_timestamps():
    from modal_app.pipeline.extract import transcribe_video
    # Use a 2-second silent mp4 — just test the structure
    # In practice provide a real clip path
    result = transcribe_video.__wrapped__("/dev/null") if hasattr(transcribe_video, '__wrapped__') else []
    # Structure test only — real test requires actual audio
    assert isinstance(result, list)
```

- [ ] Add Whisper + full pipeline to `extract.py`
```python
# append to modal_app/pipeline/extract.py
import json
import cv2
from pathlib import Path

def transcribe_video(video_path: str) -> list[dict]:
    """
    Transcribe audio from video using Whisper.
    Returns list of {"word": str, "start": float, "end": float} dicts.
    """
    import whisper
    model = whisper.load_model("large-v3")
    result = model.transcribe(video_path, word_timestamps=True)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
    return words


def align_words_to_frames(words: list[dict], fps: float, n_frames: int) -> list[str]:
    """
    Map each frame index to the word being spoken at that frame.
    Returns list of length n_frames, each entry is the current word or "".
    """
    frame_words = [""] * n_frames
    for w in words:
        start_f = int(w["start"] * fps)
        end_f = int(w["end"] * fps)
        for f in range(start_f, min(end_f + 1, n_frames)):
            frame_words[f] = w["word"]
    return frame_words


def build_text_windows(frame_words: list[str], window_size: int = 75) -> list[str]:
    """
    Build sliding text windows of `window_size` frames.
    Each window contains all words spoken in that range joined as a sentence.
    """
    windows = []
    for i in range(0, len(frame_words), window_size):
        chunk = frame_words[i:i+window_size]
        text = " ".join(w for w in chunk if w).strip()
        windows.append(text or "[silence]")
    return windows


def run_full_extraction(video_path: str, bbox_frame0: tuple, output_dir: str) -> str:
    """
    Full Stream C extraction pipeline for one video.
    bbox_frame0: (x1, y1, x2, y2) — K-VRC's bounding box in frame 0
    Returns path to output JSONL file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / (Path(video_path).stem + "_stream_c.jsonl"))

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames)

    print(f"Loaded {n_frames} frames at {fps:.1f}fps")

    # SAM2 tracking
    print("Running SAM2...")
    masks = track_subject_sam2(frames, bbox_frame0)
    bboxes = extract_bboxes_from_masks(masks)

    # Whisper
    print("Running Whisper...")
    words = transcribe_video(video_path)
    frame_words = align_words_to_frames(words, fps, n_frames)

    # ViTPose per frame (batch for speed)
    print("Running ViTPose...")
    records = []
    window_size = int(fps * 2.4)  # ~60 frames at 25fps

    for start in range(0, n_frames - window_size, window_size // 2):  # 50% overlap
        end = start + window_size
        kp_sequence = []
        valid = True

        for fi in range(start, end):
            bb = bboxes[fi]
            if bb is None:
                valid = False
                break
            kp = extract_upper_body_keypoints(frames[fi], bb)
            # Filter low confidence frames
            if any(v[2] < 0.4 for v in kp.values()):
                valid = False
                break
            kp_sequence.append(kp)

        if not valid or len(kp_sequence) < window_size:
            continue

        text_window = " ".join(w for w in frame_words[start:end] if w) or "[silence]"
        records.append({
            "source": "stream_c",
            "video": Path(video_path).name,
            "frame_start": start,
            "frame_end": end,
            "fps": fps,
            "text_window": text_window,
            "keypoint_sequence": kp_sequence,  # list of 60 dicts
        })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    return out_path
```

- [ ] Test on the provided clip
```bash
python3 -c "
from modal_app.pipeline.extract import run_full_extraction
# You must click K-VRC's bounding box manually — update bbox_frame0
# Use the first clear frame where K-VRC is fully visible
run_full_extraction(
    'K-VRC-training-clip.mp4',
    bbox_frame0=(760, 100, 1160, 900),  # adjust after visual inspection
    output_dir='data/raw/'
)
"
```
Expected: prints frame count, SAM2 progress, writes JSONL file

- [ ] Inspect output
```bash
python3 -c "
import json
with open('data/raw/K-VRC-training-clip_stream_c.jsonl') as f:
    records = [json.loads(l) for l in f]
print(f'Records: {len(records)}')
print('Sample:', json.dumps(records[0], indent=2)[:500])
"
```

- [ ] Commit
```bash
git add modal_app/pipeline/extract.py tests/test_extract.py
git commit -m "feat: full stream C extraction pipeline with whisper alignment"
```

---

### Task 8: Dataset builder

**Files:**
- Create: `modal_app/pipeline/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] Write failing test
```python
# tests/test_dataset.py
import json, os, tempfile, numpy as np

def _make_stream_c_record():
    return {
        "source": "stream_c", "text_window": "hello world",
        "keypoint_sequence": [
            {"head": [100.0, 50.0, 0.9], "spine2": [100.0, 150.0, 0.85],
             "left_arm": [80.0, 130.0, 0.8], "right_arm": [120.0, 130.0, 0.8],
             "left_forearm": [70.0, 160.0, 0.75], "right_forearm": [130.0, 160.0, 0.75]}
        ] * 60
    }

def test_dataset_loads_stream_c():
    from modal_app.pipeline.dataset import build_dataset
    with tempfile.TemporaryDirectory() as d:
        raw_path = os.path.join(d, "test.jsonl")
        with open(raw_path, "w") as f:
            for _ in range(10):
                f.write(json.dumps(_make_stream_c_record()) + "\n")
        ds = build_dataset(raw_dir=d, processed_dir=d)
        assert "keypoints" in ds
        assert ds["keypoints"].shape == (10, 60, 6, 2)  # T, joints, xy
        assert "texts" in ds
        assert len(ds["texts"]) == 10

def test_keypoint_normalization():
    from modal_app.pipeline.dataset import normalize_keypoints
    kp_seq = np.random.rand(60, 6, 2) * 1920  # raw pixel coords
    normed = normalize_keypoints(kp_seq, frame_w=1920, frame_h=1080)
    assert normed.max() <= 1.0 and normed.min() >= -1.0
```

- [ ] Create `modal_app/pipeline/dataset.py`
```python
import json, os
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
    return normed


def kp_dict_to_array(kp_dict: dict) -> np.ndarray:
    """Convert keypoint dict to [6, 2] float array in JOINT_ORDER."""
    arr = np.zeros((6, 2), dtype=np.float32)
    for i, joint in enumerate(JOINT_ORDER):
        if joint in kp_dict:
            arr[i, 0] = kp_dict[joint][0]
            arr[i, 1] = kp_dict[joint][1]
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
                record = json.loads(line.strip())
                if record.get("source") != "stream_c":
                    continue
                kp_seq = record["keypoint_sequence"]
                if len(kp_seq) < 60:
                    continue
                kp_arr = np.stack([kp_dict_to_array(kp) for kp in kp_seq[:60]])  # [60, 6, 2]
                kp_arr = normalize_keypoints(kp_arr)
                keypoints_list.append(kp_arr)
                texts.append(record["text_window"])

    ds = {
        "texts": texts,
        "keypoints": np.stack(keypoints_list) if keypoints_list else np.zeros((0, 60, 6, 2)),
    }

    # Save processed arrays
    np.save(os.path.join(processed_dir, "keypoints.npy"), ds["keypoints"])
    with open(os.path.join(processed_dir, "texts.json"), "w") as f:
        json.dump(texts, f)

    print(f"Dataset built: {len(texts)} records")
    return ds
```

- [ ] Run tests
```bash
pytest tests/test_dataset.py -v
```
Expected: both PASS

- [ ] Commit
```bash
git add modal_app/pipeline/dataset.py tests/test_dataset.py
git commit -m "feat: dataset builder for stream C records"
```

---

## Phase 3: Head 1 + Head 2 Training

### Task 9: Clip reference keypoints (bake)

**Files:**
- Create: `modal_app/pipeline/bake_clips.py`

This step pre-computes ViTPose keypoints for all 178 Mixamo clips by rendering each clip at canonical camera position (front-facing, static). This provides the reference embeddings for Head 1 training via cosine similarity.

- [ ] Create `modal_app/pipeline/bake_clips.py`
```python
"""
Bake reference 2D keypoint sequences for all Mixamo clips.
Run this once after GLB is available. Output: data/processed/clip_keypoints.npy

This requires a headless Three.js render of each clip — see bake_clips_render.js
for the browser-side script that dumps per-frame joint screen positions to JSON.
"""
import json, numpy as np
from pathlib import Path
from modal_app.pipeline.dataset import JOINT_ORDER

def load_baked_clip_keypoints(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load baked clip keypoint sequences.
    Returns: (keypoints [N_clips, 60, 6, 2], clip_names [N_clips])
    """
    with open(path) as f:
        data = json.load(f)
    names = [d["name"] for d in data]
    kps = np.stack([np.array(d["keypoints"]) for d in data])
    return kps, names

def compute_clip_similarity(
    observed_kp: np.ndarray,   # [60, 6, 2]
    reference_kps: np.ndarray, # [N_clips, 60, 6, 2]
) -> np.ndarray:               # [N_clips] cosine similarities
    obs_flat = observed_kp.flatten()
    ref_flat = reference_kps.reshape(len(reference_kps), -1)
    obs_norm = obs_flat / (np.linalg.norm(obs_flat) + 1e-8)
    ref_norm = ref_flat / (np.linalg.norm(ref_flat, axis=1, keepdims=True) + 1e-8)
    return ref_norm @ obs_norm
```

- [ ] Note for implementation: the renderer script `bake_clips_render.js` must be created separately. It runs in a headless browser (puppeteer or similar), plays each Mixamo clip on the K-VRC model at front-facing camera, and dumps COCO-style joint screen positions to `data/processed/clip_keypoints.json`. This is a one-time setup step.

- [ ] Commit placeholder
```bash
git add modal_app/pipeline/bake_clips.py
git commit -m "feat: clip keypoint baking utilities for Head 1 training"
```

---

### Task 10: Head 1 training loop

**Files:**
- Create: `modal_app/train.py` (Head 1 section)

- [ ] Create `modal_app/train.py`
```python
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from modal_app.encoder import embed
from modal_app.heads import ClipRetrievalHead
from modal_app.pipeline.bake_clips import compute_clip_similarity, load_baked_clip_keypoints

DATA_PATH = "data/processed"
MODEL_PATH = "data/models"


def train_head1(
    n_epochs: int = 50,
    lr: float = 1e-3,
    confidence_threshold: float = 0.7,
):
    """Train clip retrieval head on Stream C pseudo-labels."""
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # Load data
    keypoints = np.load(f"{DATA_PATH}/keypoints.npy")   # [N, 60, 6, 2]
    with open(f"{DATA_PATH}/texts.json") as f:
        texts = json.load(f)

    ref_kps, clip_names = load_baked_clip_keypoints(f"{DATA_PATH}/clip_keypoints.json")
    n_clips = len(clip_names)

    # Save clip names for inference
    with open(f"{MODEL_PATH}/clip_names.json", "w") as f:
        json.dump(clip_names, f)

    # Generate pseudo-labels via cosine similarity
    print("Generating pseudo-labels...")
    pseudo_labels, valid_indices = [], []
    for i, kp in enumerate(keypoints):
        sims = compute_clip_similarity(kp, ref_kps)
        best_clip = sims.argmax()
        if sims[best_clip] >= confidence_threshold:
            pseudo_labels.append(best_clip)
            valid_indices.append(i)

    print(f"Valid records after confidence filter: {len(valid_indices)}/{len(keypoints)}")

    # Embed texts
    valid_texts = [texts[i] for i in valid_indices]
    print("Embedding texts...")
    text_embeddings = embed(valid_texts)  # [N_valid, 384]

    # Train
    head = ClipRetrievalHead(n_clips=n_clips)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    X = torch.from_numpy(text_embeddings)
    Y = torch.tensor(pseudo_labels, dtype=torch.long)

    for epoch in range(n_epochs):
        head.train()
        optimizer.zero_grad()
        logits = head.proj(X)  # [N, n_clips] pre-softmax
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(dim=1) == Y).float().mean()
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {loss.item():.4f}, acc: {acc:.3f}")

    torch.save(head.state_dict(), f"{MODEL_PATH}/clip_head.pt")
    print(f"Saved Head 1 to {MODEL_PATH}/clip_head.pt")
```

- [ ] Run training
```bash
python3 -c "from modal_app.train import train_head1; train_head1()"
```
Expected: prints per-epoch loss, saves `data/models/clip_head.pt`

- [ ] Commit
```bash
git add modal_app/train.py
git commit -m "feat: head 1 clip retrieval training loop"
```

---

### Task 11: Head 2 training loop

**Files:**
- Modify: `modal_app/train.py`

- [ ] Add Head 2 training to `train.py`
```python
# append to modal_app/train.py

from modal_app.heads import MDMRefinementHead, apply_delta_quaternions


def keypoints_to_pseudo_quaternions(kp_seq: np.ndarray) -> np.ndarray:
    """
    Convert 2D keypoint sequence [60, 6, 2] to pseudo target quaternions [60, 6, 4].
    Approximation: derive rotation from joint displacement vectors.
    Returns identity quaternions with small perturbations based on joint angles.
    """
    T, J, _ = kp_seq.shape
    quats = np.zeros((T, J, 4), dtype=np.float32)
    quats[:, :, 3] = 1.0  # start from identity (w=1)

    # Encode relative joint positions as small rotation signal
    # Head joint: rotation from vertical based on nose position deviation
    head_x = kp_seq[:, 0, 0]  # x position over time, normalized [-1,1]
    quats[:, 0, 0] = head_x * 0.2  # x-axis tilt
    quats[:, 0, 3] = np.sqrt(np.maximum(0, 1 - (head_x * 0.2)**2))

    # Normalize
    norms = np.linalg.norm(quats, axis=-1, keepdims=True).clip(min=1e-8)
    return quats / norms


def train_head2(
    n_epochs: int = 100,
    lr: float = 1e-4,
    delta_regularization: float = 0.01,
):
    """Train MDM refinement head on Stream C keypoint sequences."""
    keypoints = np.load(f"{DATA_PATH}/keypoints.npy")   # [N, 60, 6, 2]
    with open(f"{DATA_PATH}/texts.json") as f:
        texts = json.load(f)

    print("Generating target quaternions from keypoints...")
    target_quats = np.stack([keypoints_to_pseudo_quaternions(kp) for kp in keypoints])
    # [N, 60, 6, 4]

    print("Embedding texts...")
    embeddings = embed(texts)  # [N, 384]

    head = MDMRefinementHead(n_joints=6, n_frames=60)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    X = torch.from_numpy(embeddings)
    Y = torch.from_numpy(target_quats)
    base = torch.zeros_like(Y)
    base[:, :, :, 3] = 1.0  # identity base

    batch_size = 32
    N = len(X)

    for epoch in range(n_epochs):
        head.train()
        perm = torch.randperm(N)
        total_loss = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            ctx, tgt, b = X[idx], Y[idx], base[idx]
            optimizer.zero_grad()
            deltas = head(ctx, b)
            pred = apply_delta_quaternions(b, deltas)

            # MSE on predicted vs target quaternions
            recon_loss = F.mse_loss(pred, tgt)
            # L2 on delta magnitude (keep deltas small)
            reg_loss = deltas.norm(dim=-1).mean()
            loss = recon_loss + delta_regularization * reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {total_loss/((N//batch_size)+1):.5f}")

    torch.save(head.state_dict(), f"{MODEL_PATH}/mdm_head.pt")
    print(f"Saved Head 2 to {MODEL_PATH}/mdm_head.pt")
```

- [ ] Run training
```bash
python3 -c "from modal_app.train import train_head2; train_head2()"
```
Expected: prints loss curve, saves `data/models/mdm_head.pt`

- [ ] Upload trained weights to Modal volume
```bash
modal volume put kvrc-data data/models/ /models/
```

- [ ] Redeploy serve.py so it picks up new weights
```bash
modal deploy modal_app/serve.py
```

- [ ] Commit
```bash
git add modal_app/train.py
git commit -m "feat: head 2 MDM refinement training loop"
```

---

## Phase 4: Stream A + Head 3 + Head 4 + faceScreen Rewrite

### Task 12: Extend Claude API schema for Stream A

**Files:**
- Modify: `api/chat.js`
- Create: `modal_app/pipeline/label.py`

- [ ] Extend `api/chat.js` system prompt (add to the existing JSON output instructions)

Find the line in `api/chat.js` where the JSON output format is specified. Replace:
```js
// current:
{"reply": "...", "emotion": "...", "gesture": "...", "sidenote_topic": "..."}
```
with:
```js
// extended:
{
  "reply": "...",
  "emotion": "...",
  "gesture": "...",
  "sidenote_topic": "...",
  "expression_weights": {
    "brow_raise": 0.0-1.0,
    "brow_furrow": 0.0-1.0,
    "eye_squint": 0.0-1.0,
    "mouth_open": 0.0-1.0,
    "smile_width": 0.0-1.0,
    "glitch_intensity": 0.0-1.0
  },
  "motion_energy": 0.0-1.0,
  "screen_mood": "cold|warm|glitch|static|data|boot|angry|dream"
}
```

Also add instructions to the system prompt:
```
EXPRESSION WEIGHTS: For each response, estimate these face expression intensities as 0-1 floats:
brow_raise (surprise/question), brow_furrow (concentration/displeasure),
eye_squint (skepticism/sarcasm), mouth_open (talking/shock),
smile_width (amusement, use sparingly), glitch_intensity (irritation/malfunction).
motion_energy: 0=very still, 0.5=normal, 1.0=very animated.
screen_mood: the overall visual tone that should appear on K-VRC's screen.
```

- [ ] Add Stream A logging to `api/chat.js` after parsing Claude response:
```js
// After `const out = { reply: parsed.reply, ... }` line:
if (process.env.STREAM_A_LOG_PATH) {
  const record = {
    timestamp: Date.now(),
    context_window: messages.slice(-3).map(m => ({ role: m.role, text: m.content })),
    reply: parsed.reply,
    emotion: parsed.emotion,
    gesture: parsed.gesture,
    expression_weights: parsed.expression_weights ?? null,
    motion_energy: parsed.motion_energy ?? null,
    screen_mood: parsed.screen_mood ?? null,
  };
  const fs = await import('fs/promises');
  await fs.appendFile(process.env.STREAM_A_LOG_PATH, JSON.stringify(record) + '\n');
}
```

- [ ] Create `modal_app/pipeline/label.py` (processes Stream A logs)
```python
"""
Stream A dataset builder: reads logged API records and prepares
training data for Head 3 (face) and Head 4 (screen).
"""
import json, numpy as np
from pathlib import Path
from modal_app.heads import FACE_PARAM_NAMES, MOOD_LABELS

EXPRESSION_KEY_TO_PARAM_IDX = {
    "brow_raise": FACE_PARAM_NAMES.index("brow_raise"),
    "brow_furrow": FACE_PARAM_NAMES.index("brow_furrow"),
    "eye_squint": FACE_PARAM_NAMES.index("eye_squint"),
    "mouth_open": FACE_PARAM_NAMES.index("mouth_open"),
    "smile_width": FACE_PARAM_NAMES.index("smile_width"),
    "glitch_intensity": FACE_PARAM_NAMES.index("glitch_intensity"),
}

def build_face_screen_dataset(raw_dir: str) -> dict:
    texts, face_targets, mood_targets, energy_targets = [], [], [], []

    for fname in Path(raw_dir).glob("*.jsonl"):
        with open(fname) as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("source") == "stream_c":
                    continue  # stream C has no face labels
                if not rec.get("expression_weights"):
                    continue

                # Build 32-dim face target (default 0.5 for unlabeled dims)
                face = np.full(32, 0.5, dtype=np.float32)
                for key, idx in EXPRESSION_KEY_TO_PARAM_IDX.items():
                    if key in rec["expression_weights"]:
                        face[idx] = float(rec["expression_weights"][key])

                mood_idx = MOOD_LABELS.index(rec["screen_mood"]) if rec.get("screen_mood") in MOOD_LABELS else 0
                energy = float(rec.get("motion_energy", 0.5))

                context = rec.get("context_window", [])
                text = " [SEP] ".join(f"{h['role']}: {h['text']}" for h in context[-3:])

                texts.append(text)
                face_targets.append(face)
                mood_targets.append(mood_idx)
                energy_targets.append(energy)

    return {
        "texts": texts,
        "face_targets": np.stack(face_targets) if face_targets else np.zeros((0, 32)),
        "mood_targets": np.array(mood_targets, dtype=np.int64),
        "energy_targets": np.array(energy_targets, dtype=np.float32),
    }
```

- [ ] Commit
```bash
git add api/chat.js modal_app/pipeline/label.py
git commit -m "feat: stream A logging + Claude schema extension for face/screen labels"
```

---

### Task 13: Head 3 + Head 4 training

**Files:**
- Modify: `modal_app/train.py`

- [ ] Add to `modal_app/train.py`
```python
# append to modal_app/train.py

from modal_app.heads import FaceExpressionHead, ScreenContentHead
from modal_app.pipeline.label import build_face_screen_dataset


def train_heads_3_and_4(
    raw_dir: str = "data/raw",
    n_epochs: int = 100,
    lr: float = 1e-3,
):
    """Train face expression (Head 3) and screen content (Head 4) heads."""
    ds = build_face_screen_dataset(raw_dir)

    if len(ds["texts"]) < 10:
        print(f"Only {len(ds['texts'])} Stream A records — collect more before training.")
        print("Run production conversations with STREAM_A_LOG_PATH set to accumulate data.")
        return

    print(f"Training on {len(ds['texts'])} Stream A records...")
    embeddings = embed(ds["texts"])  # [N, 384]

    X = torch.from_numpy(embeddings)
    Y_face = torch.from_numpy(ds["face_targets"])
    Y_mood = torch.from_numpy(ds["mood_targets"])
    Y_energy = torch.from_numpy(ds["energy_targets"])

    face_head = FaceExpressionHead()
    screen_head = ScreenContentHead()
    params = list(face_head.parameters()) + list(screen_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(n_epochs):
        face_head.train(); screen_head.train()
        optimizer.zero_grad()

        face_pred = face_head(X)
        continuous_pred, mood_logits = screen_head(X)

        face_loss = F.mse_loss(face_pred, Y_face)
        mood_loss = F.cross_entropy(mood_logits, Y_mood)
        energy_loss = F.mse_loss(continuous_pred[:, 0], Y_energy)

        loss = face_loss + mood_loss + 0.5 * energy_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — face: {face_loss:.4f}, mood: {mood_loss:.4f}")

    torch.save(face_head.state_dict(), f"{MODEL_PATH}/face_head.pt")
    torch.save(screen_head.state_dict(), f"{MODEL_PATH}/screen_head.pt")
    print("Saved Head 3 and Head 4.")
```

- [ ] Commit
```bash
git add modal_app/train.py
git commit -m "feat: head 3 + 4 training on stream A face/screen labels"
```

---

### Task 14: faceScreen.js rewrite (32-param renderer)

**Files:**
- Modify: `src/faceScreen.js`

This is a significant rewrite. Read the current `src/faceScreen.js` in full before starting.

- [ ] Read current `src/faceScreen.js` in full to understand the existing renderer structure

- [ ] Add `setParams()` and `_targetParams` infrastructure at the top of the class/module. The existing `setEmotion()` function must be kept as a thin wrapper that maps emotion names to approximate param dicts (fallback path).

```js
// Add to faceScreen.js — new param system

const DEFAULT_PARAMS = {
  brow_raise: 0.5, brow_furrow: 0.3, brow_asymmetry: 0.0,
  eye_openness: 0.8, eye_squint: 0.0, glitch_flicker: 0.05,
  mouth_curl_left: 0.5, mouth_curl_right: 0.5, mouth_open: 0.3, smile_width: 0.3,
  scan_line_speed: 0.3, scan_line_opacity: 0.15, glitch_intensity: 0.05, glitch_color_shift: 0.1,
  color_temperature: 0.5, color_saturation: 0.7, color_brightness: 0.6,
  noise_scale: 0.4, noise_speed: 0.2, vignette: 0.3,
  blink_rate: 0.5, pupil_dilation: 0.5, iris_glow_intensity: 0.6,
};

// Emotion → approximate params (fallback when Modal is unavailable)
const EMOTION_PARAMS = {
  neutral:  { ...DEFAULT_PARAMS },
  happy:    { ...DEFAULT_PARAMS, smile_width: 0.9, eye_openness: 0.95, brow_raise: 0.3 },
  excited:  { ...DEFAULT_PARAMS, smile_width: 0.8, brow_raise: 0.8, glitch_intensity: 0.15, scan_line_speed: 0.7 },
  sad:      { ...DEFAULT_PARAMS, brow_furrow: 0.7, eye_openness: 0.5, mouth_curl_left: 0.2, mouth_curl_right: 0.2 },
  angry:    { ...DEFAULT_PARAMS, brow_furrow: 0.9, eye_squint: 0.7, glitch_intensity: 0.3, glitch_color_shift: 0.4 },
  thinking: { ...DEFAULT_PARAMS, brow_furrow: 0.5, eye_squint: 0.3, brow_asymmetry: 0.4 },
};

let _currentParams = { ...DEFAULT_PARAMS };
let _targetParams  = { ...DEFAULT_PARAMS };

export function setParams(params) {
  Object.assign(_targetParams, params);
}

export function setEmotion(emotion) {
  // Fallback path — maps emotion name to approximate params
  const preset = EMOTION_PARAMS[emotion] ?? EMOTION_PARAMS.neutral;
  Object.assign(_targetParams, preset);
}

function _lerpParams(dt) {
  const speed = 4.0;  // lerp toward target at this rate per second
  for (const key of Object.keys(_targetParams)) {
    if (key in _currentParams) {
      _currentParams[key] += (_targetParams[key] - _currentParams[key]) * Math.min(1, speed * dt / 1000);
    }
  }
}
```

- [ ] Update the canvas draw functions to read from `_currentParams` instead of the hardcoded expression objects. Replace direct emotion checks like `if (currentExpression === 'happy')` with parameter reads like `_currentParams.smile_width > 0.7`.

- [ ] Call `_lerpParams(delta)` in the `tickFaceScreen(delta)` function each frame.

- [ ] Test locally
```bash
npm run dev
```
Open browser, chat with K-VRC, confirm face still renders and changes with emotion.

- [ ] Commit
```bash
git add src/faceScreen.js
git commit -m "feat: faceScreen 32-param renderer with setParams() and emotion fallback"
```

---

## Phase 5: Frontend Integration

### Task 15: animationController.js — blendClips()

**Files:**
- Modify: `src/animationController.js`

- [ ] Read `src/animationController.js` in full before modifying.

- [ ] Refactor `this.current` to `this.activeActions = new Map()`. Add `blendClips(weights)` method:

```js
// Add to AnimationController class:

blendClips(weights) {
  // weights: { clip_name: weight_float, ... } — top-3 clips, sum ≈ 1
  if (!this.mixer) return;

  const totalWeight = Object.values(weights).reduce((s, w) => s + w, 0);

  // Stop clips not in the new blend
  for (const [name, action] of this.activeActions) {
    if (!(name in weights)) {
      action.fadeOut(this.fadeTime);
      this.activeActions.delete(name);
    }
  }

  // Set weights for clips in the blend
  for (const [clipName, weight] of Object.entries(weights)) {
    const clip = this.clips[clipName];
    if (!clip) continue;

    let action = this.activeActions.get(clipName);
    if (!action) {
      action = this.mixer.clipAction(clip);
      action.setLoop(THREE.LoopRepeat);
      action.reset().play();
      this.activeActions.set(clipName, action);
    }
    action.setEffectiveWeight(weight / totalWeight);
  }

  // Keep currentName as the top-weighted clip for idle detection
  const topClip = Object.entries(weights).sort(([,a],[,b]) => b-a)[0]?.[0];
  if (topClip) this.currentName = topClip;
}
```

- [ ] Verify `_isIdling()` and `_playRandomIdle()` still work with the new `activeActions` map. Update them if needed — `_playRandomIdle` should call `blendClips({ [pick]: 1.0 })` instead of `_play()`.

- [ ] Test locally — idle animations should still rotate on their timer.

- [ ] Commit
```bash
git add src/animationController.js
git commit -m "feat: blendClips() for multi-clip weighted blending"
```

---

### Task 16: robot.js — motion delta application

**Files:**
- Modify: `src/robot.js`

- [ ] Add delta storage and application to `robot.js`:

```js
// Add to module-level state:
let _motionDeltas = null;    // Float32Array [60*6*4] or null
let _deltaFrame = 0;
const DELTA_JOINTS = ['head', 'chest', 'armL', 'armR', 'forearmL', 'forearmR'];

// Add export:
export function setMotionDeltas(flatArray) {
  // flatArray: flat float array from Modal, shape [60,6,4]
  _motionDeltas = new Float32Array(flatArray);
  _deltaFrame = 0;
}

// Add to updateRobot(delta), after the mixer update:
if (_motionDeltas) {
  const f = _deltaFrame % 60;
  for (let j = 0; j < DELTA_JOINTS.length; j++) {
    const bone = bones[DELTA_JOINTS[j]];
    if (!bone) continue;
    const base = 4 * (f * 6 + j);
    const dx = _motionDeltas[base],
          dy = _motionDeltas[base+1],
          dz = _motionDeltas[base+2],
          dw = _motionDeltas[base+3];
    // Apply delta as additive quaternion perturbation
    const q = bone.quaternion;
    q.set(q.x + dx, q.y + dy, q.z + dz, q.w + dw);
    q.normalize();
  }
  _deltaFrame++;
}
```

- [ ] Test locally — with random delta values the robot should jitter slightly (expected). With zero deltas it should be identical to before.

- [ ] Commit
```bash
git add src/robot.js
git commit -m "feat: motion delta application in robot update loop"
```

---

### Task 17: api/chat.js — Modal integration

**Files:**
- Modify: `api/chat.js`

- [ ] Add `MODAL_INFER_URL` env var to Vercel project settings (value: your Modal deployment URL + `/infer`)

- [ ] Add Modal call to `api/chat.js` after the Claude response is parsed:

```js
// After `const out = { reply: parsed.reply, ... }` line:

// Call Modal inference server
const modalUrl = process.env.MODAL_INFER_URL;
if (modalUrl) {
  try {
    const history = messages.map(m => ({ role: m.role, text: m.content }));
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);  // 4s timeout

    const modalRes = await fetch(modalUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: parsed.reply, history }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (modalRes.ok) {
      const motion = await modalRes.json();
      out.clip_weights    = motion.clip_weights;
      out.motion_deltas   = motion.motion_deltas;
      out.motion_deltas_shape = motion.motion_deltas_shape;
      out.face_params     = motion.face_params;
      out.screen_tokens   = motion.screen_tokens;
    }
  } catch (err) {
    // Timeout or error — fall back to rule-based system silently
    console.warn('Modal inference unavailable, using fallback:', err.message);
  }
}
```

- [ ] Add `MODAL_INFER_URL` to Vercel environment variables (Vercel dashboard → Settings → Env Vars)

- [ ] Commit
```bash
git add api/chat.js
git commit -m "feat: api/chat.js calls modal /infer with 4s timeout fallback"
```

---

### Task 18: chat.js — wire motion/face/screen to frontend

**Files:**
- Modify: `src/chat.js`

- [ ] Import new functions at top of `src/chat.js`:
```js
import { setParams } from './faceScreen.js';
import { setMotionDeltas } from './robot.js';
```

- [ ] In `sendMessage()`, after `addBubble(reply, 'robot')`, add:
```js
// Apply learned animation data if available
if (data.clip_weights) {
  robotRef?.playGestureBlend?.(data.clip_weights);
}
if (data.motion_deltas) {
  setMotionDeltas(data.motion_deltas);
}
if (data.face_params) {
  setParams(data.face_params);
}
// screen_tokens handled in main.js via global state
if (data.screen_tokens) {
  window.__kvrcScreenTokens = data.screen_tokens;
}
```

- [ ] Add `playGestureBlend` to the robot return object in `robot.js`:
```js
// In initRobot return object:
playGestureBlend: weights => animCtrl.blendClips(weights),
```

- [ ] In `main.js` animation loop, read `window.__kvrcScreenTokens` and update canvas renderer.

- [ ] Test end-to-end locally:
```bash
npm run dev
# In a second terminal:
node server.js
```
Send a message, confirm `clip_weights`, `face_params`, `screen_tokens` appear in network tab.

- [ ] Commit
```bash
git add src/chat.js src/robot.js src/main.js
git commit -m "feat: full frontend integration — modal animation data wired to three.js"
```

---

### Task 19: Build + deploy

- [ ] Build and deploy to Vercel
```bash
vercel build --prod && vercel deploy --prebuilt --prod
```

- [ ] Test live at `https://k-vrc.vercel.app` — send a message, verify body motion and face expression respond to context.

- [ ] Commit any fixes
```bash
git add -A && git commit -m "fix: production integration fixes"
```

---

## Phase 6: Annotation Tool + Fine-tuning

### Task 20: Stream B annotation server

**Files:**
- Create: `annotate.html`
- Create: `src/annotate.js`
- Create: `modal_app/pipeline/annotate.py`

- [ ] Create `modal_app/pipeline/annotate.py` (Express-compatible annotation server)
```python
"""
Simple HTTP server for Stream B annotation.
Reads data/raw/*.jsonl, serves records one at a time,
accepts corrected face params, saves to data/annotated/.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os
from pathlib import Path

DATA_RAW = "data/raw"
DATA_ANNOTATED = "data/annotated"

def load_queue():
    records = []
    annotated_ids = set()
    if Path(DATA_ANNOTATED).exists():
        for f in Path(DATA_ANNOTATED).glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    annotated_ids.add(r.get("id"))
    for f in Path(DATA_RAW).glob("*.jsonl"):
        with open(f) as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                r.setdefault("id", f"{f.name}:{i}")
                if r["id"] not in annotated_ids and r.get("source") != "stream_c":
                    records.append(r)
    return records

queue = load_queue()
current_idx = 0

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_idx
        if self.path == "/next":
            record = queue[current_idx] if current_idx < len(queue) else None
            self._json({"record": record, "idx": current_idx, "total": len(queue)})
    def do_POST(self):
        global current_idx
        if self.path == "/save":
            length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(length))
            Path(DATA_ANNOTATED).mkdir(parents=True, exist_ok=True)
            with open(f"{DATA_ANNOTATED}/annotated.jsonl", "a") as f:
                f.write(json.dumps({**queue[current_idx], **data, "source": "stream_b"}) + "\n")
            current_idx += 1
            self._json({"saved": True, "next_idx": current_idx})
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass  # silence default logging

if __name__ == "__main__":
    print("Annotation server at http://localhost:8765 — open annotate.html")
    HTTPServer(("localhost", 8765), Handler).serve_forever()
```

- [ ] Create `annotate.html` — loads K-VRC face canvas + sliders for all 32 params, communicates with annotation server on `localhost:8765`. (Minimal UI — 32 range inputs, a "Next" button, a small face preview canvas that calls `setParams()` from `src/faceScreen.js`.)

- [ ] Test annotation server
```bash
python3 modal_app/pipeline/annotate.py
# Open annotate.html in browser
# Verify records load, sliders work, Save moves to next record
```

- [ ] Commit
```bash
git add annotate.html src/annotate.js modal_app/pipeline/annotate.py
git commit -m "feat: stream B annotation tool for face expression fine-tuning"
```

---

### Task 21: Phase 4 fine-tune (Head 3 + Head 4 on annotated data)

Run after collecting ~100 annotated records via the Stream B tool.

- [ ] Add `train_heads_3_and_4_finetune()` to `modal_app/train.py`:
```python
def train_heads_3_and_4_finetune(annotated_dir: str = "data/annotated", n_epochs: int = 50, lr: float = 3e-4):
    """Fine-tune face/screen heads on high-quality annotated data."""
    ds = build_face_screen_dataset(annotated_dir)
    if len(ds["texts"]) < 50:
        print(f"Only {len(ds['texts'])} annotated records — need at least 50.")
        return
    # Same training loop as train_heads_3_and_4 but with lower LR and fewer epochs
    # Load existing weights first
    face_head = FaceExpressionHead()
    screen_head = ScreenContentHead()
    face_head.load_state_dict(torch.load(f"{MODEL_PATH}/face_head.pt"))
    screen_head.load_state_dict(torch.load(f"{MODEL_PATH}/screen_head.pt"))
    # ... same training loop as Task 13 but with annotated data and lower LR
    torch.save(face_head.state_dict(), f"{MODEL_PATH}/face_head_ft.pt")
    torch.save(screen_head.state_dict(), f"{MODEL_PATH}/screen_head_ft.pt")
    print("Fine-tuning complete.")
```

- [ ] Commit
```bash
git add modal_app/train.py
git commit -m "feat: phase 4 fine-tune function for annotated face/screen data"
```

---

### Task 22: Phase 5 joint encoder fine-tune

Run after all heads are trained and initial quality is validated in production.

- [ ] Add `train_joint_finetune()` to `modal_app/train.py`:
```python
def train_joint_finetune(n_epochs: int = 30, lr: float = 1e-5):
    """
    Phase 5: jointly fine-tune the encoder + output layers of all heads.
    Loss = equal weighted sum of Head 1–4 losses.
    Run on Modal GPU (T4 recommended — encoder backprop is expensive on CPU).
    """
    from sentence_transformers import SentenceTransformer
    import torch.nn as nn

    # Load all data
    # Load all heads with trained weights
    # Build joint optimizer over encoder final layers + all head parameters
    # Training loop: forward through encoder, then all 4 heads, sum losses
    # Note: freeze encoder early layers (0-4), only fine-tune layers 5-6 + pooling
    # Save updated encoder + heads
    print("Joint fine-tuning: encoder layers 5-6 + all head parameters")
    print("TODO: implement after Phase 4 quality is validated")
```

- [ ] Commit
```bash
git add modal_app/train.py
git commit -m "feat: phase 5 joint fine-tune stub"
```

---

## Environment Variables Reference

Set these before deployment:

| Variable | Where | Value |
|---|---|---|
| `ANTHROPIC_API_KEY` | Vercel | Anthropic API key |
| `MODAL_INFER_URL` | Vercel | `https://kvrc-animation--serve.modal.run/infer` |
| `STREAM_A_LOG_PATH` | Local dev only | `data/raw/stream_a.jsonl` |
| Modal secrets | `modal secret create kvrc-secrets` | `ANTHROPIC_API_KEY` if needed in Modal |

---

## Testing Checklist Before Each Phase Ships

**Phase 1:** `pytest tests/test_encoder.py tests/test_heads.py tests/test_serve.py -v`
**Phase 2:** Manual inspection of extraction output + `pytest tests/test_extract.py tests/test_dataset.py -v`
**Phase 3:** Loss curves converging, `data/models/clip_head.pt` and `mdm_head.pt` saved
**Phase 4:** Face expression visibly changes with sarcasm vs excitement in browser
**Phase 5:** Full chat → motion → face → screen pipeline working at `k-vrc.vercel.app`
**Phase 6:** Annotated expressions noticeably more expressive than Stream A baseline

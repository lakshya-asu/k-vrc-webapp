import modal
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import torch

from modal_app.app import app, volume, image, VOLUME_PATH
from modal_app.encoder import embed_context
from modal_app.heads import (
    ClipRetrievalHead, MDMRefinementHead,
    FaceExpressionHead, ScreenContentHead,
    MOOD_LABELS, FACE_PARAM_NAMES,
)

N_CLIPS = 178
N_JOINTS = 6
N_FRAMES = 60

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


def _get_clip_names() -> list[str]:
    path = f"{VOLUME_PATH}/models/clip_names.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return [f"clip_{i}" for i in range(N_CLIPS)]


def _load_heads():
    heads = {
        "clip": ClipRetrievalHead(n_clips=N_CLIPS),
        "mdm": MDMRefinementHead(n_joints=N_JOINTS, n_frames=N_FRAMES),
        "face": FaceExpressionHead(),
        "screen": ScreenContentHead(),
    }
    for name, head in heads.items():
        path = f"{VOLUME_PATH}/models/{name}_head.pt"
        if os.path.exists(path):
            head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        head.eval()
    return heads


# Loaded once at module import time inside the Modal container
_heads = None

def _get_heads():
    global _heads
    if _heads is None:
        _heads = _load_heads()
    return _heads


@web_app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    heads = _get_heads()
    history = req.history + [{"role": "assistant", "text": req.text}]
    ctx = embed_context(history)
    ctx_t = torch.from_numpy(ctx).unsqueeze(0)  # [1, 384]

    with torch.no_grad():
        clip_w = heads["clip"](ctx_t)[0].numpy()
        top3_idx = np.argsort(clip_w)[-3:][::-1]

        base_rot = torch.zeros(1, N_FRAMES, N_JOINTS, 4)
        base_rot[..., 3] = 1.0  # identity quaternion w=1
        deltas = heads["mdm"](ctx_t, base_rot)[0]  # [60, 6, 4]

        face = heads["face"](ctx_t)[0].numpy()

        continuous, mood_logits = heads["screen"](ctx_t)
        continuous = continuous[0].numpy()
        mood_idx = mood_logits[0].softmax(dim=0).argmax().item()

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


@app.function(
    volumes={VOLUME_PATH: volume},
    min_containers=1,
    image=image,
)
@modal.asgi_app()
def serve():
    return web_app

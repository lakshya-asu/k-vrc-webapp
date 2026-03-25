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

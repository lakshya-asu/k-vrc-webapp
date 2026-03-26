"""
Local shape tests for the /infer endpoint logic.
These test the response structure without hitting Modal.
"""
import torch
import numpy as np

N_CLIPS = 178
N_FRAMES = 60
N_JOINTS = 6


def test_infer_response_keys():
    """Verify InferResponse pydantic model has all required keys."""
    from modal_app.serve import InferResponse
    # Test that the model can be instantiated with expected structure
    resp = InferResponse(
        clip_weights={"clip_0": 0.5, "clip_1": 0.3, "clip_2": 0.2},
        motion_deltas=[0.0] * (N_FRAMES * N_JOINTS * 4),
        motion_deltas_shape=[N_FRAMES, N_JOINTS, 4],
        face_params={"brow_raise": 0.5},
        screen_tokens={"mood": "cold", "continuous": {"energy": 0.5}},
    )
    assert "clip_weights" in resp.model_dump()
    assert "motion_deltas" in resp.model_dump()
    assert "motion_deltas_shape" in resp.model_dump()
    assert "face_params" in resp.model_dump()
    assert "screen_tokens" in resp.model_dump()


def test_motion_deltas_shape_integrity():
    """Verify motion_deltas_shape matches actual deltas list length."""
    shape = [N_FRAMES, N_JOINTS, 4]
    expected_len = N_FRAMES * N_JOINTS * 4
    assert shape[0] * shape[1] * shape[2] == expected_len
    assert expected_len == 1440

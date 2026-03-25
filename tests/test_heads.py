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

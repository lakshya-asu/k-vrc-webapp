import numpy as np
import pytest


def test_sam2_fallback_mask_shape():
    """OpenCV fallback tracker returns one mask per frame with correct shape."""
    from modal_app.pipeline.extract import _track_opencv
    frames = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    bbox_frame0 = (10, 10, 50, 50)
    masks = _track_opencv(frames, bbox_frame0)
    assert len(masks) == 10
    assert masks[0].shape == (64, 64)
    assert masks[0].dtype == bool


def test_extract_bboxes_from_masks():
    """Bbox extraction from masks returns correct coordinates."""
    from modal_app.pipeline.extract import extract_bboxes_from_masks
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True
    bboxes = extract_bboxes_from_masks([mask])
    assert bboxes[0] == (30, 20, 69, 59)


def test_extract_bboxes_empty_mask():
    """Empty mask returns None."""
    from modal_app.pipeline.extract import extract_bboxes_from_masks
    mask = np.zeros((64, 64), dtype=bool)
    bboxes = extract_bboxes_from_masks([mask])
    assert bboxes[0] is None


def test_keypoints_heuristic_shape():
    """Heuristic keypoint fallback returns all 6 K-VRC joints with [x,y,conf]."""
    from modal_app.pipeline.extract import _keypoints_heuristic
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 50, 300, 400)
    kps = _keypoints_heuristic(frame, bbox)
    expected = ["head", "spine2", "left_arm", "right_arm", "left_forearm", "right_forearm"]
    for joint in expected:
        assert joint in kps
        assert len(kps[joint]) == 3


def test_coco_kps_to_kvrc():
    """COCO keypoint conversion maps correctly to K-VRC joints."""
    from modal_app.pipeline.extract import _coco_kps_to_kvrc
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = [100, 50, 0.9]   # nose → head
    kps[5] = [80, 130, 0.8]   # left_shoulder → left_arm + spine2
    kps[6] = [120, 130, 0.8]  # right_shoulder → right_arm + spine2
    result = _coco_kps_to_kvrc(kps)
    assert "head" in result
    assert result["head"][0] == pytest.approx(100.0)
    assert result["left_arm"][0] == pytest.approx(80.0)

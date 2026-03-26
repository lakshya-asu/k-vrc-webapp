import json
import os
import tempfile
import numpy as np


def _make_stream_c_record():
    return {
        "source": "stream_c",
        "text_window": "hello world",
        "keypoint_sequence": [
            {
                "head": [100.0, 50.0, 0.9],
                "spine2": [100.0, 150.0, 0.85],
                "left_arm": [80.0, 130.0, 0.8],
                "right_arm": [120.0, 130.0, 0.8],
                "left_forearm": [70.0, 160.0, 0.75],
                "right_forearm": [130.0, 160.0, 0.75],
            }
        ] * 60,
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
    assert ds["keypoints"].shape == (10, 60, 6, 2)
    assert "texts" in ds
    assert len(ds["texts"]) == 10


def test_keypoint_normalization():
    from modal_app.pipeline.dataset import normalize_keypoints
    kp_seq = np.random.rand(60, 6, 2) * 1920
    normed = normalize_keypoints(kp_seq, frame_w=1920, frame_h=1080)
    assert normed.max() <= 1.0 and normed.min() >= -1.0


def test_dataset_skips_short_sequences():
    from modal_app.pipeline.dataset import build_dataset
    with tempfile.TemporaryDirectory() as d:
        raw_path = os.path.join(d, "short.jsonl")
        short = _make_stream_c_record()
        short["keypoint_sequence"] = short["keypoint_sequence"][:30]  # only 30 frames
        with open(raw_path, "w") as f:
            f.write(json.dumps(short) + "\n")
        ds = build_dataset(raw_dir=d, processed_dir=d)
    assert ds["keypoints"].shape == (0, 60, 6, 2)


def test_dataset_skips_non_stream_c():
    from modal_app.pipeline.dataset import build_dataset
    with tempfile.TemporaryDirectory() as d:
        raw_path = os.path.join(d, "stream_a.jsonl")
        rec = _make_stream_c_record()
        rec["source"] = "stream_a"
        with open(raw_path, "w") as f:
            f.write(json.dumps(rec) + "\n")
        ds = build_dataset(raw_dir=d, processed_dir=d)
    assert len(ds["texts"]) == 0

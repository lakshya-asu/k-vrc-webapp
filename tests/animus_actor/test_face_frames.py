"""Face-frame generator tests: deterministic visor frames for a take.

The drawing itself runs under Node (the ported webapp renderer); these
tests build jobs from take data and hold the renderer to byte-level
determinism for a given job and seed.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor.embodiment import load_profile  # noqa: E402
from animus_actor.face_frames import (  # noqa: E402
    FACE_JOB_KIND,
    FaceFrameError,
    build_face_job,
    face_screen_config,
    render_face_frames,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KVRC_PROFILE = os.path.join(REPO, "src", "animus", "embodiment", "kvrc.profile.json")
TESTRIG_PROFILE = os.path.join(
    REPO, "src", "animus", "embodiment", "kvrc-testrig.profile.json"
)

PLAN = {
    "beats": [
        {
            "id": "greet-1",
            "at_ms": 0,
            "duration_ms": 1800,
            "face": {"expression": "warm_amused", "intensity": 0.5},
            "speech": {"text": "Hello", "delivery": "neutral"},
        }
    ]
}

VISEME_SAMPLES = [
    {"frame": 1, "at_ms": 0, "viseme": "X", "shape_key": "mouth_open", "weight": 0.0},
    {"frame": 6, "at_ms": 208, "viseme": "C", "shape_key": "mouth_open", "weight": 0.45},
    {"frame": 10, "at_ms": 375, "viseme": "X", "shape_key": "mouth_open", "weight": 0.0},
]


def node_available():
    return shutil.which("node") is not None or os.environ.get("ANIMUS_NODE")


class FaceJobTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile(KVRC_PROFILE)
        self.tmp = tempfile.mkdtemp(prefix="animus-face-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_artifact(self, frame_end=12):
        path = os.path.join(self.tmp, "take.animus.json")
        artifact = {
            "kind": "animus_viseme_take",
            "fps": 24,
            "frame_start": 1,
            "frame_end": frame_end,
            "samples": VISEME_SAMPLES,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle)
        return path

    def test_face_screen_config_only_for_screen_profiles(self):
        self.assertEqual(
            face_screen_config(self.profile),
            {"object": "screen", "emission_strength": 3.5},
        )
        self.assertIsNone(face_screen_config(load_profile(TESTRIG_PROFILE)))

    def test_build_face_job_collects_beats_visemes_and_frame_range(self):
        take_json = self.write_artifact(frame_end=12)
        layers = [
            {"request": {"params": {"frame_end": 27}}},
            {"request": {"params": {"frame_end": 43}}},
        ]
        job = build_face_job(
            PLAN, self.profile, [{"take_json": take_json}], layers
        )
        self.assertEqual(job["kind"], FACE_JOB_KIND)
        self.assertEqual(job["fps"], 24)
        self.assertEqual(job["frame_end"], 43)  # layers reach further
        self.assertEqual(
            job["face_beats"],
            [
                {
                    "expression": "warm_amused",
                    "intensity": 0.5,
                    "at_ms": 0,
                    "duration_ms": 1800,
                }
            ],
        )
        self.assertEqual(len(job["viseme_samples"]), len(VISEME_SAMPLES))
        self.assertEqual(job["seed"], self.profile["speech"]["seed"])

    def test_build_face_job_takes_frame_range_from_visemes_too(self):
        take_json = self.write_artifact(frame_end=60)
        job = build_face_job(PLAN, self.profile, [{"take_json": take_json}], [])
        self.assertEqual(job["frame_end"], 60)


@unittest.skipUnless(node_available(), "node is required for the face renderer")
class FaceRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="animus-face-render-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.job = {
            "kind": FACE_JOB_KIND,
            "fps": 24,
            "frame_end": 12,
            "face_beats": [
                {
                    "expression": "warm_amused",
                    "intensity": 0.5,
                    "at_ms": 0,
                    "duration_ms": 1800,
                }
            ],
            "viseme_samples": VISEME_SAMPLES,
            "seed": 7,
        }

    def render(self, sub, job=None):
        return render_face_frames(job or self.job, os.path.join(self.tmp, sub))

    def hashes(self, report):
        with open(report["manifest"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return [entry["sha256"] for entry in manifest["frames"]]

    def test_renderer_writes_every_frame_nonzero(self):
        report = self.render("a")
        self.assertEqual(report["frame_count"], 12)
        self.assertEqual(report["moods"], ["warm"])
        for index in range(1, 13):
            path = os.path.join(report["dir"], f"face_{index:04d}.png")
            self.assertTrue(os.path.exists(path), path)
            self.assertGreater(os.path.getsize(path), 0, path)

    def test_same_job_and_seed_render_identical_bytes(self):
        first = self.render("a")
        second = self.render("b")
        self.assertEqual(self.hashes(first), self.hashes(second))

    def test_seed_changes_the_frames(self):
        # 12 s of frames guarantees the seeded blink timer fires inside
        # the take, so two seeds must disagree somewhere.
        long_job = dict(self.job, frame_end=288)
        other = dict(long_job, seed=99)
        first = self.render("a", long_job)
        second = self.render("b", other)
        self.assertNotEqual(self.hashes(first), self.hashes(second))

    def test_manifest_covers_every_frame_with_hashes(self):
        report = self.render("a")
        with open(report["manifest"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["kind"], "animus_face_frames")
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual([e["frame"] for e in manifest["frames"]], list(range(1, 13)))
        for entry in manifest["frames"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(entry["mood"], "warm")

    def test_a_missing_node_binary_is_a_loud_error(self):
        from unittest import mock

        from animus_actor import face_frames as module

        env = {k: v for k, v in os.environ.items() if k != "ANIMUS_NODE"}
        with mock.patch.object(module.shutil, "which", return_value=None):
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(FaceFrameError):
                    module.find_node()


if __name__ == "__main__":
    unittest.main()

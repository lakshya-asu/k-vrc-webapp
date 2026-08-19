"""Converter and artifact tests. Pure, offline, no network, no binaries."""

import copy
import json
import os
import unittest

from animus_voice.converter import (
    ARTIFACT_KIND,
    BAD_CUE,
    BAD_CUE_FILE,
    BAD_FPS,
    BAD_JSON,
    UNKNOWN_VISEME,
    ConverterRefusal,
    cues_to_viseme_take,
    load_rhubarb_cues,
)
from animus_voice.visemes import (
    KVRC_SHAPE_KEYS,
    PROFILE,
    RHUBARB_SHAPES,
    VISEME_TO_SHAPES,
    full_weights,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "hello.rhubarb.json")


def _load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return json.load(handle)


class LoadCuesTest(unittest.TestCase):
    def test_loads_fixture_from_path(self):
        cues, duration = load_rhubarb_cues(os.path.abspath(FIXTURE))
        self.assertEqual(len(cues), 13)
        self.assertAlmostEqual(duration, 1.60)
        self.assertEqual(cues[0]["value"], "X")
        self.assertEqual(cues[1]["value"], "H")

    def test_loads_from_parsed_dict(self):
        cues, duration = load_rhubarb_cues(_load_fixture())
        self.assertEqual(len(cues), 13)
        self.assertAlmostEqual(duration, 1.60)

    def test_loads_from_json_string(self):
        text = json.dumps(_load_fixture())
        cues, _duration = load_rhubarb_cues(text)
        self.assertEqual(len(cues), 13)

    def test_duration_absent_is_none(self):
        data = _load_fixture()
        del data["metadata"]["duration"]
        _cues, duration = load_rhubarb_cues(data)
        self.assertIsNone(duration)


class MalformedInputTest(unittest.TestCase):
    def test_bad_json_string(self):
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues("{not valid json")
        self.assertEqual(ctx.exception.code, BAD_JSON)

    def test_missing_mouthcues(self):
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues({"metadata": {}})
        self.assertEqual(ctx.exception.code, BAD_CUE_FILE)

    def test_empty_mouthcues(self):
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues({"mouthCues": []})
        self.assertEqual(ctx.exception.code, BAD_CUE_FILE)

    def test_unknown_viseme(self):
        data = {"mouthCues": [{"start": 0.0, "end": 0.1, "value": "Z"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, UNKNOWN_VISEME)

    def test_cue_missing_value(self):
        data = {"mouthCues": [{"start": 0.0, "end": 0.1}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, UNKNOWN_VISEME)

    def test_cue_non_numeric_start(self):
        data = {"mouthCues": [{"start": "soon", "end": 0.1, "value": "X"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)

    def test_cue_end_before_start(self):
        data = {"mouthCues": [{"start": 0.5, "end": 0.1, "value": "X"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)

    def test_cue_negative_time(self):
        data = {"mouthCues": [{"start": -0.1, "end": 0.1, "value": "X"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)

    def test_cues_out_of_order(self):
        data = {
            "mouthCues": [
                {"start": 0.3, "end": 0.4, "value": "X"},
                {"start": 0.1, "end": 0.2, "value": "B"},
            ]
        }
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)

    def test_nan_start_rejected(self):
        data = {"mouthCues": [{"start": float("nan"), "end": 0.1, "value": "X"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)

    def test_bool_start_rejected(self):
        # True is an int subclass; it must not pass as a time.
        data = {"mouthCues": [{"start": True, "end": 0.1, "value": "X"}]}
        with self.assertRaises(ConverterRefusal) as ctx:
            load_rhubarb_cues(data)
        self.assertEqual(ctx.exception.code, BAD_CUE)


class ArtifactShapeTest(unittest.TestCase):
    def setUp(self):
        self.cues, self.duration = load_rhubarb_cues(_load_fixture())
        self.take = cues_to_viseme_take(self.cues, duration=self.duration)

    def test_top_level_fields(self):
        take = self.take
        self.assertEqual(take["schema_version"], "0.1")
        self.assertEqual(take["kind"], ARTIFACT_KIND)
        self.assertEqual(take["object"], "KVRC")
        self.assertEqual(take["name_hint"], "animus_speech")
        self.assertEqual(take["fps"], 24)
        self.assertEqual(take["frame_start"], 1)
        self.assertEqual(take["cue_count"], 13)

    def test_embodiment_block(self):
        emb = self.take["embodiment"]
        self.assertEqual(emb["profile"], PROFILE)
        self.assertEqual(emb["target"], "shape_keys")
        self.assertEqual(emb["shape_keys"], list(KVRC_SHAPE_KEYS))

    def test_every_sample_matches_bridge_vocabulary(self):
        # Analog of the bridge pose sample: one target channel, one scalar,
        # one integer frame within the declared range.
        fs = self.take["frame_start"]
        fe = self.take["frame_end"]
        for sample in self.take["samples"]:
            self.assertEqual(
                set(sample), {"frame", "at_ms", "viseme", "shape_key", "weight"}
            )
            self.assertIsInstance(sample["frame"], int)
            self.assertFalse(isinstance(sample["frame"], bool))
            self.assertGreaterEqual(sample["frame"], fs)
            self.assertLessEqual(sample["frame"], fe)
            self.assertIn(sample["shape_key"], KVRC_SHAPE_KEYS)
            self.assertIsInstance(sample["weight"], float)
            self.assertGreaterEqual(sample["weight"], 0.0)
            self.assertLessEqual(sample["weight"], 1.0)
            self.assertIn(sample["viseme"], RHUBARB_SHAPES)

    def test_every_frame_writes_full_shape_key_set(self):
        # One cue produces exactly len(KVRC_SHAPE_KEYS) samples so the mouth
        # is fully specified and never inherits a stale weight.
        self.assertEqual(
            len(self.take["samples"]), len(self.cues) * len(KVRC_SHAPE_KEYS)
        )
        by_frame = {}
        for sample in self.take["samples"]:
            by_frame.setdefault(sample["frame"], set()).add(sample["shape_key"])
        for keys in by_frame.values():
            self.assertEqual(keys, set(KVRC_SHAPE_KEYS))

    def test_first_cue_maps_to_idle_shut_mouth(self):
        first = self.take["cues"][0]
        self.assertEqual(first["viseme"], "X")
        self.assertEqual(first["weights"]["mouth_open"], 0.0)

    def test_frame_math(self):
        # Cue at 0.34s at 24 fps lands on frame 1 + round(0.34*24) = 1 + 8.
        target = next(c for c in self.take["cues"] if c["at_ms"] == 340)
        self.assertEqual(target["frame"], 1 + 8)

    def test_source_provenance_optional_and_attached(self):
        src = {"text": "Hello, I am K-VRC.", "voice": "af_heart"}
        take = cues_to_viseme_take(self.cues, duration=self.duration, source=src)
        self.assertEqual(take["source"], src)
        self.assertNotIn("source", self.take)


class DeterminismTest(unittest.TestCase):
    def test_identical_inputs_yield_identical_bytes(self):
        cues, duration = load_rhubarb_cues(_load_fixture())
        a = cues_to_viseme_take(cues, duration=duration)
        b = cues_to_viseme_take(copy.deepcopy(cues), duration=duration)
        self.assertEqual(
            json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
        )

    def test_fps_changes_frames(self):
        cues, duration = load_rhubarb_cues(_load_fixture())
        at24 = cues_to_viseme_take(cues, duration=duration, fps=24)
        at30 = cues_to_viseme_take(cues, duration=duration, fps=30)
        self.assertNotEqual(at24["frame_end"], at30["frame_end"])


class ConverterParamGuardTest(unittest.TestCase):
    def setUp(self):
        self.cues, self.duration = load_rhubarb_cues(_load_fixture())

    def test_bad_fps_rejected(self):
        with self.assertRaises(ConverterRefusal) as ctx:
            cues_to_viseme_take(self.cues, fps=0)
        self.assertEqual(ctx.exception.code, BAD_FPS)

    def test_bool_fps_rejected(self):
        with self.assertRaises(ConverterRefusal) as ctx:
            cues_to_viseme_take(self.cues, fps=True)
        self.assertEqual(ctx.exception.code, BAD_FPS)

    def test_empty_cues_rejected(self):
        with self.assertRaises(ConverterRefusal):
            cues_to_viseme_take([])


class MappingTableTest(unittest.TestCase):
    def test_every_rhubarb_shape_has_a_mapping(self):
        for shape in RHUBARB_SHAPES:
            self.assertIn(shape, VISEME_TO_SHAPES)

    def test_full_weights_covers_all_shape_keys(self):
        for shape in VISEME_TO_SHAPES:
            weights = full_weights(shape)
            self.assertEqual(set(weights), set(KVRC_SHAPE_KEYS))
            for value in weights.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

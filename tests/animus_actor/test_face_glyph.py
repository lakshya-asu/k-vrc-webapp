"""Face v2 units: the face_glyph beat channel, the glyph face job,
the reusable stage look resolution, and the wav-reuse voice mode.

The glyph drawing itself is covered byte-for-byte by the JS suite
(tests_js/animus-glyph.test.js); here the Python side of the contract
and pipeline is held to the same vocabulary and rules.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor.contract import (  # noqa: E402
    FACE_GLYPH_BROWS,
    FACE_GLYPH_EYES,
    FACE_GLYPH_MOUTHS,
    validate_actor_plan,
)
from animus_actor.face_frames import build_face_job  # noqa: E402
from animus_actor.stage_look import (  # noqa: E402
    DEFAULT_LOOK,
    clamped_energy,
    resolve_look,
)
from animus_actor.voice import run_voice_job  # noqa: E402

AUTHORITY = {"actor_id": "kvrc", "control_level": "suggest"}


def plan_with(beat_extra):
    beat = {"id": "b-1", "at_ms": 0, "duration_ms": 1200}
    beat.update(beat_extra)
    return {
        "schema_version": "0.1",
        "summary": "glyph channel check",
        "beats": [beat],
    }


class FaceGlyphContractTest(unittest.TestCase):
    def test_composed_glyph_beat_validates(self):
        checked = validate_actor_plan(
            plan_with(
                {
                    "face_glyph": {
                        "eyes": "bar",
                        "brows": "angry_in",
                        "mouth": "gritted",
                        "mood": "angry",
                    }
                }
            ),
            AUTHORITY,
        )
        self.assertTrue(checked["ok"], checked["errors"])

    def test_text_glyph_beat_validates_and_counts_as_a_channel(self):
        checked = validate_actor_plan(
            plan_with({"face_glyph": {"text": "SCAN", "mood": "data"}}),
            AUTHORITY,
        )
        self.assertTrue(checked["ok"], checked["errors"])

    def test_vocabulary_matches_the_js_composer(self):
        # The JS module is the reference; a drifted vocabulary would let
        # one side validate what the other refuses.
        repo = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        source_path = os.path.join(
            repo, "src", "animus", "face", "glyphComposer.js"
        )
        with open(source_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for name in FACE_GLYPH_EYES + FACE_GLYPH_BROWS + FACE_GLYPH_MOUTHS:
            self.assertIn(f"'{name}'", source, name)

    def test_rejections(self):
        bad_cases = [
            {"face_glyph": {"eyes": "laser"}},
            {"face_glyph": {"eyes": "round", "brows": "wiggly"}},
            {"face_glyph": {"eyes": "round", "mood": "salsa"}},
            {"face_glyph": {"text": "WAYTOOLONG"}},
            {"face_glyph": {"text": "ok;drop table"}},
            {"face_glyph": {"text": "OK", "eyes": "round"}},
            {"face_glyph": {"eyes": "round", "keyframes": []}},
            {"face_glyph": {"eyes": "round", "intensity": 2}},
            {
                "face": {"expression": "neutral_idle"},
                "face_glyph": {"eyes": "round"},
            },
        ]
        for extra in bad_cases:
            checked = validate_actor_plan(plan_with(extra), AUTHORITY)
            self.assertFalse(checked["ok"], extra)

    def test_glyph_beats_land_in_the_face_job(self):
        plan = {
            "beats": [
                {
                    "id": "b-1",
                    "at_ms": 0,
                    "duration_ms": 1000,
                    "face": {"expression": "warm_smile", "intensity": 1},
                },
                {
                    "id": "b-2",
                    "at_ms": 1000,
                    "duration_ms": 1000,
                    "face_glyph": {"text": "SCAN", "mood": "data"},
                },
            ]
        }
        profile = {"fps": 24, "speech": {"seed": 3}}
        job = build_face_job(plan, profile, [], [])
        self.assertEqual(len(job["face_beats"]), 2)
        self.assertEqual(job["face_beats"][0]["expression"], "warm_smile")
        self.assertEqual(job["face_beats"][1]["glyph"]["text"], "SCAN")
        self.assertNotIn("expression", job["face_beats"][1])


class StageLookTest(unittest.TestCase):
    def test_no_look_resolves_to_none(self):
        self.assertIsNone(resolve_look({}))
        self.assertIsNone(resolve_look({"look": False}))
        self.assertIsNone(resolve_look(None))

    def test_true_and_preset_name_resolve_to_defaults(self):
        self.assertEqual(resolve_look({"look": True}), DEFAULT_LOOK)
        self.assertEqual(resolve_look({"look": "studio_v2"}), DEFAULT_LOOK)

    def test_overrides_merge_and_unknowns_fail_loudly(self):
        look = resolve_look({"look": {"bloom_strength": 0.5, "stripes": False}})
        self.assertEqual(look["bloom_strength"], 0.5)
        self.assertFalse(look["stripes"])
        self.assertEqual(look["preset"], "studio_v2")
        with self.assertRaises(ValueError):
            resolve_look({"look": "film_noir"})
        with self.assertRaises(ValueError):
            resolve_look({"look": {"blom_strength": 0.5}})

    def test_wash_guard_ships_in_the_preset(self):
        # Scene-5 lesson: key 1150 / fill 340 washed the shell white.
        look = resolve_look({"look": True})
        self.assertEqual(look["key_energy_max"], 1000.0)
        self.assertEqual(look["fill_energy_max"], 300.0)

    def test_clamped_energy_caps_hot_scenes_and_passes_sane_ones(self):
        # The exact scene-5 request is pulled back under the ceiling.
        self.assertEqual(clamped_energy(1150.0, 1000.0), 1000.0)
        self.assertEqual(clamped_energy(340.0, 300.0), 300.0)
        # Requests inside the ceiling pass through untouched.
        self.assertEqual(clamped_energy(950.0, 1000.0), 950.0)
        # A scene may disable the clamp knowingly.
        self.assertEqual(clamped_energy(1150.0, None), 1150.0)
        self.assertEqual(clamped_energy(1150.0, 0), 1150.0)


class VoiceReuseTest(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="animus-reuse-")
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)
        self.job = {
            "beat_id": "b-1",
            "stem": "b-1_speech",
            "text": "Hello",
        }

    def _write_artifacts(self):
        artifact = {
            "cue_count": 3,
            "duration_ms": 1500,
            "samples": [
                {"frame": 1, "shape_key": "mouth_open", "weight": 0.0},
                {"frame": 5, "shape_key": "mouth_open", "weight": 0.6},
            ],
        }
        take = os.path.join(self.out_dir, "b-1_speech.animus.json")
        with open(take, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle)
        wav = os.path.join(self.out_dir, "b-1_speech.wav")
        with open(wav, "wb") as handle:
            handle.write(b"RIFFfake")
        return take, wav

    def test_reuse_returns_the_existing_take(self):
        take, wav = self._write_artifacts()
        receipt = run_voice_job(self.job, self.out_dir, "reuse")
        self.assertEqual(receipt["voice_mode"], "reuse")
        self.assertEqual(receipt["wav"], wav)
        self.assertEqual(receipt["take_json"], take)
        self.assertEqual(receipt["sample_count"], 2)
        self.assertEqual(receipt["duration_ms"], 1500)

    def test_missing_artifacts_fail_loudly(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_voice_job(self.job, self.out_dir, "reuse")
        self.assertIn("reuse", str(ctx.exception))

    def test_unknown_mode_still_refused(self):
        with self.assertRaises(ValueError):
            run_voice_job(self.job, self.out_dir, "resynth")


if __name__ == "__main__":
    unittest.main()

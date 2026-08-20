"""Actor loop tests against fake bpy: the whole chain minus real Blender.

The loop's sender hook routes every mapped request through the real
bridge validators and operations (fake scene), so these tests prove the
loop logic end to end: line -> plan -> contract -> mapping -> visemes ->
bridge ops -> receipts. Real Blender is covered by
blender/animus_bridge/acceptance/run_actor_acceptance.py.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402
import fake_bpy  # noqa: E402

from animus_actor import loop  # noqa: E402
from animus_bridge import operations, protocol  # noqa: E402

LINE = "Hello, I am K-VRC"


def fake_sender(request):
    request_id, op, raw = protocol.validate_envelope(request)
    try:
        params = protocol.validate_params(op, raw)
        result = operations.HANDLERS[op](params)
        return protocol.success(request_id, op, result)
    except protocol.Refusal as refusal:
        return protocol.failure(request_id, op, refusal)


def build_fake_scene(profile_keys=True):
    fake_bpy.reset()
    profile = loop.load_profile(loop.DEFAULT_PROFILE)
    rig = profile["rig"]
    fake_bpy.add_armature_object(rig["object"], rig["bones"])
    keys = ["Basis"] + (rig["shape_keys"] if profile_keys else [])
    fake_bpy.add_shape_keyed_object(rig["face_object"], keys)
    return profile


class ActorLoopTests(unittest.TestCase):
    def run_loop(self, **overrides):
        options = {
            "voice_mode": "convert",
            "sender": fake_sender,
            "control_level": "perform",
        }
        options.update(overrides)
        with tempfile.TemporaryDirectory() as out_dir:
            options.setdefault("out_dir", out_dir)
            options.setdefault(
                "receipt_path", os.path.join(out_dir, "receipt.json")
            )
            receipt = loop.run_actor_loop(LINE, **options)
            receipt_file = options["receipt_path"]
            written = None
            if os.path.exists(receipt_file):
                with open(receipt_file, "r", encoding="utf-8") as handle:
                    written = json.load(handle)
        return receipt, written

    def test_perform_lands_every_layer(self):
        build_fake_scene()
        receipt, _ = self.run_loop()
        self.assertTrue(receipt["performed"])
        channels = [layer["channel"] for layer in receipt["layers"]]
        self.assertEqual(channels, ["body", "gaze", "face", "speech"])
        for layer in receipt["layers"]:
            self.assertIs(layer["response"]["ok"], True, layer["response"])

        # One Action per layer, nothing overwritten, right datablocks.
        actions = [
            layer["response"]["result"]["action"] for layer in receipt["layers"]
        ]
        self.assertEqual(len(actions), len(set(actions)))
        self.assertEqual(len(fake_bpy.data.actions), 4)
        kvrc = fake_bpy.data.objects.get("KVRC")
        face_key = fake_bpy.data.objects.get("KVRC_face").data.shape_keys
        self.assertEqual(len(kvrc.animation_data.nla_tracks), 2)
        self.assertEqual(len(face_key.animation_data.nla_tracks), 2)
        self.assertTrue(loop.loop_succeeded(receipt))

    def test_summary_reports_the_take(self):
        build_fake_scene()
        receipt, _ = self.run_loop()
        summary = loop.summarize(receipt)
        self.assertEqual(summary["operator"], "deterministic")
        self.assertTrue(summary["fallback"])
        self.assertEqual(summary["beats"], 1)
        self.assertEqual(summary["beats_executed"], 1)
        self.assertGreater(summary["viseme_count"], 0)
        self.assertTrue(summary["voice_track"].endswith(".animus.json"))
        self.assertEqual(len(summary["layers"]), 4)
        for layer in summary["layers"]:
            self.assertIs(layer["ok"], True)
            self.assertTrue(layer["action"])

    def test_receipt_file_written_and_complete(self):
        build_fake_scene()
        receipt, written = self.run_loop()
        self.assertIsNotNone(written)
        self.assertEqual(written["kind"], "animus_actor_receipt")
        self.assertEqual(written["line"], LINE)
        self.assertEqual(
            written["plan"]["provenance"],
            {"operator": "deterministic", "model": None, "fallback": True},
        )
        self.assertEqual(written["profile"]["name"], "kvrc-testrig")
        self.assertEqual(len(written["voice"]), 1)
        self.assertGreater(written["voice"][0]["sample_count"], 0)
        self.assertNotIn("artifact", written["voice"][0])

    def test_suggest_maps_but_performs_nothing(self):
        build_fake_scene()
        receipt, _ = self.run_loop(control_level="suggest", sender=None)
        self.assertFalse(receipt["performed"])
        self.assertEqual(len(fake_bpy.data.actions), 0)
        self.assertIsNone(receipt["bridge"])
        for layer in receipt["layers"]:
            self.assertNotIn("response", layer)
        summary = loop.summarize(receipt)
        self.assertEqual(summary["beats_executed"], 0)
        self.assertTrue(loop.loop_succeeded(receipt))

    def test_voice_skip_omits_the_speech_layer(self):
        build_fake_scene()
        receipt, _ = self.run_loop(voice_mode="skip")
        channels = [layer["channel"] for layer in receipt["layers"]]
        self.assertEqual(channels, ["body", "gaze", "face"])
        self.assertEqual(receipt["voice"], [])
        self.assertTrue(loop.loop_succeeded(receipt))

    def test_scene_refusal_lands_in_receipts_not_exceptions(self):
        build_fake_scene(profile_keys=False)  # face mesh lacks the visemes
        receipt, _ = self.run_loop()
        speech = receipt["layers"][3]
        self.assertIs(speech["response"]["ok"], False)
        self.assertEqual(speech["response"]["error"]["code"], "unknown_shape_key")
        self.assertFalse(loop.loop_succeeded(receipt))

    def test_bad_control_level_is_refused_before_any_work(self):
        build_fake_scene()
        with self.assertRaises(loop.ActorLoopError):
            self.run_loop(control_level="autopilot")


if __name__ == "__main__":
    unittest.main()

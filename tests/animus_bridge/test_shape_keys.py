"""apply_shape_keys: validation, refusals, and acceptance invariants.

Mirrors the pose-key coverage: unknown object, unknown shape key,
non-finite weight, and out-of-range frame all refuse mutating nothing;
success returns a receipt; every take gets a fresh Action and NLA strip
on the shape-key Key datablock; repeats never overwrite.

Also proves the voice pipeline's viseme take artifact maps onto the
request payload directly: the artifact built by animus_voice.converter
from the committed Rhubarb fixture drives a take without rewriting its
samples list.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import fake_bpy  # noqa: E402
from animus_bridge import operations, protocol  # noqa: E402
from animus_bridge.executor import Executor, WorkItem  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VOICE_DIR = os.path.join(_REPO, "voice")
if _VOICE_DIR not in sys.path:
    sys.path.insert(0, _VOICE_DIR)

from animus_voice import converter  # noqa: E402
from animus_voice.visemes import KVRC_SHAPE_KEYS  # noqa: E402

SHAPE_KEYS = list(KVRC_SHAPE_KEYS)

RHUBARB_FIXTURE = os.path.join(
    _VOICE_DIR, "animus_voice", "fixtures", "hello.rhubarb.json"
)


def speech_params():
    return {
        "object": "KVRC_face",
        "name_hint": "speech",
        "frame_start": 1,
        "frame_end": 48,
        "samples": [
            {"shape_key": "mouth_open", "frame": 1, "weight": 0.0},
            {"shape_key": "mouth_open", "frame": 12, "weight": 0.8},
            {"shape_key": "smile_width", "frame": 12, "weight": 0.4},
            {"shape_key": "mouth_open", "frame": 48, "weight": 0.0},
        ],
    }


class ProtocolTests(unittest.TestCase):
    def refusal(self, params):
        with self.assertRaises(protocol.Refusal) as ctx:
            protocol.validate_params("apply_shape_keys", params)
        return ctx.exception

    def test_valid_params(self):
        params = protocol.validate_params("apply_shape_keys", speech_params())
        self.assertEqual(params["name_hint"], "speech")
        self.assertEqual(len(params["samples"]), 4)
        self.assertEqual(params["samples"][1]["weight"], 0.8)

    def test_voice_artifact_extras_accepted_and_dropped(self):
        params = speech_params()
        params["samples"][0]["at_ms"] = 0
        params["samples"][0]["viseme"] = "X"
        clean = protocol.validate_params("apply_shape_keys", params)
        self.assertEqual(
            clean["samples"][0], {"shape_key": "mouth_open", "frame": 1, "weight": 0.0}
        )

    def test_unknown_sample_key_refused(self):
        params = speech_params()
        params["samples"][0]["bone"] = "arm.R"
        self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)

    def test_missing_shape_key_refused(self):
        params = speech_params()
        del params["samples"][2]["shape_key"]
        self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)

    def test_non_finite_weight_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            params = speech_params()
            params["samples"][1]["weight"] = bad
            self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)

    def test_boolean_weight_refused(self):
        params = speech_params()
        params["samples"][1]["weight"] = True
        self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)

    def test_frame_outside_declared_range_refused(self):
        params = speech_params()
        params["samples"][3]["frame"] = 480
        self.assertEqual(self.refusal(params).code, protocol.FRAME_OUT_OF_RANGE)

    def test_frame_not_integer_refused(self):
        params = speech_params()
        params["samples"][0]["frame"] = 1.5
        self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)

    def test_empty_samples_refused(self):
        params = speech_params()
        params["samples"] = []
        self.assertEqual(self.refusal(params).code, protocol.INVALID_PARAMS)


class OperationsBase(unittest.TestCase):
    def setUp(self):
        fake_bpy.reset()
        fake_bpy.add_shape_keyed_object("KVRC_face", ["Basis"] + SHAPE_KEYS)

    def key_datablock(self):
        return fake_bpy.data.objects.get("KVRC_face").data.shape_keys

    def scene_counts(self):
        key = self.key_datablock()
        strips = []
        if key.animation_data is not None:
            strips = [s for t in key.animation_data.nla_tracks for s in t.strips]
        return len(fake_bpy.data.actions), strips

    def refusal(self, params):
        with self.assertRaises(protocol.Refusal) as ctx:
            operations.apply_shape_keys(params)
        return ctx.exception


class ApplyShapeKeysTests(OperationsBase):
    def test_receipt_names_action_track_and_strip(self):
        receipt = operations.apply_shape_keys(speech_params())
        self.assertEqual(receipt["action"], "ANIMUS_speech_take001")
        self.assertEqual(receipt["strip"], "ANIMUS_speech_take001")
        self.assertEqual(receipt["track"], "ANIMUS_speech_take001_track")
        self.assertEqual(receipt["shape_keys"], ["mouth_open", "smile_width"])
        self.assertEqual(receipt["sample_count"], 4)
        self.assertEqual(receipt["key_count"], 4)

    def test_keys_land_in_key_block_fcurves(self):
        operations.apply_shape_keys(speech_params())
        action = fake_bpy.data.actions.get("ANIMUS_speech_take001")
        snapshot = action.key_snapshot()
        self.assertEqual(
            snapshot[('key_blocks["mouth_open"].value', 0)],
            [(1, 0.0), (12, 0.8), (48, 0.0)],
        )
        self.assertEqual(snapshot[('key_blocks["smile_width"].value', 0)], [(12, 0.4)])
        self.assertTrue(action.get(operations.MARKER))

    def test_take_lands_on_the_key_datablock(self):
        operations.apply_shape_keys(speech_params())
        action_count, strips = self.scene_counts()
        self.assertEqual(action_count, 1)
        self.assertEqual(len(strips), 1)
        obj = fake_bpy.data.objects.get("KVRC_face")
        self.assertIsNone(obj.animation_data)

    def test_repeat_creates_a_separate_take(self):
        first = operations.apply_shape_keys(speech_params())
        first_action = fake_bpy.data.actions.get(first["action"])
        first_keys = first_action.key_snapshot()

        second = operations.apply_shape_keys(speech_params())

        self.assertNotEqual(first["action"], second["action"])
        self.assertNotEqual(first["strip"], second["strip"])
        self.assertNotEqual(first["track"], second["track"])
        action_count, strips = self.scene_counts()
        self.assertEqual(action_count, 2)
        self.assertEqual(len(strips), 2)
        self.assertEqual(len({s.name for s in strips}), 2)
        self.assertEqual(first_action.key_snapshot(), first_keys)

    def test_unknown_object_refused_without_mutation(self):
        params = speech_params()
        params["object"] = "Ghost"
        error = self.refusal(params)
        self.assertEqual(error.code, protocol.UNKNOWN_OBJECT)
        self.assertEqual(self.scene_counts(), (0, []))

    def test_object_without_shape_keys_refused(self):
        fake_bpy.add_mesh_without_shape_keys("Prop")
        params = speech_params()
        params["object"] = "Prop"
        error = self.refusal(params)
        self.assertEqual(error.code, protocol.NO_SHAPE_KEYS)
        self.assertEqual(self.scene_counts(), (0, []))

    def test_object_with_no_data_refused(self):
        fake_bpy.add_plain_object("Empty")
        params = speech_params()
        params["object"] = "Empty"
        error = self.refusal(params)
        self.assertEqual(error.code, protocol.NO_SHAPE_KEYS)
        self.assertEqual(self.scene_counts(), (0, []))

    def test_unknown_shape_key_mid_payload_leaves_no_partial_artifact(self):
        params = speech_params()
        params["samples"][2]["shape_key"] = "eyebrow_raise"
        error = self.refusal(params)
        self.assertEqual(error.code, protocol.UNKNOWN_SHAPE_KEY)
        self.assertEqual(self.scene_counts(), (0, []))

    def test_user_actions_untouched(self):
        user_action = fake_bpy.data.actions.new("UserBlink")
        user_snapshot = user_action.key_snapshot()
        operations.apply_shape_keys(speech_params())
        self.assertEqual(user_action.key_snapshot(), user_snapshot)
        self.assertFalse(user_action.get(operations.MARKER))


class ExecutorAcceptanceTests(OperationsBase):
    """Same invariants as the pose-key acceptance, through the executor."""

    def setUp(self):
        super().setUp()
        self.executor = Executor()
        self.executor.install()
        self.responses = []

    def run_request(self, params):
        validated = protocol.validate_params("apply_shape_keys", params)
        self.executor.submit(
            WorkItem(
                {"id": "face", "op": "apply_shape_keys", "params": validated},
                self.responses.append,
            )
        )
        fake_bpy.app.timers.pump()
        return self.responses[-1]

    def test_take_creates_exactly_one_action_and_one_strip(self):
        response = self.run_request(copy.deepcopy(speech_params()))
        self.assertTrue(response["ok"])
        action_count, strips = self.scene_counts()
        self.assertEqual(action_count, 1)
        self.assertEqual(len(strips), 1)
        self.assertEqual(strips[0].action.name, response["result"]["action"])

    def test_refusal_through_executor_leaves_counts_unchanged(self):
        params = speech_params()
        params["samples"][0]["shape_key"] = "tail_wag"
        response = self.run_request(params)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.UNKNOWN_SHAPE_KEY)
        self.assertEqual(self.scene_counts(), (0, []))


class VoiceArtifactTests(OperationsBase):
    """The real converter's artifact drives a take without rewriting."""

    def build_artifact(self):
        cues, duration = converter.load_rhubarb_cues(RHUBARB_FIXTURE)
        return converter.cues_to_viseme_take(
            cues, duration=duration, obj="KVRC_face", name_hint="animus_speech"
        )

    def test_artifact_samples_pass_validation_as_is(self):
        artifact = self.build_artifact()
        params = protocol.validate_params(
            "apply_shape_keys",
            {
                "object": artifact["object"],
                "name_hint": artifact["name_hint"],
                "frame_start": artifact["frame_start"],
                "frame_end": artifact["frame_end"],
                "samples": artifact["samples"],
            },
        )
        self.assertEqual(len(params["samples"]), len(artifact["samples"]))

    def test_artifact_take_applies_with_one_action_and_strip(self):
        artifact = self.build_artifact()
        params = protocol.validate_params(
            "apply_shape_keys",
            {
                "object": artifact["object"],
                "name_hint": artifact["name_hint"],
                "frame_start": artifact["frame_start"],
                "frame_end": artifact["frame_end"],
                "samples": artifact["samples"],
            },
        )
        receipt = operations.apply_shape_keys(params)
        self.assertEqual(receipt["key_count"], len(artifact["samples"]))
        self.assertEqual(receipt["shape_keys"], sorted(KVRC_SHAPE_KEYS))
        action_count, strips = self.scene_counts()
        self.assertEqual(action_count, 1)
        self.assertEqual(len(strips), 1)


if __name__ == "__main__":
    unittest.main()

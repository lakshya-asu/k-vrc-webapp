"""Structural request validation. Pure protocol, no bpy needed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_bridge import protocol  # noqa: E402


def wave_params():
    return {
        "object": "KVRC",
        "name_hint": "wave",
        "frame_start": 1,
        "frame_end": 48,
        "samples": [
            {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
            {"bone": "arm.R", "frame": 24, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
            {"bone": "hand.R", "frame": 48, "location": [0.0, 0.1, 0.0]},
        ],
    }


class EnvelopeTests(unittest.TestCase):
    def refusal(self, obj):
        with self.assertRaises(protocol.Refusal) as ctx:
            request_id, op, params = protocol.validate_envelope(obj)
            protocol.validate_params(op, params)
        return ctx.exception

    def test_valid_envelope(self):
        request_id, op, params = protocol.validate_envelope(
            {"id": "r1", "op": "inspect_rig", "params": {"object": "KVRC"}}
        )
        self.assertEqual(request_id, "r1")
        self.assertEqual(op, "inspect_rig")
        self.assertEqual(protocol.validate_params(op, params), {"object": "KVRC"})

    def test_non_object_request(self):
        self.assertEqual(self.refusal([1, 2]).code, protocol.BAD_REQUEST)

    def test_missing_id(self):
        error = self.refusal({"op": "inspect_rig", "params": {}})
        self.assertEqual(error.code, protocol.BAD_REQUEST)

    def test_unknown_envelope_key(self):
        error = self.refusal(
            {"id": "r1", "op": "inspect_rig", "params": {}, "authority": "perform"}
        )
        self.assertEqual(error.code, protocol.BAD_REQUEST)

    def test_unknown_operation(self):
        error = self.refusal({"id": "r1", "op": "execute_python", "params": {}})
        self.assertEqual(error.code, protocol.UNKNOWN_OPERATION)

    def test_params_must_be_object(self):
        error = self.refusal({"id": "r1", "op": "inspect_rig", "params": []})
        self.assertEqual(error.code, protocol.BAD_REQUEST)


class ParamTests(unittest.TestCase):
    def refusal(self, op, params):
        with self.assertRaises(protocol.Refusal) as ctx:
            protocol.validate_params(op, params)
        return ctx.exception

    def test_inspect_rig_unknown_param(self):
        error = self.refusal("inspect_rig", {"object": "KVRC", "code": "x"})
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_create_action_bad_hint(self):
        error = self.refusal("create_action", {"name_hint": "../etc"})
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_perform_take_valid(self):
        params = protocol.validate_params("perform_take", wave_params())
        self.assertEqual(params["name_hint"], "wave")
        self.assertEqual(len(params["samples"]), 3)

    def test_frame_not_integer(self):
        params = wave_params()
        params["frame_start"] = 1.5
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_frame_range_inverted(self):
        params = wave_params()
        params["frame_start"] = 60
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.FRAME_OUT_OF_RANGE)

    def test_sample_frame_outside_declared_range(self):
        params = wave_params()
        params["samples"][1]["frame"] = 480
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.FRAME_OUT_OF_RANGE)

    def test_sample_frame_not_integer(self):
        params = wave_params()
        params["samples"][0]["frame"] = True
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_non_finite_location_refused(self):
        params = wave_params()
        params["samples"][2]["location"] = [0.0, float("nan"), 0.0]
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_infinite_rotation_refused(self):
        params = wave_params()
        params["samples"][0]["rotation_quaternion"] = [1.0, 0.0, float("inf"), 0.0]
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_quaternion_wrong_length(self):
        params = wave_params()
        params["samples"][0]["rotation_quaternion"] = [1.0, 0.0, 0.0]
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_sample_without_channels(self):
        params = wave_params()
        params["samples"][0] = {"bone": "arm.R", "frame": 1}
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_sample_unknown_key_refused(self):
        params = wave_params()
        params["samples"][0]["keyframes"] = [[1, 0.0]]
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_empty_samples_refused(self):
        params = wave_params()
        params["samples"] = []
        error = self.refusal("perform_take", params)
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_apply_pose_keys_missing_action(self):
        error = self.refusal(
            "apply_pose_keys",
            {
                "object": "KVRC",
                "frame_start": 1,
                "frame_end": 10,
                "samples": [{"bone": "arm.R", "frame": 1, "location": [0, 0, 0]}],
            },
        )
        self.assertEqual(error.code, protocol.INVALID_PARAMS)

    def test_push_to_nla_defaults(self):
        params = protocol.validate_params(
            "push_to_nla", {"object": "KVRC", "action": "ANIMUS_wave_take001"}
        )
        self.assertEqual(params["name_hint"], "take")
        self.assertEqual(params["frame_start"], 1)


if __name__ == "__main__":
    unittest.main()

"""P1 acceptance invariants from docs/animus/poc-plan.md.

1. A wave request creates exactly one new Action and one new NLA strip,
   and existing Actions are unchanged.
2. Repeating the request creates a separate take with new names. Nothing
   is overwritten.
3. An invalid payload or a disconnect mid-request leaves no partial
   artifact. A queue item either fully applies or is discarded.
"""

import copy
import json
import os
import socket
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import fake_bpy  # noqa: E402
from animus_bridge import operations, protocol  # noqa: E402
from animus_bridge.executor import Executor, WorkItem  # noqa: E402
from animus_bridge.server import BridgeServer  # noqa: E402

BONES = ["root", "spine", "head", "arm.L", "arm.R", "hand.L", "hand.R"]

WAVE_PARAMS = {
    "object": "KVRC",
    "name_hint": "wave",
    "frame_start": 1,
    "frame_end": 48,
    "samples": [
        {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"bone": "arm.R", "frame": 12, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
        {"bone": "arm.R", "frame": 24, "rotation_quaternion": [0.98, 0.0, 0.0, -0.2]},
        {"bone": "arm.R", "frame": 36, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
        {"bone": "arm.R", "frame": 48, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"bone": "hand.R", "frame": 24, "location": [0.0, 0.05, 0.0]},
    ],
}


def scene_counts():
    obj = fake_bpy.data.objects.get("KVRC")
    strips = []
    if obj.animation_data is not None:
        strips = [s for t in obj.animation_data.nla_tracks for s in t.strips]
    return len(fake_bpy.data.actions), strips


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        fake_bpy.reset()
        fake_bpy.add_armature_object("KVRC", BONES)
        self.executor = Executor()
        self.executor.install()
        self.responses = []

    def run_request(self, params):
        validated = protocol.validate_params("perform_take", params)
        self.executor.submit(
            WorkItem(
                {"id": "acc", "op": "perform_take", "params": validated},
                self.responses.append,
            )
        )
        fake_bpy.app.timers.pump()
        return self.responses[-1]

    def test_wave_creates_exactly_one_action_and_one_strip(self):
        user_action = fake_bpy.data.actions.new("UserWalk")
        user_snapshot = user_action.key_snapshot()

        response = self.run_request(copy.deepcopy(WAVE_PARAMS))
        self.assertTrue(response["ok"])
        receipt = response["result"]

        action_count, strips = scene_counts()
        self.assertEqual(action_count, 2)
        self.assertEqual(len(strips), 1)
        self.assertEqual(receipt["action"], "ANIMUS_wave_take001")
        self.assertEqual(receipt["strip"], strips[0].name)
        self.assertEqual(strips[0].action.name, receipt["action"])
        self.assertEqual(receipt["sample_count"], 6)
        self.assertEqual(receipt["key_count"], 5 * 4 + 3)

        self.assertEqual(user_action.key_snapshot(), user_snapshot)
        self.assertFalse(user_action.get(operations.MARKER))

    def test_repeat_creates_a_separate_take(self):
        first = self.run_request(copy.deepcopy(WAVE_PARAMS))["result"]
        first_action = fake_bpy.data.actions.get(first["action"])
        first_keys = first_action.key_snapshot()

        second = self.run_request(copy.deepcopy(WAVE_PARAMS))["result"]

        self.assertNotEqual(first["action"], second["action"])
        self.assertNotEqual(first["strip"], second["strip"])
        self.assertNotEqual(first["track"], second["track"])

        action_count, strips = scene_counts()
        self.assertEqual(action_count, 2)
        self.assertEqual(len(strips), 2)
        self.assertEqual(len({s.name for s in strips}), 2)
        self.assertEqual(first_action.key_snapshot(), first_keys)

    def test_invalid_bone_mid_payload_leaves_no_partial_artifact(self):
        params = copy.deepcopy(WAVE_PARAMS)
        params["samples"][3]["bone"] = "tentacle.R"
        response = self.run_request(params)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.UNKNOWN_BONE)
        action_count, strips = scene_counts()
        self.assertEqual(action_count, 0)
        self.assertEqual(strips, [])

    def test_invalid_frame_mid_payload_leaves_no_partial_artifact(self):
        params = copy.deepcopy(WAVE_PARAMS)
        params["samples"][4]["frame"] = 9000
        with self.assertRaises(protocol.Refusal) as ctx:
            protocol.validate_params("perform_take", params)
        self.assertEqual(ctx.exception.code, protocol.FRAME_OUT_OF_RANGE)
        action_count, strips = scene_counts()
        self.assertEqual(action_count, 0)
        self.assertEqual(strips, [])

    def test_disconnect_mid_request_leaves_no_partial_artifact(self):
        server = BridgeServer(self.executor, port=0)
        server.start()
        self.addCleanup(server.stop)

        client = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        payload = json.dumps(
            {"id": "cut", "op": "perform_take", "params": WAVE_PARAMS}
        ).encode("utf-8")
        client.sendall(payload[: len(payload) // 2])
        client.close()
        time.sleep(0.2)
        fake_bpy.app.timers.pump()

        self.assertEqual(self.executor.queue.qsize(), 0)
        action_count, strips = scene_counts()
        self.assertEqual(action_count, 0)
        self.assertEqual(strips, [])


if __name__ == "__main__":
    unittest.main()

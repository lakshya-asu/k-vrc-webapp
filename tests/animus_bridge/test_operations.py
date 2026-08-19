"""Operations against the fake bpy scene: receipts and refusal paths."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import fake_bpy  # noqa: E402
from animus_bridge import operations, protocol  # noqa: E402

BONES = ["root", "spine", "head", "arm.L", "arm.R", "hand.L", "hand.R"]


def wave_samples():
    return [
        {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"bone": "arm.R", "frame": 24, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
        {"bone": "hand.R", "frame": 48, "location": [0.0, 0.1, 0.0]},
    ]


class OperationsBase(unittest.TestCase):
    def setUp(self):
        fake_bpy.reset()
        fake_bpy.add_armature_object("KVRC", BONES)

    def refusal(self, handler, params):
        with self.assertRaises(protocol.Refusal) as ctx:
            handler(params)
        return ctx.exception


class InspectRigTests(OperationsBase):
    def test_reports_bones_actions_and_nla(self):
        existing = fake_bpy.data.actions.new("UserWalk")
        result = operations.inspect_rig({"object": "KVRC"})
        self.assertEqual(result["object"], "KVRC")
        self.assertEqual(result["armature"], "KVRC_rig")
        self.assertEqual(result["bones"], sorted(BONES))
        self.assertIn(existing.name, result["actions"])
        self.assertEqual(result["nla_tracks"], [])

    def test_unknown_object_refused(self):
        error = self.refusal(operations.inspect_rig, {"object": "Ghost"})
        self.assertEqual(error.code, protocol.UNKNOWN_OBJECT)

    def test_non_armature_refused(self):
        fake_bpy.add_plain_object("Cube")
        error = self.refusal(operations.inspect_rig, {"object": "Cube"})
        self.assertEqual(error.code, protocol.NOT_AN_ARMATURE)

    def test_inspect_mutates_nothing(self):
        operations.inspect_rig({"object": "KVRC"})
        self.assertEqual(len(fake_bpy.data.actions), 0)


class CreateActionTests(OperationsBase):
    def test_receipt_names_the_action(self):
        receipt = operations.create_action({"name_hint": "wave"})
        self.assertEqual(receipt, {"action": "ANIMUS_wave_take001"})
        action = fake_bpy.data.actions.get("ANIMUS_wave_take001")
        self.assertTrue(action.get(operations.MARKER))
        self.assertTrue(action.use_fake_user)

    def test_repeat_creates_new_take_name(self):
        first = operations.create_action({"name_hint": "wave"})
        second = operations.create_action({"name_hint": "wave"})
        self.assertNotEqual(first["action"], second["action"])
        self.assertEqual(second["action"], "ANIMUS_wave_take002")
        self.assertEqual(len(fake_bpy.data.actions), 2)


class ApplyPoseKeysTests(OperationsBase):
    def make_action(self):
        return operations.create_action({"name_hint": "wave"})["action"]

    def params(self, action):
        return {
            "object": "KVRC",
            "action": action,
            "frame_start": 1,
            "frame_end": 48,
            "samples": wave_samples(),
        }

    def test_receipt_counts_keys(self):
        action = self.make_action()
        receipt = operations.apply_pose_keys(self.params(action))
        self.assertEqual(receipt["action"], action)
        self.assertEqual(receipt["sample_count"], 3)
        self.assertEqual(receipt["key_count"], 4 + 4 + 3)
        self.assertEqual(receipt["bones"], ["arm.R", "hand.R"])

    def test_keys_land_in_fcurves(self):
        action_name = self.make_action()
        operations.apply_pose_keys(self.params(action_name))
        action = fake_bpy.data.actions.get(action_name)
        snapshot = action.key_snapshot()
        quat_w = snapshot[('pose.bones["arm.R"].rotation_quaternion', 0)]
        self.assertEqual(quat_w, [(1, 1.0), (24, 0.92)])
        loc_y = snapshot[('pose.bones["hand.R"].location', 1)]
        self.assertEqual(loc_y, [(48, 0.1)])

    def test_unknown_action_refused(self):
        error = self.refusal(operations.apply_pose_keys, self.params("Nope"))
        self.assertEqual(error.code, protocol.UNKNOWN_ACTION)

    def test_user_action_protected(self):
        fake_bpy.data.actions.new("UserWalk")
        error = self.refusal(operations.apply_pose_keys, self.params("UserWalk"))
        self.assertEqual(error.code, protocol.PROTECTED_ACTION)
        self.assertEqual(len(fake_bpy.data.actions.get("UserWalk").fcurves), 0)

    def test_unknown_bone_refused_without_mutation(self):
        action = self.make_action()
        params = self.params(action)
        params["samples"][1]["bone"] = "tail"
        error = self.refusal(operations.apply_pose_keys, params)
        self.assertEqual(error.code, protocol.UNKNOWN_BONE)
        self.assertEqual(len(fake_bpy.data.actions.get(action).fcurves), 0)

    def test_unknown_object_refused(self):
        action = self.make_action()
        params = self.params(action)
        params["object"] = "Ghost"
        error = self.refusal(operations.apply_pose_keys, params)
        self.assertEqual(error.code, protocol.UNKNOWN_OBJECT)


class PushToNlaTests(OperationsBase):
    def test_new_track_and_strip(self):
        action = operations.create_action({"name_hint": "wave"})["action"]
        operations.apply_pose_keys(
            {
                "object": "KVRC",
                "action": action,
                "frame_start": 1,
                "frame_end": 48,
                "samples": wave_samples(),
            }
        )
        receipt = operations.push_to_nla(
            {"object": "KVRC", "action": action, "name_hint": "wave", "frame_start": 1}
        )
        self.assertEqual(receipt["action"], action)
        self.assertEqual(receipt["strip"], "ANIMUS_wave_take001")
        self.assertEqual(receipt["track"], "ANIMUS_wave_take001_track")
        obj = fake_bpy.data.objects.get("KVRC")
        self.assertEqual(len(obj.animation_data.nla_tracks), 1)

    def test_repeat_never_overwrites(self):
        action = operations.create_action({"name_hint": "wave"})["action"]
        base = {"object": "KVRC", "action": action, "name_hint": "wave", "frame_start": 1}
        first = operations.push_to_nla(dict(base))
        second = operations.push_to_nla(dict(base))
        self.assertNotEqual(first["strip"], second["strip"])
        obj = fake_bpy.data.objects.get("KVRC")
        strips = [s.name for t in obj.animation_data.nla_tracks for s in t.strips]
        self.assertEqual(len(strips), len(set(strips)))
        self.assertEqual(len(strips), 2)

    def test_unknown_action_refused(self):
        error = self.refusal(
            operations.push_to_nla,
            {"object": "KVRC", "action": "Nope", "name_hint": "wave", "frame_start": 1},
        )
        self.assertEqual(error.code, protocol.UNKNOWN_ACTION)
        obj = fake_bpy.data.objects.get("KVRC")
        self.assertIsNone(obj.animation_data)


if __name__ == "__main__":
    unittest.main()

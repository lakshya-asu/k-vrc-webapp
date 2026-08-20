"""Deterministic fallback tests: same keyword table as fallback.js."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor.contract import validate_actor_plan  # noqa: E402
from animus_actor.fallback import deterministic_actor_plan  # noqa: E402

AUTHORITY = {"actor_id": "kvrc", "control_level": "perform"}


class FallbackTests(unittest.TestCase):
    def plan_for(self, instruction, **extra):
        request = {"instruction": instruction}
        request.update(extra)
        return deterministic_actor_plan(request)

    def test_hello_line_maps_to_warm_wave(self):
        plan = self.plan_for("Hello, I am K-VRC")
        body = plan["beats"][0]["body"]
        self.assertEqual(body["action"], "gesture")
        self.assertEqual(body["gesture"], "wave")
        self.assertEqual(body["style"], "warm")

    def test_walk_maps_to_walk_to_with_target(self):
        plan = self.plan_for("Walk to the door", target="door")
        body = plan["beats"][0]["body"]
        self.assertEqual(body["action"], "walk_to")
        self.assertEqual(body["target"], "door")

    def test_look_maps_to_turn_to(self):
        plan = self.plan_for("Look at the camera")
        self.assertEqual(plan["beats"][0]["body"]["action"], "turn_to")

    def test_wait_maps_to_wait(self):
        plan = self.plan_for("Hold still for a moment")
        self.assertEqual(plan["beats"][0]["body"]["action"], "wait")

    def test_other_lines_map_to_talk_gesture(self):
        plan = self.plan_for("Recite the shipping forecast")
        body = plan["beats"][0]["body"]
        self.assertEqual(body["action"], "gesture")
        self.assertEqual(body["gesture"], "talk")

    def test_speech_is_carried_and_capped(self):
        plan = self.plan_for("Say something", speech="a" * 600)
        speech = plan["beats"][0]["speech"]
        self.assertEqual(len(speech["text"]), 500)
        self.assertEqual(speech["delivery"], "neutral")

    def test_no_speech_means_null_channel(self):
        plan = self.plan_for("Wave")
        self.assertIsNone(plan["beats"][0]["speech"])

    def test_empty_instruction_still_validates(self):
        plan = self.plan_for("")
        self.assertEqual(plan["summary"], "Use the deterministic idle behavior.")
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertTrue(checked["ok"], checked["errors"])

    def test_every_keyword_branch_validates(self):
        for instruction in (
            "wave hello",
            "walk over there",
            "look left",
            "wait here",
            "unmatched line",
        ):
            plan = self.plan_for(instruction, target="mark", speech="Hi.")
            checked = validate_actor_plan(plan, AUTHORITY)
            self.assertTrue(checked["ok"], (instruction, checked["errors"]))


if __name__ == "__main__":
    unittest.main()

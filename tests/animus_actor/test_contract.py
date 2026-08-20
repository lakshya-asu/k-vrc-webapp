"""Contract validator tests: the Python port must refuse what JS refuses."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor.contract import (  # noqa: E402
    CONTROL_LEVELS,
    SCHEMA_VERSION,
    validate_actor_plan,
)
from animus_actor.fallback import deterministic_actor_plan  # noqa: E402

AUTHORITY = {"actor_id": "kvrc", "control_level": "perform"}


def valid_plan():
    return deterministic_actor_plan(
        {"instruction": "Wave and say hello.", "target": "camera", "speech": "Hi."}
    )


class ContractTests(unittest.TestCase):
    def test_fallback_plan_validates(self):
        checked = validate_actor_plan(valid_plan(), AUTHORITY)
        self.assertTrue(checked["ok"], checked["errors"])
        value = checked["value"]
        self.assertEqual(value["schema_version"], SCHEMA_VERSION)
        self.assertEqual(value["actor_id"], "kvrc")
        self.assertEqual(value["control_level"], "perform")
        self.assertEqual(value["summary"], "Wave and say hello.")

    def test_non_object_plan_refused(self):
        checked = validate_actor_plan(["not", "a", "plan"], AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertEqual(checked["errors"], ["plan must be an object"])

    def test_missing_actor_id_refused(self):
        checked = validate_actor_plan(valid_plan(), {"control_level": "perform"})
        self.assertFalse(checked["ok"])
        self.assertIn("authority.actor_id is required", checked["errors"])

    def test_bad_control_level_refused(self):
        checked = validate_actor_plan(
            valid_plan(), {"actor_id": "kvrc", "control_level": "autopilot"}
        )
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("authority.control_level" in error for error in checked["errors"])
        )
        self.assertEqual(CONTROL_LEVELS, ("suggest", "preview", "perform"))

    def test_forbidden_keyframes_refused_anywhere(self):
        plan = valid_plan()
        plan["beats"][0]["body"]["keyframes"] = [{"bone": "arm.R", "frame": 1}]
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("forbidden" in error for error in checked["errors"]), checked["errors"]
        )

    def test_missing_summary_refused(self):
        plan = valid_plan()
        del plan["summary"]
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(any("plan.summary" in error for error in checked["errors"]))

    def test_empty_beats_refused(self):
        plan = valid_plan()
        plan["beats"] = []
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertIn("plan.beats must contain at least one beat", checked["errors"])

    def test_nine_beats_refused(self):
        plan = valid_plan()
        beat = plan["beats"][0]
        plan["beats"] = [dict(beat, id=f"beat-{index}") for index in range(9)]
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertIn("plan.beats may contain at most 8 beats", checked["errors"])

    def test_channelless_beat_refused(self):
        plan = valid_plan()
        plan["beats"][0].update(body=None, gaze=None, face=None, speech=None)
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("at least one actor channel" in error for error in checked["errors"])
        )

    def test_walk_without_target_refused(self):
        plan = valid_plan()
        plan["beats"][0]["body"] = {"action": "walk_to", "style": "neutral"}
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("target is required for walk_to" in error for error in checked["errors"])
        )

    def test_out_of_range_intensity_refused(self):
        plan = valid_plan()
        plan["beats"][0]["gaze"]["intensity"] = 1.5
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("intensity" in error for error in checked["errors"])
        )

    def test_boolean_at_ms_refused(self):
        plan = valid_plan()
        plan["beats"][0]["at_ms"] = True
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(any("at_ms" in error for error in checked["errors"]))

    def test_unknown_top_level_key_refused(self):
        plan = valid_plan()
        plan["notes"] = "sneaky"
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertIn("plan.notes is not allowed", checked["errors"])

    def test_long_speech_refused(self):
        plan = valid_plan()
        plan["beats"][0]["speech"] = {"text": "x" * 501}
        checked = validate_actor_plan(plan, AUTHORITY)
        self.assertFalse(checked["ok"])
        self.assertTrue(any("speech.text" in error for error in checked["errors"]))


if __name__ == "__main__":
    unittest.main()

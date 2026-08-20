"""Model brain tests: prompt build, JSON extraction, retry-then-fallback.

No network and no model anywhere here; the requester is injected. The
live endpoint is exercised by the acceptance runs, not by this suite.
"""

import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor import loop, model_brain  # noqa: E402
from animus_actor.contract import SCHEMA_VERSION  # noqa: E402

AUTHORITY = {"actor_id": "kvrc", "control_level": "perform"}


def valid_plan(summary="Greet the viewer"):
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "beats": [
            {
                "id": "greet-1",
                "at_ms": 0,
                "duration_ms": 1800,
                "body": {
                    "action": "gesture",
                    "gesture": "wave",
                    "style": "warm",
                    "intensity": 0.7,
                },
                "gaze": {"target": "camera", "intensity": 0.7},
                "face": {"expression": "warm_amused", "intensity": 0.5},
                "speech": None,
            }
        ],
    }


def make_requester(candidates):
    """Requester returning (or raising) each candidate in order."""
    queue = list(candidates)

    def requester(request, base_url=None, model=None, timeout_ms=None):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return requester


class ExtractJsonTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(model_brain.extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(
            model_brain.extract_json('```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_first_balanced_object_wins(self):
        text = '{"a": {"b": 2}} {"second": true} trailing prose'
        self.assertEqual(model_brain.extract_json(text), {"a": {"b": 2}})

    def test_braces_inside_strings_do_not_confuse_depth(self):
        text = '{"a": "closing } brace and \\" quote"}'
        self.assertEqual(
            model_brain.extract_json(text),
            {"a": 'closing } brace and " quote'},
        )

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            model_brain.extract_json("no object here")

    def test_unclosed_object_raises(self):
        with self.assertRaises(ValueError):
            model_brain.extract_json('{"a": 1')


class PromptTests(unittest.TestCase):
    def test_user_prompt_shape(self):
        prompt = model_brain.build_actor_user_prompt(
            {"instruction": "Wave", "target": "camera", "speech": "Hi"}
        )
        self.assertTrue(prompt.endswith("\n/no_think"))
        payload = json.loads(prompt.rsplit("\n", 1)[0])
        self.assertEqual(
            payload,
            {
                "instruction": "Wave",
                "target": "camera",
                "speech": "Hi",
                "scene": {},
                "capabilities": {},
            },
        )

    def test_system_prompt_names_the_contract(self):
        self.assertIn(f'schema_version "{SCHEMA_VERSION}"', model_brain.ACTOR_SYSTEM_PROMPT)
        self.assertIn("gesture", model_brain.ACTOR_SYSTEM_PROMPT)
        self.assertIn("Never emit Python", model_brain.ACTOR_SYSTEM_PROMPT)


class ModelActorPlanTests(unittest.TestCase):
    REQUEST = {"instruction": "Wave and say hello.", "target": "camera",
               "speech": "Hello, I am K-VRC"}

    def test_first_attempt_success(self):
        plan = model_brain.model_actor_plan(
            self.REQUEST,
            AUTHORITY,
            attempts=3,
            requester=make_requester([valid_plan()]),
        )
        self.assertEqual(plan["provenance"]["operator"], "local-small")
        self.assertIs(plan["provenance"]["fallback"], False)
        self.assertEqual(plan["provenance"]["model"], model_brain.DEFAULT_MODEL)
        self.assertNotIn("prior_failures", plan["provenance"])
        self.assertEqual(plan["control_level"], "perform")

    def test_retry_after_invalid_plan_records_failure(self):
        invalid = {"schema_version": SCHEMA_VERSION, "summary": "bad",
                   "keyframes": [{"bone": "arm.R"}]}
        plan = model_brain.model_actor_plan(
            self.REQUEST,
            AUTHORITY,
            attempts=3,
            requester=make_requester([invalid, valid_plan()]),
        )
        self.assertEqual(plan["provenance"]["operator"], "local-small-retry1")
        self.assertIs(plan["provenance"]["fallback"], False)
        failures = plan["provenance"]["prior_failures"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["provider"], "local-small")

    def test_all_attempts_fail_falls_back_loudly(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            plan = model_brain.model_actor_plan(
                self.REQUEST,
                AUTHORITY,
                attempts=2,
                requester=make_requester(
                    [RuntimeError("provider unreachable: refused"),
                     {"schema_version": SCHEMA_VERSION, "summary": "bad",
                      "keyframes": []}]
                ),
            )
        self.assertEqual(plan["provenance"]["operator"], "deterministic")
        self.assertIs(plan["provenance"]["fallback"], True)
        self.assertEqual(len(plan["provenance"]["prior_failures"]), 2)
        self.assertIn("MODEL BRAIN FELL BACK", stderr.getvalue())
        # The fallback still speaks the requested line.
        self.assertEqual(
            plan["beats"][0]["speech"]["text"], "Hello, I am K-VRC"
        )

    def test_model_cannot_grab_authority(self):
        # A model plan that tries to set control_level violates the
        # contract outright, so it is rejected and the deterministic
        # fallback performs under the CALLER's control level.
        sneaky = valid_plan()
        sneaky["control_level"] = "perform"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            plan = model_brain.model_actor_plan(
                self.REQUEST,
                {"actor_id": "kvrc", "control_level": "suggest"},
                attempts=1,
                requester=make_requester([sneaky]),
            )
        self.assertIs(plan["provenance"]["fallback"], True)
        self.assertIn(
            "control_level", plan["provenance"]["prior_failures"][0]["reason"]
        )
        self.assertEqual(plan["control_level"], "suggest")

    def test_clean_model_plan_gets_callers_control_level(self):
        plan = model_brain.model_actor_plan(
            self.REQUEST,
            {"actor_id": "kvrc", "control_level": "suggest"},
            attempts=1,
            requester=make_requester([valid_plan()]),
        )
        self.assertIs(plan["provenance"]["fallback"], False)
        self.assertEqual(plan["control_level"], "suggest")


class LoopBrainTests(unittest.TestCase):
    def test_unknown_brain_refused(self):
        with self.assertRaises(loop.ActorLoopError):
            loop.build_plan("Hello", brain="psychic")

    def test_fallback_brain_stays_default(self):
        plan = loop.build_plan("Hello, I am K-VRC")
        self.assertEqual(plan["provenance"]["operator"], "deterministic")
        self.assertIs(plan["provenance"]["fallback"], True)

    def test_model_brain_routes_through_model_actor_plan(self):
        with mock.patch.object(
            model_brain,
            "request_model_plan",
            side_effect=lambda request, **kwargs: valid_plan(),
        ):
            plan = loop.build_plan("Hello, I am K-VRC", brain="model")
        self.assertEqual(plan["provenance"]["operator"], "local-small")
        self.assertIs(plan["provenance"]["fallback"], False)


if __name__ == "__main__":
    unittest.main()

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
from animus_actor.contract import (  # noqa: E402
    FACE_GLYPH_BROWS,
    FACE_GLYPH_EYES,
    FACE_GLYPH_MOODS,
    FACE_GLYPH_MOUTHS,
    SCHEMA_VERSION,
)

AUTHORITY = {"actor_id": "kvrc", "control_level": "perform"}
HERE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def glyph_plan():
    """A valid plan whose second beat composes the visor itself."""
    plan = valid_plan()
    plan["beats"].append(
        {
            "id": "greet-2",
            "at_ms": 1800,
            "duration_ms": 1200,
            "body": None,
            "gaze": None,
            "face_glyph": {
                "eyes": "happy_arc",
                "mouth": "grin_rect",
                "mood": "warm",
                "intensity": 0.8,
            },
            "speech": None,
        }
    )
    return plan


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

    def test_plan_schema_matches_the_js_provider_fixture(self):
        # The decode-time grammar schema must stay identical in both
        # languages; the committed fixture is generated from the JS
        # module and pins them together.
        fixture_path = os.path.join(
            os.path.dirname(os.path.dirname(HERE_DIR)),
            "tests", "animus_bridge", "fixtures", "actor_plan_schema.json",
        )
        with open(fixture_path, "r", encoding="utf-8-sig") as handle:
            fixture = json.load(handle)
        self.assertEqual(model_brain.ACTOR_PLAN_JSON_SCHEMA, fixture)

    def test_plan_schema_is_structural_and_names_every_channel(self):
        schema = model_brain.ACTOR_PLAN_JSON_SCHEMA
        beat = schema["properties"]["beats"]["items"]
        for channel in ("body", "gaze", "face", "face_glyph", "speech"):
            self.assertIn(channel, beat["properties"])
        glyph_forms = beat["properties"]["face_glyph"]["anyOf"]
        self.assertEqual(glyph_forms[0], {"type": "null"})
        self.assertEqual(
            glyph_forms[1]["properties"]["eyes"]["enum"],
            list(FACE_GLYPH_EYES),
        )
        self.assertEqual(glyph_forms[2]["required"], ["text"])
        self.assertEqual(beat["required"], ["id", "at_ms", "duration_ms"])

    def test_system_prompt_teaches_the_glyph_channel(self):
        prompt = model_brain.ACTOR_SYSTEM_PROMPT
        self.assertIn("face_glyph", prompt)
        for vocab in (FACE_GLYPH_EYES, FACE_GLYPH_BROWS, FACE_GLYPH_MOUTHS,
                      FACE_GLYPH_MOODS):
            self.assertIn(", ".join(vocab), prompt)
        self.assertIn("1 to 6 characters", prompt)
        self.assertIn("Never use face and face_glyph in the same beat.", prompt)
        # The example teaches the shape the validator accepts.
        self.assertIn('"face_glyph":{"eyes":"happy_arc"', prompt)


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

    def test_glyph_plan_is_marked_model_authored(self):
        plan = model_brain.model_actor_plan(
            self.REQUEST,
            AUTHORITY,
            attempts=1,
            requester=make_requester([glyph_plan()]),
        )
        self.assertIs(plan["provenance"]["fallback"], False)
        self.assertEqual(plan["provenance"]["face_glyph"], "model-authored")

    def test_plain_plan_carries_no_glyph_marker(self):
        plan = model_brain.model_actor_plan(
            self.REQUEST,
            AUTHORITY,
            attempts=1,
            requester=make_requester([valid_plan()]),
        )
        self.assertNotIn("face_glyph", plan["provenance"])

    def test_fallback_plan_carries_no_glyph_marker(self):
        with contextlib.redirect_stderr(io.StringIO()):
            plan = model_brain.model_actor_plan(
                self.REQUEST,
                AUTHORITY,
                attempts=1,
                requester=make_requester([RuntimeError("refused")]),
            )
        self.assertIs(plan["provenance"]["fallback"], True)
        self.assertNotIn("face_glyph", plan["provenance"])

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

    def test_loop_request_advertises_the_glyph_capability(self):
        seen = {}

        def capture(request, **kwargs):
            seen.update(request)
            return valid_plan()

        with mock.patch.object(
            model_brain, "request_model_plan", side_effect=capture
        ):
            loop.build_plan("Hello, I am K-VRC", brain="model")
        self.assertIs(seen["capabilities"]["face_glyph"], True)


if __name__ == "__main__":
    unittest.main()

"""The selectable TTS backend seam (kokoro default, chatterbox opt-in).

No model is loaded anywhere here: these tests cover the profile
validation, the voice-job mapping, the interpreter choice, and the
plan-file replay path that the reel re-renders use.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from animus_actor.embodiment import (  # noqa: E402
    load_profile,
    map_plan_to_bridge_jobs,
    validate_embodiment_profile,
)
from animus_actor.fallback import deterministic_actor_plan  # noqa: E402
from animus_actor.contract import validate_actor_plan  # noqa: E402
from animus_actor.loop import ActorLoopError, load_plan_file  # noqa: E402
from animus_actor.voice import find_voice_python  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROFILE_PATH = os.path.join(
    REPO, "src", "animus", "embodiment", "kvrc-testrig.profile.json"
)


def profile_dict():
    with open(PROFILE_PATH, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def validated_plan(speech="Hello, I am K-VRC"):
    request = {
        "instruction": speech,
        "target": "camera",
        "speech": speech,
        "capabilities": {"body": True, "gaze": True, "face": True, "speech": True},
    }
    candidate = deterministic_actor_plan(request)
    checked = validate_actor_plan(
        candidate, {"actor_id": "kvrc", "control_level": "perform"}
    )
    assert checked["ok"], checked["errors"]
    return checked["value"]


class BackendProfileValidationTests(unittest.TestCase):
    def test_backend_chatterbox_is_accepted(self):
        profile = profile_dict()
        profile["speech"]["backend"] = "chatterbox"
        profile["speech"]["tts"] = {"exaggeration": 0.7, "cfg_weight": 0.35}
        checked = validate_embodiment_profile(profile)
        self.assertTrue(checked["ok"], checked["errors"])

    def test_unknown_backend_is_refused(self):
        profile = profile_dict()
        profile["speech"]["backend"] = "elevenlabs"
        checked = validate_embodiment_profile(profile)
        self.assertFalse(checked["ok"])
        self.assertTrue(
            any("speech.backend" in error for error in checked["errors"])
        )

    def test_non_object_tts_is_refused(self):
        profile = profile_dict()
        profile["speech"]["tts"] = "loud"
        checked = validate_embodiment_profile(profile)
        self.assertFalse(checked["ok"])
        self.assertTrue(any("speech.tts" in error for error in checked["errors"]))


class BackendVoiceJobTests(unittest.TestCase):
    def test_default_backend_is_kokoro_with_no_opts(self):
        profile = load_profile(PROFILE_PATH)
        jobs = map_plan_to_bridge_jobs(validated_plan(), profile)
        job = jobs["voice_jobs"][0]
        self.assertEqual(job["backend"], "kokoro")
        self.assertEqual(job["tts_opts"], {})

    def test_chatterbox_backend_and_opts_reach_the_job(self):
        profile = profile_dict()
        profile["speech"]["backend"] = "chatterbox"
        profile["speech"]["tts"] = {
            "exaggeration": 0.7,
            "cfg_weight": 0.35,
            "temperature": 0.8,
            "device": "cuda",
        }
        checked = validate_embodiment_profile(profile)
        self.assertTrue(checked["ok"], checked["errors"])
        jobs = map_plan_to_bridge_jobs(validated_plan(), checked["value"])
        job = jobs["voice_jobs"][0]
        self.assertEqual(job["backend"], "chatterbox")
        self.assertEqual(job["tts_opts"]["exaggeration"], 0.7)
        self.assertEqual(job["tts_opts"]["cfg_weight"], 0.35)
        self.assertEqual(job["tts_opts"]["device"], "cuda")


class BackendInterpreterTests(unittest.TestCase):
    def test_global_override_wins_for_any_backend(self):
        old = os.environ.get("ANIMUS_VOICE_PYTHON")
        os.environ["ANIMUS_VOICE_PYTHON"] = r"C:\fake\python.exe"
        try:
            self.assertEqual(
                find_voice_python("chatterbox"), r"C:\fake\python.exe"
            )
            self.assertEqual(find_voice_python("kokoro"), r"C:\fake\python.exe")
        finally:
            if old is None:
                del os.environ["ANIMUS_VOICE_PYTHON"]
            else:
                os.environ["ANIMUS_VOICE_PYTHON"] = old

    def test_xtts_backend_and_opts_reach_the_job(self):
        profile = profile_dict()
        profile["speech"]["backend"] = "xtts"
        profile["speech"]["tts"] = {
            "speaker": "Torcull Diarmuid",
            "tempo": 1.2,
            "temperature": 0.7,
        }
        checked = validate_embodiment_profile(profile)
        self.assertTrue(checked["ok"], checked["errors"])
        jobs = map_plan_to_bridge_jobs(validated_plan(), checked["value"])
        job = jobs["voice_jobs"][0]
        self.assertEqual(job["backend"], "xtts")
        self.assertEqual(job["tts_opts"]["speaker"], "Torcull Diarmuid")
        self.assertEqual(job["tts_opts"]["tempo"], 1.2)

    def test_xtts_env_override(self):
        old = os.environ.get("ANIMUS_VOICE_PYTHON_XTTS")
        os.environ["ANIMUS_VOICE_PYTHON_XTTS"] = r"C:\fake\xtts.exe"
        try:
            self.assertEqual(find_voice_python("xtts"), r"C:\fake\xtts.exe")
        finally:
            if old is None:
                del os.environ["ANIMUS_VOICE_PYTHON_XTTS"]
            else:
                os.environ["ANIMUS_VOICE_PYTHON_XTTS"] = old

    def test_chatterbox_env_override(self):
        old = os.environ.get("ANIMUS_VOICE_PYTHON_CHATTERBOX")
        os.environ["ANIMUS_VOICE_PYTHON_CHATTERBOX"] = r"C:\fake\cbx.exe"
        try:
            self.assertEqual(find_voice_python("chatterbox"), r"C:\fake\cbx.exe")
        finally:
            if old is None:
                del os.environ["ANIMUS_VOICE_PYTHON_CHATTERBOX"]
            else:
                os.environ["ANIMUS_VOICE_PYTHON_CHATTERBOX"] = old


class PlanFileReplayTests(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_replays_a_bare_plan_and_keeps_provenance(self):
        plan = dict(validated_plan())
        plan["provenance"] = {
            "operator": "local-small",
            "model": "Qwen3-4B-Q4_K_M",
            "fallback": False,
        }
        path = self._write(plan)
        replayed = load_plan_file(path)
        self.assertEqual(replayed["provenance"]["operator"], "local-small")
        self.assertFalse(replayed["provenance"]["fallback"])
        self.assertEqual(
            replayed["provenance"]["replayed_from"], os.path.basename(path)
        )
        self.assertEqual(replayed["control_level"], "perform")

    def test_replays_the_plan_inside_a_receipt(self):
        plan = dict(validated_plan())
        plan["provenance"] = {"operator": "x", "model": None, "fallback": True}
        path = self._write({"kind": "animus_actor_receipt", "plan": plan})
        replayed = load_plan_file(path)
        self.assertEqual(len(replayed["beats"]), len(plan["beats"]))
        self.assertIn("replayed_from", replayed["provenance"])

    def test_invalid_plan_is_refused_loudly(self):
        path = self._write({"plan": {"beats": "nope"}})
        with self.assertRaises(ActorLoopError):
            load_plan_file(path)


if __name__ == "__main__":
    unittest.main()

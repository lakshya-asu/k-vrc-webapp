"""Pipeline tests.

convert_only runs fully offline against the fixture. The live end-to-end
test runs only when both Kokoro and Rhubarb are installed, and skips
cleanly otherwise so the suite never needs the network or the binaries.
"""

import json
import os
import tempfile
import unittest

from animus_voice.pipeline import convert_only, run_pipeline
from animus_voice.rhubarb import rhubarb_available
from animus_voice.tts_kokoro import tts_available

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "hello.rhubarb.json")


class ConvertOnlyTest(unittest.TestCase):
    def test_writes_take_json(self):
        with tempfile.TemporaryDirectory() as out:
            artifact, path = convert_only(os.path.abspath(FIXTURE), out, stem="hello")
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(os.path.basename(path), "hello.animus.json")
            with open(path, "r", encoding="utf-8") as handle:
                on_disk = json.load(handle)
            self.assertEqual(on_disk["cue_count"], artifact["cue_count"])
            self.assertEqual(on_disk["kind"], "animus_viseme_take")

    def test_convert_only_is_deterministic_on_disk(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _, pa = convert_only(os.path.abspath(FIXTURE), a, stem="hello")
            _, pb = convert_only(os.path.abspath(FIXTURE), b, stem="hello")
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())


@unittest.skipUnless(
    tts_available() and rhubarb_available(),
    "live pipeline needs Kokoro TTS and the Rhubarb binary installed",
)
class LivePipelineTest(unittest.TestCase):
    def test_text_to_take(self):
        with tempfile.TemporaryDirectory() as out:
            receipt = run_pipeline(
                "Hello, I am K-VRC.", out, stem="hello", voice="af_heart"
            )
            self.assertTrue(os.path.isfile(receipt["wav"]))
            self.assertTrue(os.path.isfile(receipt["cue_json"]))
            self.assertTrue(os.path.isfile(receipt["take_json"]))
            artifact = receipt["artifact"]
            self.assertEqual(artifact["kind"], "animus_viseme_take")
            self.assertGreater(artifact["cue_count"], 0)
            self.assertIn("audio_sha256", artifact["source"])


if __name__ == "__main__":
    unittest.main()

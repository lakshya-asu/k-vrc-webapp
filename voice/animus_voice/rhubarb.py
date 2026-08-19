"""Rhubarb Lip Sync CLI runner.

Rhubarb is MIT (https://github.com/DanielSWolf/rhubarb-lip-sync). It is a
standalone binary. This module locates it, runs it on an exact WAV plus
its transcript, and returns the parsed cue JSON. It shells out only; it
imports nothing heavy, so importing this module is always cheap.

The transcript is passed with Rhubarb's --dialogFile option. Giving
Rhubarb the exact spoken text improves its phonetic recognition, which is
why the pipeline renders audio first and lip-syncs that same audio with
that same text (see research: animus-rigging-and-voice.md, speech-to-face
path).
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

# Environment override for a binary that is not on PATH.
ENV_BINARY = "RHUBARB_BIN"
# Names to look for on PATH.
_CANDIDATES = ("rhubarb", "rhubarb.exe")


def find_rhubarb():
    """Return the path to the Rhubarb binary, or None when absent.

    Checks RHUBARB_BIN first, then PATH, then a vendored copy under
    voice/animus_voice/vendor/rhubarb/.
    """
    override = os.environ.get(ENV_BINARY)
    if override and os.path.isfile(override):
        return override
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    here = os.path.dirname(os.path.abspath(__file__))
    vendor = os.path.join(here, "vendor")
    for name in _CANDIDATES:
        # A flat vendor/rhubarb/ copy or an unpacked versioned release
        # folder such as vendor/Rhubarb-Lip-Sync-1.14.0-Windows/.
        direct = os.path.join(vendor, "rhubarb", name)
        if os.path.isfile(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(vendor, "**", name), recursive=True))
        if matches:
            return matches[0]
    return None


def rhubarb_available():
    return find_rhubarb() is not None


def run_rhubarb(
    wav_path,
    transcript=None,
    out_json=None,
    recognizer="pocketSphinx",
    extended_shapes=True,
    binary=None,
):
    """Run Rhubarb on wav_path and return the parsed cue dict.

    When out_json is given the raw Rhubarb JSON is also written there.
    Raises RuntimeError when the binary is missing or the run fails.
    extended_shapes True enables the G and H shapes.
    """
    binary = binary or find_rhubarb()
    if binary is None:
        raise RuntimeError(
            "Rhubarb binary not found; set RHUBARB_BIN, put it on PATH, or "
            "vendor it under voice/animus_voice/vendor/rhubarb/"
        )
    if not os.path.isfile(wav_path):
        raise RuntimeError(f"wav file not found: {wav_path}")

    cmd = [binary, "-f", "json", "-r", recognizer]
    if extended_shapes:
        # GHX extended mouth shapes. G and H are the extra shapes; X is
        # always available.
        cmd += ["--extendedShapes", "GHX"]

    dialog_path = None
    try:
        if transcript:
            handle = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            handle.write(transcript)
            handle.close()
            dialog_path = handle.name
            cmd += ["--dialogFile", dialog_path]
        cmd.append(wav_path)

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        if dialog_path and os.path.isfile(dialog_path):
            os.unlink(dialog_path)

    if completed.returncode != 0:
        raise RuntimeError(
            f"rhubarb exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        )

    try:
        data = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"rhubarb output was not valid JSON: {error}")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")

    return data

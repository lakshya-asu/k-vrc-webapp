"""Voice stage of the actor loop: text to viseme take artifact.

Two modes, both CPU only:

    live     spawns the .venv-voice interpreter and runs the full
             Kokoro + Rhubarb pipeline (python -m animus_voice speak).
             This is the real voice track path.
    convert  rebuilds the viseme take from the committed fixture cue
             file in-process (the converter is pure stdlib), so the
             loop runs end to end with no tools installed. Used by
             tests and as the no-tools fallback the caller must choose
             explicitly; nothing falls back silently.
"""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(REPO, "voice")
FIXTURE_CUES = os.path.join(
    REPO, "voice", "animus_voice", "fixtures", "hello.rhubarb.json"
)

VOICE_MODES = ("live", "convert", "skip")


def find_voice_python(backend="kokoro"):
    """The interpreter that has the backend's TTS installed.

    ANIMUS_VOICE_PYTHON always wins. Otherwise kokoro lives in
    .venv-voice and chatterbox in .venv-voice2 (two envs because their
    torch pins differ); ANIMUS_VOICE_PYTHON_CHATTERBOX overrides the
    chatterbox env specifically.
    """
    env = os.environ.get("ANIMUS_VOICE_PYTHON")
    if env:
        return env
    if backend == "chatterbox":
        env = os.environ.get("ANIMUS_VOICE_PYTHON_CHATTERBOX")
        if env:
            return env
        venv = os.path.join(REPO, ".venv-voice2", "Scripts", "python.exe")
        if os.path.exists(venv):
            return venv
    if backend == "xtts":
        env = os.environ.get("ANIMUS_VOICE_PYTHON_XTTS")
        if env:
            return env
        venv = os.path.join(REPO, ".venv-voice3", "Scripts", "python.exe")
        if os.path.exists(venv):
            return venv
    venv = os.path.join(REPO, ".venv-voice", "Scripts", "python.exe")
    if os.path.exists(venv):
        return venv
    return "python"


def _run_convert(job, out_dir):
    if VOICE_DIR not in sys.path:
        sys.path.insert(0, VOICE_DIR)
    from animus_voice.pipeline import convert_only

    artifact, take_path = convert_only(
        FIXTURE_CUES,
        out_dir,
        stem=job["stem"],
        fps=job["fps"],
        frame_start=job["frame_start"],
        obj=job["object"],
        name_hint=job["name_hint"],
        source={"text": job["text"], "mode": "convert-fixture"},
    )
    return {
        "beat_id": job["beat_id"],
        "voice_mode": "convert",
        "wav": None,
        "take_json": take_path,
        "cue_count": artifact["cue_count"],
        "sample_count": len(artifact["samples"]),
        "duration_ms": artifact["duration_ms"],
        "artifact": artifact,
    }


def _run_live(job, out_dir, timeout=600):
    backend = job.get("backend", "kokoro")
    python = find_voice_python(backend)
    command = [
        python,
        "-m",
        "animus_voice",
        "speak",
        "--backend",
        backend,
        "--text",
        job["text"],
        "--voice",
        job["voice"],
        "--lang",
        job["lang"],
        "--speed",
        str(job["speed"]),
        "--seed",
        str(job["seed"]),
        "--out",
        out_dir,
        "--stem",
        job["stem"],
        "--fps",
        str(job["fps"]),
        "--frame-start",
        str(job["frame_start"]),
        "--object",
        job["object"],
        "--name-hint",
        job["name_hint"],
    ]
    tts_opts = job.get("tts_opts") or {}
    for key, flag in (
        ("exaggeration", "--exaggeration"),
        ("cfg_weight", "--cfg-weight"),
        ("temperature", "--temperature"),
        ("device", "--device"),
        ("pitch_semitones", "--pitch-semitones"),
        ("speaker", "--speaker"),
        ("tempo", "--tempo"),
    ):
        if key in tts_opts:
            command += [flag, str(tts_opts[key])]
    env = dict(os.environ)
    env["PYTHONPATH"] = VOICE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"voice pipeline failed for beat '{job['beat_id']}' "
            f"(exit {result.returncode}): {result.stderr[-500:]}"
        )
    take_path = os.path.join(out_dir, f"{job['stem']}.animus.json")
    with open(take_path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    wav_path = os.path.join(out_dir, f"{job['stem']}.wav")
    return {
        "beat_id": job["beat_id"],
        "voice_mode": "live",
        "wav": wav_path if os.path.exists(wav_path) else None,
        "take_json": take_path,
        "cue_count": artifact["cue_count"],
        "sample_count": len(artifact["samples"]),
        "duration_ms": artifact["duration_ms"],
        "artifact": artifact,
    }


def run_voice_job(job, out_dir, mode):
    """One speech beat to one viseme take artifact. Never touches the GPU."""
    if mode not in ("live", "convert"):
        raise ValueError(f"unknown voice mode '{mode}'")
    os.makedirs(out_dir, exist_ok=True)
    if mode == "convert":
        return _run_convert(job, out_dir)
    return _run_live(job, out_dir)

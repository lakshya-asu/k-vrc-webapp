"""End-to-end Animus voice pipeline.

text in -> Kokoro WAV -> Rhubarb cue JSON -> Animus viseme take JSON.

Every stage writes a file to the output directory so the take is fully
cached with its provenance, as the architecture doc requires: audio,
transcript, voice id, synthesis settings, cue file, and the converted
face track travel together. Regenerating the audio invalidates the old
cue timing, so the pipeline always rewrites all three from one call.

The converter stage is deterministic and pure. The TTS and Rhubarb stages
call out to installed tools; run_pipeline raises a clear error if they are
absent. convert_only rebuilds the viseme take from an existing cue file
with no tools at all, which is what tests and the fixture path use.
"""

import hashlib
import json
import os

from .converter import cues_to_viseme_take, load_rhubarb_cues
from .visemes import PROFILE, PROFILE_VERSION


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def convert_only(
    cue_source,
    out_dir,
    stem="line",
    fps=24,
    frame_start=1,
    obj="KVRC",
    name_hint="animus_speech",
    source=None,
):
    """Build the viseme take from an existing Rhubarb cue file or object.

    No TTS and no Rhubarb binary are used. Returns the artifact and writes
    it to out_dir/<stem>.animus.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    cues, duration = load_rhubarb_cues(cue_source)
    artifact = cues_to_viseme_take(
        cues,
        duration=duration,
        fps=fps,
        frame_start=frame_start,
        obj=obj,
        name_hint=name_hint,
        source=source,
    )
    take_path = os.path.join(out_dir, f"{stem}.animus.json")
    _write_json(take_path, artifact)
    return artifact, take_path


TTS_BACKENDS = ("kokoro", "chatterbox")


def run_pipeline(
    text,
    out_dir,
    stem="line",
    voice="af_heart",
    lang="a",
    speed=1.0,
    seed=0,
    fps=24,
    frame_start=1,
    obj="KVRC",
    name_hint="animus_speech",
    backend="kokoro",
    tts_opts=None,
):
    """Full live pipeline. Requires a TTS backend and Rhubarb installed.

    backend picks the TTS engine: 'kokoro' (default, deterministic,
    CPU) or 'chatterbox' (expressive, built-in voice only, CUDA when
    available). tts_opts carries backend-specific knobs; chatterbox
    reads exaggeration, cfg_weight, temperature, and device from it.

    Writes <stem>.wav, <stem>.rhubarb.json, and <stem>.animus.json into
    out_dir and returns a receipt describing all three plus the artifact.
    Deterministic given the same inputs on the kokoro backend.
    """
    from .rhubarb import run_rhubarb

    if backend not in TTS_BACKENDS:
        raise RuntimeError(
            f"unknown TTS backend '{backend}'; choose from {TTS_BACKENDS}"
        )

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{stem}.wav")
    cue_path = os.path.join(out_dir, f"{stem}.rhubarb.json")
    take_path = os.path.join(out_dir, f"{stem}.animus.json")

    opts = dict(tts_opts or {})
    if backend == "chatterbox":
        from .tts_chatterbox import render_to_wav

        tts_receipt = render_to_wav(
            text,
            wav_path,
            exaggeration=opts.get("exaggeration", 0.5),
            cfg_weight=opts.get("cfg_weight", 0.5),
            temperature=opts.get("temperature", 0.8),
            seed=seed,
            device=opts.get("device", "auto"),
            pitch_semitones=opts.get("pitch_semitones", 0.0),
        )
    else:
        from .tts_kokoro import render_to_wav

        tts_receipt = render_to_wav(
            text, wav_path, voice=voice, lang=lang, speed=speed, seed=seed
        )
    audio_sha = _sha256(wav_path)

    cue_data = run_rhubarb(wav_path, transcript=text, out_json=cue_path)
    cues, duration = load_rhubarb_cues(cue_data)

    source = {
        "text": text,
        "tts": tts_receipt,
        "audio_wav": os.path.basename(wav_path),
        "audio_sha256": audio_sha,
        "rhubarb": {
            "engine": "rhubarb-lip-sync",
            "license": "MIT",
            "cue_file": os.path.basename(cue_path),
        },
        "profile": PROFILE,
        "profile_version": PROFILE_VERSION,
    }

    artifact = cues_to_viseme_take(
        cues,
        duration=duration,
        fps=fps,
        frame_start=frame_start,
        obj=obj,
        name_hint=name_hint,
        source=source,
    )
    _write_json(take_path, artifact)

    return {
        "wav": wav_path,
        "cue_json": cue_path,
        "take_json": take_path,
        "audio_sha256": audio_sha,
        "duration_ms": artifact["duration_ms"],
        "cue_count": artifact["cue_count"],
        "sample_count": len(artifact["samples"]),
        "artifact": artifact,
    }

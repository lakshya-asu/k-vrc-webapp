# Runbook: install and run the live voice pipeline

CPU only. No GPU, no CUDA, no server. This records exactly what ran live
on the build host and how to reproduce it.

## What ran live on the build host (2026-08-19)

Windows 11, CPU only. Both tools installed and the full end-to-end
pipeline ran for the line "Hello, I am K-VRC.":

- Kokoro-82M rendered `hello.wav` (24kHz mono, ~2.1s, ~100KB).
- Rhubarb 1.14.0 produced 10 real mouth cues from that wav plus the
  transcript.
- The converter produced `hello.animus.json` (10 cues, 40 shape-key
  samples, frames 1..51 at 24 fps).
- End to end wall time was about 13s after weights were cached (first run
  is slower because it downloads the Kokoro and spaCy weights).

Determinism observed: the converter is fully deterministic (covered by
tests). Across two separate Kokoro processes the resulting cue timing and
viseme track were identical to the millisecond, but the raw wav bytes were
not bit-identical, so `source.audio_sha256` can differ run to run. The
semantic track is stable; the embedded audio hash is not guaranteed byte
stable across processes on this host.

## Isolated environment

An isolated venv was created inside the worktree so system Python and the
other project envs are untouched:

    py -3.11 -m venv .venv-voice
    .venv-voice\Scripts\python -m pip install --upgrade pip

## Install Kokoro (Apache-2.0), CPU torch

    .venv-voice\Scripts\python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu torch
    .venv-voice\Scripts\python -m pip install kokoro soundfile

First `speak` run downloads the Kokoro-82M weights (~327MB model repo) and
the spaCy `en_core_web_sm` model. These are cached under the user's
HuggingFace and spaCy caches after the first run.

Download sizes recorded for the resource claim:

- CPU torch wheel plus deps: torch 2.13.0+cpu (the large item, a few
  hundred MB unpacked) plus small deps.
- Kokoro stack: kokoro 0.9.4, transformers, misaki, spaCy, soundfile, and
  friends (tens of MB of wheels).
- Kokoro-82M weights: about 327MB, pulled on first `speak`.
- spaCy en_core_web_sm: about 12MB, pulled on first `speak`.
- Rhubarb 1.14.0 Windows zip: 87,374,842 bytes (about 83MB); about 156MB
  unpacked (binary plus PocketSphinx and eSpeak data under res/).

## Install Rhubarb (MIT)

Rhubarb is a standalone binary. Download the release zip and unpack it,
then point the pipeline at it one of three ways:

1. set `RHUBARB_BIN` to the full path of `rhubarb` / `rhubarb.exe`, or
2. put the binary on `PATH`, or
3. unpack the release under `voice/animus_voice/vendor/` (the runner finds
   `vendor/rhubarb/rhubarb.exe` or any unpacked
   `vendor/Rhubarb-Lip-Sync-*/rhubarb.exe`).

On the build host the Windows release was unpacked to
`voice/animus_voice/vendor/Rhubarb-Lip-Sync-1.14.0-Windows/`. That folder
and the venv are gitignored; they are rebuilt locally, not committed.

Latest release URL used:

    https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Windows.zip

## Run

    set PYTHONPATH=voice
    .venv-voice\Scripts\python -m animus_voice doctor
    .venv-voice\Scripts\python -m animus_voice speak \
        --text "Hello, I am K-VRC." --out voice/animus_voice/out --stem hello

Outputs land in the out dir: `hello.wav`, `hello.rhubarb.json`, and
`hello.animus.json`.

## Reproduce the offline path (no tools)

    set PYTHONPATH=voice
    python -m animus_voice convert \
        --cues voice/animus_voice/fixtures/hello.rhubarb.json \
        --out voice/animus_voice/out --stem hello

## Tests

    set PYTHONPATH=voice
    python -m unittest discover -s voice/animus_voice/tests

32 tests. The one live integration test runs only when both tools are
present and skips cleanly otherwise.

## Remaining live step not yet done

Applying the viseme take inside Blender. The P1 bridge writes bone
channels, not shape keys. Applying this track needs one added typed
bridge operation, `apply_shape_keys`, that writes
`key_blocks["<name>"].value` FCurves the way `apply_pose_keys` writes bone
channels. The artifact is already shaped for it (each sample is a
`(frame, shape_key, weight)` triple). That Blender-side step was not run
here; this proof produces and validates the track that would feed it.

# Animus voice pipeline (P2 speech and lip-sync)

Proves the P2 voice stage independently of Blender:

    text -> Kokoro TTS wav -> Rhubarb lip-sync cues -> Animus viseme track

The viseme track is a structured artifact shaped so the P1 Blender bridge
(`blender/animus_bridge`) could later apply it as K-VRC face-screen
shape-key weights. Both tools are license-clear and CPU-viable: Kokoro is
Apache-2.0, Rhubarb is MIT.

## Layout

    visemes.py      Rhubarb shape set and the shape-key weight table
    converter.py    cues -> viseme take artifact, with typed refusals
    tts_kokoro.py   Kokoro-82M CPU wrapper (lazy torch import)
    rhubarb.py      Rhubarb CLI runner and binary locator
    pipeline.py     end-to-end and offline convert_only
    __main__.py     CLI: convert, speak, doctor
    fixtures/       hello.rhubarb.json checked-in cue fixture
    tests/          stdlib unittest, offline; live test skips if tools absent
    MAPPING.md      the viseme -> shape-key mapping and artifact shape
    RUNBOOK.md      how to install the live tools and run a live render

The converter and artifact layer are pure Python: no torch, no binary, no
network. The TTS and Rhubarb stages are optional and skip cleanly when the
tools are absent.

## Quick start

Offline, no tools needed, rebuild the viseme take from a cue file:

    python -m animus_voice convert --cues fixtures/hello.rhubarb.json \
        --out out --stem hello

Report which live tools are installed:

    python -m animus_voice doctor

Live, needs Kokoro and Rhubarb (see RUNBOOK.md):

    python -m animus_voice speak --text "Hello, I am K-VRC." \
        --out out --stem hello --voice af_heart

## Tests

    python -m unittest discover -s tests

The one live integration test skips unless both Kokoro and the Rhubarb
binary are installed, so the suite always runs offline with no network.

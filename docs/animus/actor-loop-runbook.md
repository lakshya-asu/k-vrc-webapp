# Runbook: the one-command actor loop (python -m animus_actor)

One command takes a line of dialogue and produces a complete performed
take in headless Blender:

    python -m animus_actor "Hello, I am K-VRC"

Run it from the repo root. What happens, in order:

1. The line becomes a semantic actor plan through the DETERMINISTIC
   fallback brain (animus_actor/fallback.py, a port of
   src/animus/fallback.js). No model, no GPU.
2. The plan is checked by the strict contract validator
   (animus_actor/contract.py, a port of src/animus/contract.js): beats
   only, never raw keyframes, authority owned by the caller.
3. The embodiment profile maps beats to numbers
   (animus_actor/embodiment.py, a port of src/animus/embodiment.js).
   The plan names things; the profile supplies every quaternion, frame,
   and weight.
4. Speech beats run through the CPU voice pipeline: Kokoro-82M TTS plus
   Rhubarb visemes (voice/animus_voice, spawned in .venv-voice).
5. The loop launches Blender headless with animus_actor/stage.py, which
   builds the profile's test rig, starts the typed bridge, and drains
   the executor on the main thread.
6. Every layer lands as one atomic bridge op (perform_take for body and
   gaze, apply_shape_keys for face and speech), one new Action plus NLA
   strip each, with receipts.
7. The CLI prints a summary: voice track path, viseme count, beats
   executed, and the bridge op receipts. --receipt PATH writes the full
   receipt JSON.

Useful flags:

    --attach --port N          use a bridge that is already listening
                               (the acceptance harness does this)
    --control-level suggest    map everything, perform nothing
    --voice-mode convert       no TTS tools: rebuild visemes from the
                               committed fixture cues
    --voice-mode skip          no speech layer at all
    --blender PATH             stage binary (also ANIMUS_BLENDER; the
                               loop otherwise checks PATH and the
                               ~/tools/blender-4.5 install)
    --receipt PATH             write the full receipt JSON

## Contract boundaries

The bridge (blender/animus_bridge/protocol.py, operations.py) is the
fixed side. The loop maps onto it; it never asks the bridge to change.
The Python contract and mapper must match the JS ones; the parity test
in tests/animus_actor/test_embodiment.py replays the committed JS
fixture and compares ops and params exactly.

## Tests

    python -m unittest discover -s tests/animus_actor -t .    # 43 tests
    blender --background --factory-startup \
        --python blender/animus_bridge/acceptance/run_actor_acceptance.py

The acceptance harness runs the real python -m animus_actor as a
separate process against a live bridge in Blender: a suggest run that
must mutate nothing, then a perform run with live voice where all four
layers (body, gaze, face, speech) must land with keyframes. 17 checks.
Committed evidence: acceptance/results-actor.json and
acceptance/actor-receipt-perform.json.

## What ran live on the build host (2026-08-20)

Windows 11, CPU only, Blender 4.5.12 LTS headless, Kokoro + Rhubarb in
.venv-voice. The default command produced a live wav (10 cues, 40
viseme samples, 2000 ms) and four takes on the test rig
(ANIMUS_wave_take001 40 keys, ANIMUS_gaze_camera_take001 12,
ANIMUS_face_neutral_idle_take001 16, ANIMUS_animus_speech_take001 40).
Acceptance: run_acceptance 27/27, run_director_acceptance 22/22
(fallback mode), run_actor_acceptance 17/17.

## Honest gaps

- The brain here is the deterministic fallback only. Model-authored
  plans stay a director concern (scripts/animus-director.mjs); wiring a
  local model into the Python loop is future work.
- The rig is the plain acceptance armature, not the real K-VRC
  character.
- suggest and preview behave the same: map, never perform.

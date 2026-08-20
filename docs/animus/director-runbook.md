# Runbook: the director loop

One script line becomes performed takes on the rig:

    line -> actor plan (local model or deterministic fallback)
         -> contract validation (src/animus/contract.js)
         -> embodiment mapping (src/animus/embodiment.js + profile JSON)
         -> voice pipeline for speech beats (Kokoro + Rhubarb, CPU)
         -> typed bridge requests over the localhost socket
         -> one Action + one NLA strip per layer, receipts throughout

The embodiment profile (`src/animus/embodiment/kvrc-testrig.profile.json`)
owns every number: gesture pose keys, gaze targets, expression weights
(decision A-008). The model only names things. Authority is caller-owned:
only `--control-level perform` opens a socket (decision A-009).

## Run the director by hand

Requires Node plus, for `perform`, a Blender with the bridge listening.
Voice `live` mode needs the `.venv-voice` setup from
`voice/animus_voice/RUNBOOK.md`; `convert` mode needs nothing installed;
`skip` maps no speech.

    node scripts/animus-director.mjs \
      --line "Wave to the viewer and say hello." \
      --speech "Hello, I am K-VRC." \
      --control-level perform --port 8765 \
      --voice-mode live --receipt receipt.json

Useful flags: `--fallback-only` (no model), `--provider stub-invalid`
(force an invalid plan, proves the fallback path), `--emit-requests`
(print mapped bridge requests and exit before any socket use with
suggest/preview).

Environment: `ANIMUS_LLM_BASE_URL` (default `http://127.0.0.1:8081/v1`),
`ANIMUS_MODEL`, `ANIMUS_TIMEOUT_MS`, `ANIMUS_LLM_ATTEMPTS` (default 3;
the 4B model sometimes emits malformed JSON, retries are caller policy),
`ANIMUS_VOICE_PYTHON`.

## The local model server (port 8081 only)

    powershell scripts/start-animus-model.ps1 -GpuLayers 0

Port 8080 belongs to the fleet 27B server; never reuse it. `-GpuLayers 0`
keeps the run CPU-only. Claim the run in `flux-work/boards/_claims.md`
first, stop by PID, release after.

## Headless acceptance

    blender --background --factory-startup \
      --python blender/animus_bridge/acceptance/run_director_acceptance.py

Environment: `ANIMUS_DIRECTOR_PORT` (default 8792), `ANIMUS_DIRECTOR_LIVE`
("1" uses the local model, otherwise fallback-only),
`ANIMUS_DIRECTOR_VOICE_MODE` (`live` default, or `convert`),
`ANIMUS_DIRECTOR_OUT`, `ANIMUS_DIRECTOR_DEADLINE`.

Three runs, all checked inside Blender: `suggest` mutates nothing,
`perform` lands every mapped layer (gesture, gaze, expression, voiced
mouth take) as its own Action and NLA strip, and `stub-invalid` proves an
invalid plan is rejected by the contract while the deterministic fallback
still performs. Committed results:

- `acceptance/results-director.json`: fallback plan, live voice, 22/22.
- `acceptance/results-director-live.json`: live Qwen3-4B plan
  (`operator: local-small-retry1`, `fallback: false`), live voice, 22/22,
  on Blender 4.5.12 headless, CPU only.
- `acceptance/results-director-live-gpu.json`: same live mode with the
  model on the GPU (`start-animus-model.ps1` defaults, `-ngl 99`),
  22/22, model-authored suggest and perform plans, 2026-08-20.

## Tests

    npm run test:animus                      # 17 JS tests
    python -m unittest discover -s tests/animus_bridge -t .   # 84 fake-bpy tests

`tests/animus_bridge/test_director_requests.py` replays the committed
fixture `tests/animus_bridge/fixtures/director_requests.json` (generated
by the JS mapper) through the real Python bridge validators and
operations, so the two languages cannot drift silently. Regenerate it
with:

    node scripts/animus-director.mjs --fallback-only --voice-mode convert \
      --emit-requests --line "Wave to the viewer and say hello." \
      --speech "Hello, I am K-VRC." > tests/animus_bridge/fixtures/director_requests.json

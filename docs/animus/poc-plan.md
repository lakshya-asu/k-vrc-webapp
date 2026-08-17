# Animus proof plan

## P0: repository and contract

Status: implemented on `feat/animus-poc`.

- One provider-neutral actor-plan contract
- Caller-owned control level
- Strict validation and forbidden raw-control keys
- Local OpenAI-compatible provider
- Deterministic fallback
- Small-model launch script
- Tests for validation, fallback, and provider parsing

Acceptance:

- `npm run test:animus` passes.
- `npm run animus:poc -- --fallback-only` returns valid JSON.
- Invalid provider output cannot reach an executor.

## P1: typed Blender bridge

- Add-on plus localhost socket
- Socket thread queues work
- `bpy.app.timers` drains work on Blender's main thread
- Read-only rig and animation inspection
- Create Action
- Write validated semantic pose samples
- Push Action to named NLA strip
- Return a mutation receipt

Acceptance:

- A wave request creates one new Action and one new NLA strip.
- Existing Actions are unchanged.
- Repeating the request creates a separate take.
- Disconnecting mid-request does not leave an unnamed partial artifact.

## P2: K-VRC embodiment profile

- Map semantic gestures to K-VRC clips
- Add gaze targets
- Add expression mapping
- Add speech and Rhubarb cue import
- Record profile version and source license metadata

Acceptance:

- `wave and say hello` produces body, gaze, face, speech, and lip-sync layers.
- The animator can mute or delete every generated layer.
- The result plays without the actor model running.

## P3: operator fallbacks

- Codex Luna runbook trial
- Hermes Codex Luna runbook trial
- Local-small model trial
- Deterministic-only trial

Acceptance:

- All four routes produce the same contract version.
- All receipts identify their operator.
- A provider outage falls back without an unvalidated scene mutation.

## P4: optional learned jobs

- Kimodo motion generation as a serialized GPU job
- UniRig evaluation as a serialized GPU job
- Stable Fast 3D or TripoSR intake experiment

Acceptance:

- Each job can run with the actor LLM unloaded.
- Each output enters through the same embodiment and NLA validation boundary.
- Licensed clips remain the default fallback.

## Not part of the first proof

- unattended general scene control
- arbitrary rig support
- production-quality image-to-3D
- motion generation beside the current 27B model
- cloud dependency as a required path
- model-authored Python or raw keyframes

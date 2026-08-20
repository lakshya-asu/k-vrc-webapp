# Animus Bridge: manual install into Blender 4.5 LTS

Status note, 2026-08-19 (updated same day): real-Blender acceptance has
RUN and PASSED. Blender 4.5.12 LTS (portable zip from the official
Blender builder CDN, cdn.builder.blender.org, build hash 84afd5f785f7,
399,332,846 bytes, SHA-256 verified against the published checksum) was
installed to `C:\Users\jainl\tools\blender-4.5\` on the build machine.
The headless acceptance harness in `acceptance/run_acceptance.py` ran
all six steps against a live Blender process and all 27 in-Blender
checks passed. Full transcript: `acceptance/transcript-2026-08-19.txt`;
machine-readable results: `acceptance/results-2026-08-19.json`. The
fake-bpy unit suite in `tests/animus_bridge/` still covers every
behavior for machines without Blender.

## What this add-on is

A localhost socket server inside Blender. It accepts newline-delimited
JSON requests on 127.0.0.1 and applies typed animation operations:
`inspect_rig`, `create_action`, `apply_pose_keys`, `apply_shape_keys`,
`push_to_nla`, and the atomic `perform_take`. There is no code
execution path. Socket threads only parse and queue. A `bpy.app.timers`
callback runs the queued work on Blender's main thread. Every mutation
lands in a new Action and a new named NLA strip and returns the names
it created.

## Install (legacy add-on path, simplest)

1. Zip the `animus_bridge` folder itself, so the zip contains
   `animus_bridge/__init__.py` at its top level.
2. In Blender 4.5: Edit, Preferences, Add-ons, arrow menu,
   Install from Disk, pick the zip.
3. Enable "Animus Bridge" in the add-on list.
4. The console prints `[animus_bridge] listening on 127.0.0.1:8765`.

To change the port, set the environment variable `ANIMUS_BRIDGE_PORT`
before starting Blender. Default is 8765.

The folder also carries `blender_manifest.toml`, so it can be built as
an extension with `blender --command extension build` once extension
packaging matters. The legacy path above is enough for the P1 proof.

## Run the wave demo

1. Open a scene with an armature object. Note its object name.
2. Keep Blender in the foreground GUI. The server refuses to start in
   `--background` mode because queued work would never drain (the
   acceptance harness drains the queue manually instead; see below).
3. From any terminal:

   ```
   python blender/animus_bridge/examples/wave_client.py --object YourArmature
   ```

4. Expected result: one new Action named like `ANIMUS_wave_take001`,
   one new NLA track and strip with matching take names, and a JSON
   receipt printed by the client. Run it again and you get `take002`.
   Nothing existing is overwritten.

## Run a speech take from the voice pipeline

The `apply_shape_keys` operation accepts the voice pipeline's viseme
take artifact (`*.animus.json` from `voice/animus_voice`) as a request
payload with a direct field-for-field mapping: `object`, `name_hint`,
`frame_start`, `frame_end`, and the `samples` list are sent as-is. The
artifact's per-sample `at_ms` and `viseme` provenance fields are
accepted by validation and dropped, so no client-side rewriting is
needed. `examples/speech_client.py` does the mapping:

```
python blender/animus_bridge/examples/speech_client.py \
    --artifact voice/animus_voice/out/hello.animus.json \
    --object YourShapeKeyedMesh
```

The Action and NLA strip for a shape-key take land on the Key
datablock (`object.data.shape_keys`), where Blender puts shape-key
animation, not on the object itself. Refusal discipline matches pose
keys: unknown object, object without shape keys, unknown shape key,
non-finite weight, and out-of-range frame each refuse mutating
nothing; success returns a receipt; each request is a fresh named
take and never overwrites.

## Real-Blender acceptance (RAN 2026-08-19, all checks passed)

Headless run, exactly as scripted in `acceptance/run_acceptance.py`:

```
set ANIMUS_VOICE_ARTIFACT=<path to *.animus.json>
blender --background --factory-startup ^
    --python blender/animus_bridge/acceptance/run_acceptance.py
```

Environment: Blender 4.5.12 LTS portable, Windows 11, CPU only, no
GUI, no CUDA/OptiX initialization. The scene armature was built
directly through bpy (plain armature, seven named bones); Rigify was
not used because generating a Rigify rig headless needs the addon
enabled plus a metarig generate step, and P1 only needs named bones.
In `--background` the event loop that fires `bpy.app.timers` never
runs, so the harness calls `start_bridge()` explicitly and drains the
executor queue itself on the main thread; socket threads still only
parse and enqueue. The client ran as a separate OS process over the
real socket. The shape-key step used a REAL voice artifact generated
live by the Kokoro + Rhubarb pipeline in `.venv-voice`
(17 cues, 68 samples, 3050 ms); the exact artifact is committed at
`acceptance/fixtures/acceptance.animus.json`.

- [x] Add-on module loads on Blender 4.5 LTS; `register()` correctly
      refuses to start the server under `--background`.
- [x] `inspect_rig` lists the armature bones and existing actions,
      mutating nothing.
- [x] Wave demo creates exactly one new Action and one new NLA strip
      (`ANIMUS_wave_take001`, 20 keys on 4 quaternion FCurves).
- [x] Existing Actions are unchanged.
- [x] A second run creates a separate take with new names
      (`ANIMUS_wave_take002`); the first take is untouched.
- [x] Invalid payloads are refused without mutation: a structural
      refusal (`frame_out_of_range`) answered by the socket thread and
      a scene refusal (`unknown_bone`) answered by the main thread.
- [x] Killing the client mid-request leaves no partial artifact
      (half a JSON line sent, connection closed; scene unchanged).
- [x] `apply_shape_keys` from the real voice artifact creates exactly
      one new Action and one new NLA strip on the face mesh's Key
      datablock (68 keys across 4 `key_blocks[...].value` FCurves);
      the armature's NLA is untouched.
- [x] FCurve writes behave under the Blender 4.4+ slotted Action
      model. Finding: the legacy `action.fcurves` API works and
      auto-creates exactly one legacy slot per Action. A pose take
      gets slot identifier `OBLegacy Slot` (target `OBJECT`); a
      shape-key take gets `KELegacy Slot` (target `KEY`). NLA strip
      assignment picks the slot automatically in both cases. No code
      change was needed for 4.5; a future move to the slotted API
      (`action.layers` / `slot` handles) is only required if one
      Action must ever drive multiple datablocks.

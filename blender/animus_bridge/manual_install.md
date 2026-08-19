# Animus Bridge: manual install into Blender 4.5 LTS

Status note, 2026-08-19: Blender is not installed on the build machine.
Real-Blender acceptance has NOT run. Every behavior below is covered by
the fake-bpy unit suite in `tests/animus_bridge/`. The first person with
Blender 4.5 LTS should run this checklist and record the result in the
Animus board log.

## What this add-on is

A localhost socket server inside Blender. It accepts newline-delimited
JSON requests on 127.0.0.1 and applies typed animation operations:
`inspect_rig`, `create_action`, `apply_pose_keys`, `push_to_nla`, and
the atomic `perform_take`. There is no code execution path. Socket
threads only parse and queue. A `bpy.app.timers` callback runs the
queued work on Blender's main thread. Every mutation lands in a new
Action and a new named NLA strip and returns the names it created.

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
an extension with `blender --command extension build` once real-Blender
testing starts. The legacy path above is enough for the P1 proof.

## Run the wave demo

1. Open a scene with an armature object. Note its object name.
2. Keep Blender in the foreground GUI. The server refuses to start in
   `--background` mode because queued work would never drain.
3. From any terminal:

   ```
   python blender/animus_bridge/examples/wave_client.py --object YourArmature
   ```

4. Expected result: one new Action named like `ANIMUS_wave_take001`,
   one new NLA track and strip with matching take names, and a JSON
   receipt printed by the client. Run it again and you get `take002`.
   Nothing existing is overwritten.

## Real-Blender acceptance checklist (still open)

- [ ] Add-on installs and enables on Blender 4.5 LTS.
- [ ] `inspect_rig` lists the armature bones and existing actions.
- [ ] Wave demo creates exactly one new Action and one new NLA strip.
- [ ] Existing Actions are unchanged.
- [ ] A second run creates a separate take with new names.
- [ ] Killing the client mid-request leaves no partial artifact.
- [ ] FCurve writes behave under the Blender 4.4+ slotted Action model.
      The bridge uses the legacy `action.fcurves` API, which the
      research brief flags for compatibility testing on 4.5 and 5.2.

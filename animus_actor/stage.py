"""Headless Blender stage for the actor loop. Runs INSIDE Blender.

Launched by the actor loop (or by hand):

    blender --background --factory-startup \
        --python animus_actor/stage.py -- \
        --port 8794 --profile src/animus/embodiment/kvrc-testrig.profile.json \
        --done-file <path> --report <path> [--deadline 300]

Builds the profile's test rig (plain armature with the profile's bones
plus a shape-keyed face mesh), starts the bridge on the given port, and
drains the executor queue on the main thread until the done-file
appears, exactly the drain contract the acceptance harness uses. Then
it writes a report JSON (scene snapshot plus keyframe counts for every
bridge Action) and exits. Exit 0 means the stage finished cleanly;
exit 1 means the deadline passed without a done-file.

CPU only by construction: --background, no render, no CUDA.
"""

import argparse
import json
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BLENDER_DIR = os.path.join(REPO, "blender")
if BLENDER_DIR not in sys.path:
    sys.path.insert(0, BLENDER_DIR)

import animus_bridge  # noqa: E402


def parse_args():
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(prog="animus_actor.stage")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--done-file", dest="done_file", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--deadline", type=float, default=300.0)
    return parser.parse_args(script_args)


def build_scene(profile):
    """The profile's rig as plain bpy data, same shape as the acceptance rig."""
    rig = profile["rig"]
    armature = bpy.data.armatures.new(f"{rig['object']}_rig")
    obj = bpy.data.objects.new(rig["object"], armature)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for index, name in enumerate(rig["bones"]):
        bone = armature.edit_bones.new(name)
        bone.head = (0.0, 0.0, 0.2 * index)
        bone.tail = (0.0, 0.2, 0.2 * index)
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = bpy.data.meshes.new(f"{rig['face_object']}_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)]
    )
    face = bpy.data.objects.new(rig["face_object"], mesh)
    bpy.context.scene.collection.objects.link(face)
    face.shape_key_add(name="Basis")
    for name in rig["shape_keys"]:
        face.shape_key_add(name=name)
    return obj, face


def strip_names(animation_data):
    if animation_data is None:
        return []
    return sorted(
        strip.name for track in animation_data.nla_tracks for strip in track.strips
    )


def snapshot(profile):
    rig = profile["rig"]
    obj = bpy.data.objects[rig["object"]]
    key = bpy.data.objects[rig["face_object"]].data.shape_keys
    return {
        "actions": sorted(action.name for action in bpy.data.actions),
        "pose_strips": strip_names(obj.animation_data),
        "face_strips": strip_names(key.animation_data),
    }


def action_keyframe_counts():
    return {
        action.name: sum(len(fc.keyframe_points) for fc in action.fcurves)
        for action in bpy.data.actions
    }


def drain():
    executor = animus_bridge._executor
    if executor is not None:
        executor.drain()


def main():
    args = parse_args()
    with open(args.profile, "r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)

    build_scene(profile)
    animus_bridge.register()  # refuses to start a server in background mode
    server = animus_bridge.start_bridge(port=args.port)

    deadline = time.monotonic() + args.deadline
    timed_out = False
    while not os.path.exists(args.done_file):
        drain()
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(0.01)
    # A request fully received right at done-file time still executes.
    for _ in range(20):
        drain()
        time.sleep(0.005)

    animus_bridge.stop_bridge()

    report = {
        "kind": "animus_actor_stage_report",
        "blender_version": bpy.app.version_string,
        "background": bpy.app.background,
        "port": server.port,
        "timed_out": timed_out,
        "scene": snapshot(profile),
        "action_keyframes": action_keyframe_counts(),
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[animus_actor.stage] report written to {args.report}")
    return 1 if timed_out else 0


if __name__ == "__main__":
    raise SystemExit(main())

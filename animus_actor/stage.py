"""Headless Blender stage for the actor loop. Runs INSIDE Blender.

Launched by the actor loop (or by hand):

    blender --background --factory-startup \
        --python animus_actor/stage.py -- \
        --port 8794 --profile src/animus/embodiment/kvrc-testrig.profile.json \
        --done-file <path> --report <path> [--deadline 300]

Builds the profile's stage (the humanoid test biped from
acceptance/biped.py plus a shape-keyed face mesh, or the profile's own
imported character), starts the bridge on the given port, and
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
ACCEPTANCE_DIR = os.path.join(BLENDER_DIR, "animus_bridge", "acceptance")
for _entry in (BLENDER_DIR, ACCEPTANCE_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import animus_bridge  # noqa: E402
import biped  # noqa: E402

from animus_actor.face_material import bind_face_screen  # noqa: E402


def parse_args():
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(prog="animus_actor.stage")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--done-file", dest="done_file", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--deadline", type=float, default=300.0)
    parser.add_argument(
        "--render-dir",
        dest="render_dir",
        default=None,
        help="after the take lands, render it (video plus stills) here",
    )
    parser.add_argument(
        "--render-audio",
        dest="render_audio",
        default=None,
        help="wav file muxed into the rendered video (AAC in the MP4)",
    )
    parser.add_argument(
        "--render-audio-start",
        dest="render_audio_start",
        type=int,
        default=None,
        help="frame the voice starts on (the speech beat's frame_start); "
        "without it the wav lands on the take's first frame, which is "
        "wrong for any plan whose speech beat starts after 0 ms",
    )
    parser.add_argument(
        "--render-size", dest="render_size", default="960x540", help="WxH"
    )
    parser.add_argument(
        "--render-engine",
        dest="render_engine",
        default="BLENDER_WORKBENCH",
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT"),
        help="Workbench is the fast headless default; EEVEE renders "
        "real materials (the emissive visor face needs it)",
    )
    parser.add_argument(
        "--face-frames",
        dest="face_frames",
        default=None,
        help="directory of face_*.png visor frames; the profile's "
        "stage.face_screen object gets them as an emissive image "
        "sequence before the render",
    )
    return parser.parse_args(script_args)


def _import_scene(profile):
    """Import the profile's real character and verify the rig it names.

    The imported file may carry its own baked clips (kvrc.glb ships 180
    Mixamo takes); those are stripped so the stage starts silent and the
    only motion in the scene is what the actor loop performs. The clip
    data stays in the source file; nothing here rewrites the asset.
    """
    stage_cfg = profile.get("stage") or {}
    spec = stage_cfg["import"]
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.join(REPO, path)
    if spec.get("format", "glb") not in ("glb", "gltf"):
        raise RuntimeError(f"unsupported stage import format: {spec}")
    bpy.ops.import_scene.gltf(filepath=path)

    rig = profile["rig"]
    obj = bpy.data.objects.get(rig["object"])
    if obj is None or obj.type != "ARMATURE":
        raise RuntimeError(
            f"stage import did not provide armature object '{rig['object']}'"
        )
    missing = [name for name in rig["bones"] if name not in obj.data.bones]
    if missing:
        raise RuntimeError(f"imported rig is missing profile bones: {missing}")

    if stage_cfg.get("strip_imported_animation", True):
        stripped = len(bpy.data.actions)
        for action in list(bpy.data.actions):
            bpy.data.actions.remove(action)
        for holder in bpy.data.objects:
            if holder.animation_data is not None:
                holder.animation_data_clear()
        print(f"[animus_actor.stage] stripped {stripped} imported actions")

    face = None
    if rig.get("face_object"):
        face = bpy.data.objects.get(rig["face_object"])
        if face is None:
            raise RuntimeError(
                f"stage import did not provide face object '{rig['face_object']}'"
            )
    return obj, face


def build_scene(profile):
    """The profile's rig: imported character, or the plain test rig."""
    if (profile.get("stage") or {}).get("import"):
        return _import_scene(profile)
    rig = profile["rig"]
    if biped.covers(rig["bones"]):
        # The developer test rig is a small humanoid figure; the
        # profile's bones are a subset of its skeleton.
        obj = biped.build_biped_armature(rig["object"])
    else:
        # Unknown bone names: fall back to a plain line-of-bones rig.
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

    face = None
    if rig.get("face_object"):
        mesh = bpy.data.meshes.new(f"{rig['face_object']}_mesh")
        mesh.from_pydata(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)]
        )
        face = bpy.data.objects.new(rig["face_object"], mesh)
        bpy.context.scene.collection.objects.link(face)
        face.shape_key_add(name="Basis")
        for name in rig.get("shape_keys") or []:
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
    face_strips = []
    if rig.get("face_object"):
        key = bpy.data.objects[rig["face_object"]].data.shape_keys
        if key is not None:
            face_strips = strip_names(key.animation_data)
    return {
        "actions": sorted(action.name for action in bpy.data.actions),
        "pose_strips": strip_names(obj.animation_data),
        "face_strips": face_strips,
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


# --- render support -------------------------------------------------------
# The test rig is a bare armature; armatures do not appear in final
# renders, so the render step dresses every bone with a small cuboid
# parented to it (parent_type BONE follows the animated pose). This is
# visualization only: it happens after the take has fully landed and
# touches no Action, strip, or shape key.


def _take_frame_range():
    """Frame span covered by the landed NLA strips (fallback 1..48)."""
    holders = [
        obj.animation_data for obj in bpy.data.objects if obj.animation_data
    ]
    holders += [
        key.animation_data for key in bpy.data.shape_keys if key.animation_data
    ]
    lo = hi = None
    for data in holders:
        for track in data.nla_tracks:
            for strip in track.strips:
                lo = strip.frame_start if lo is None else min(lo, strip.frame_start)
                hi = strip.frame_end if hi is None else max(hi, strip.frame_end)
    if lo is None:
        return 1, 48
    return int(lo), max(int(hi + 0.5), int(lo) + 1)


def _cuboid(name, size_x, size_y, size_z, center_y):
    """A box mesh centered on local X and Z, spanning center_y on Y."""
    x, z = size_x / 2.0, size_z / 2.0
    y0, y1 = center_y - size_y / 2.0, center_y + size_y / 2.0
    verts = [
        (-x, y0, -z), (x, y0, -z), (x, y1, -z), (-x, y1, -z),
        (-x, y0, z), (x, y0, z), (x, y1, z), (-x, y1, z),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    return mesh


def _dress_rig(profile):
    """One slim bone-shaped cuboid per armature bone, following the pose.

    Every bone of the rig is dressed (not only the profile's subset), so
    the test biped renders as a complete humanoid figure.
    """
    rig = profile["rig"]
    arm_obj = bpy.data.objects[rig["object"]]
    palette = [
        (0.85, 0.35, 0.20, 1.0), (0.95, 0.75, 0.20, 1.0),
        (0.30, 0.70, 0.90, 1.0), (0.40, 0.80, 0.40, 1.0),
        (0.80, 0.40, 0.80, 1.0), (0.95, 0.95, 0.95, 1.0),
        (0.55, 0.55, 0.95, 1.0),
    ]
    for index, bone in enumerate(arm_obj.data.bones):
        name = bone.name
        length = bone.length
        girth = max(0.03, min(0.09, length * 0.35))
        # A few bones get wider boxes so the figure reads at a glance:
        # the head as a head, the torso as a torso.
        girth = {
            "head": length * 0.7,
            "chest": 0.16,
            "spine": 0.13,
            "root": 0.17,
        }.get(name, girth)
        mesh = _cuboid(
            f"VIZ_{name}", girth, length * 0.9, girth, -length * 0.55
        )
        viz = bpy.data.objects.new(f"VIZ_{name}", mesh)
        bpy.context.scene.collection.objects.link(viz)
        viz.parent = arm_obj
        viz.parent_type = "BONE"
        viz.parent_bone = name
        viz.color = palette[index % len(palette)]


def _clear_startup_objects():
    """Drop the factory-startup Cube, Camera, and Light; keep the take."""
    for name in ("Cube", "Camera", "Light"):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.animation_data is None:
            bpy.data.objects.remove(obj, do_unlink=True)


def _add_camera_and_lights(render_cfg):
    scene = bpy.context.scene
    target = bpy.data.objects.new("RenderTarget", None)
    target.location = tuple(render_cfg.get("camera_target", (0.0, 0.0, 0.88)))
    scene.collection.objects.link(target)

    camera = bpy.data.objects.new("RenderCamera", bpy.data.cameras.new("RenderCamera"))
    camera.location = tuple(render_cfg.get("camera_location", (1.2, -3.2, 1.15)))
    camera.data.lens = float(render_cfg.get("lens", 40.0))
    scene.collection.objects.link(camera)
    track = camera.constraints.new(type="TRACK_TO")
    track.target = target
    scene.camera = camera

    key = bpy.data.objects.new("KeyLight", bpy.data.lights.new("KeyLight", "AREA"))
    key.data.energy = float(render_cfg.get("key_energy", 400.0))
    key.data.size = 3.0
    key.location = tuple(render_cfg.get("key_location", (2.0, -2.0, 2.5)))
    # Optional light colors (RGB, 0..1 each) so a profile can set the
    # scene's warmth per take; the default stays plain white.
    if render_cfg.get("key_color"):
        key.data.color = tuple(render_cfg["key_color"])
    scene.collection.objects.link(key)
    fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLight", "POINT"))
    fill.data.energy = float(render_cfg.get("fill_energy", 120.0))
    fill.location = tuple(render_cfg.get("fill_location", (-2.0, -1.0, 1.0)))
    if render_cfg.get("fill_color"):
        fill.data.color = tuple(render_cfg["fill_color"])
    scene.collection.objects.link(fill)


def render_take(args, profile):
    """Render the landed take: one H.264 MP4 plus four still PNGs."""
    os.makedirs(args.render_dir, exist_ok=True)
    scene = bpy.context.scene
    frame_start, frame_end = _take_frame_range()
    render_cfg = (profile.get("stage") or {}).get("render") or {}

    _clear_startup_objects()
    imported = bool((profile.get("stage") or {}).get("import"))
    if render_cfg.get("dress_bones", not imported):
        _dress_rig(profile)
    if not imported and profile["rig"].get("face_object"):
        # The synthetic shape-key proxy is dev scaffolding, not anatomy.
        proxy = bpy.data.objects.get(profile["rig"]["face_object"])
        if proxy is not None:
            proxy.hide_render = True
    for name in render_cfg.get("hide_objects", []):
        hidden = bpy.data.objects.get(name)
        if hidden is not None:
            hidden.hide_render = True
    _add_camera_and_lights(render_cfg)

    face_binding = None
    if args.face_frames:
        face_binding = bind_face_screen(
            bpy, profile, args.face_frames, frame_start=frame_start
        )
        print(f"[animus_actor.stage] visor face bound: {face_binding}")

    width, height = (int(part) for part in args.render_size.lower().split("x"))
    scene.render.engine = args.render_engine
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = int(profile.get("fps", 24))
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = render_cfg.get(
        "color_type", "TEXTURE" if imported else "OBJECT"
    )

    stills = {
        "start": frame_start,
        "wave-peak": min(frame_start + 15, frame_end),
        "mid-speech": (frame_start + frame_end) // 2,
        "end": frame_end,
    }
    scene.render.image_settings.file_format = "PNG"
    still_paths = {}
    for label, frame in stills.items():
        scene.frame_set(frame)
        path = os.path.join(args.render_dir, f"still-{label}-f{frame:03d}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        still_paths[label] = path

    audio = None
    if args.render_audio and os.path.exists(args.render_audio):
        editor = scene.sequence_editor_create()
        strips = getattr(editor, "sequences", None) or editor.strips
        audio_start = (
            args.render_audio_start
            if args.render_audio_start is not None
            else frame_start
        )
        strips.new_sound(
            "voice", filepath=args.render_audio, channel=1, frame_start=audio_start
        )
        audio = args.render_audio

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    if audio:
        scene.render.ffmpeg.audio_codec = "AAC"
    video_path = os.path.join(args.render_dir, "animus-take.mp4")
    scene.render.filepath = video_path
    scene.frame_set(frame_start)
    bpy.ops.render.render(animation=True)

    return {
        "engine": args.render_engine,
        "fps": scene.render.fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_end - frame_start + 1,
        "resolution": [width, height],
        "video": video_path if os.path.exists(video_path) else None,
        "audio": audio,
        "stills": still_paths,
        "face_screen": face_binding,
    }


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

    render_report = None
    if args.render_dir and not timed_out:
        render_report = render_take(args, profile)

    report = {
        "kind": "animus_actor_stage_report",
        "blender_version": bpy.app.version_string,
        "background": bpy.app.background,
        "port": server.port,
        "timed_out": timed_out,
        "scene": snapshot(profile),
        "action_keyframes": action_keyframe_counts(),
        "render": render_report,
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[animus_actor.stage] report written to {args.report}")
    return 1 if timed_out else 0


if __name__ == "__main__":
    raise SystemExit(main())

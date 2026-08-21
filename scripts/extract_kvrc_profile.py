"""Generate the real K-VRC embodiment profile from public/models/kvrc.glb.

Runs INSIDE headless Blender:

    blender --background --factory-startup \
        --python scripts/extract_kvrc_profile.py

The GLB carries a 27-bone Mixamo-style armature (KVRCArmature) and 180
baked clips. This script samples a handful of those clips into profile
gesture samples, so the numbers in the profile come from the licensed
asset's own motion, not from a model and not hand-invented (decision
A-008: the plan names things, the profile supplies every number).

Face and speech are MECHANICAL on this rig: the model has no shape keys
anywhere (verified at extraction time), so expressions are a Head-bone
pose (head tilt) and visemes drive the two ear antennae through the
speech bone mapping. No shape keys are faked.

The result is validated with the same validator the actor loop uses and
written to src/animus/embodiment/kvrc.profile.json.
"""

import json
import os
import sys

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from animus_actor.embodiment import validate_embodiment_profile  # noqa: E402

GLB = os.path.join(REPO, "public", "models", "kvrc.glb")
OUT = os.path.join(REPO, "src", "animus", "embodiment", "kvrc.profile.json")

RIG_OBJECT = "KVRCArmature"

# Upper-body bones sampled for gestures. Neck and Head stay out of the
# ordinary gestures so the gaze (Neck) and face (Head) channels own
# those data paths on the NLA stack; nod is the exception because a nod
# IS neck-and-head motion.
BODY_BONES = [
    "Hips", "Spine", "Spine1", "Spine2",
    "LeftShoulderPlate", "RightShoulderPlate",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]
HEAD_BONES = ["Neck", "Head"]
EAR_BONES = ["LeftEar", "RightEar"]

# gesture name -> (source clip, frames to keep, sampling stride, bones)
CLIPS = {
    "wave": ("waving_1", 27, 3, BODY_BONES),
    "idle": ("breathing_idle", 48, 6, BODY_BONES),
    "talk": ("talking", 72, 6, BODY_BONES),
    "nod": ("head_nod_yes", 47, 4, BODY_BONES + HEAD_BONES),
    "point": ("angry_point", 60, 5, BODY_BONES),
    # Scene-work harvest (fable-scenes): more of the GLB's own baked
    # clips, same extraction rules, so stage directions can name them.
    "think": ("thinking", 100, 6, BODY_BONES),
    "sad": ("sad_idle", 96, 6, BODY_BONES),
    "cocky": ("being_cocky", 56, 4, BODY_BONES),
    "focus": ("focus", 96, 6, BODY_BONES),
    "dance": ("silly_dancing", 180, 5, BODY_BONES),
}


def _round6(value):
    return round(value * 1e6) / 1e6


def _fcurves_for(action, bone):
    curves = {"rotation_quaternion": {}, "location": {}}
    for fcurve in action.fcurves:
        for prop in curves:
            if fcurve.data_path == f'pose.bones["{bone}"].{prop}':
                curves[prop][fcurve.array_index] = fcurve
    return curves


def sample_clip(action, frames, stride, bones):
    """Evaluate the baked fcurves at a fixed stride, pose space."""
    picked = sorted(set(list(range(1, frames + 1, stride)) + [frames]))
    samples = []
    for bone in bones:
        curves = _fcurves_for(action, bone)
        quat = curves["rotation_quaternion"]
        loc = curves["location"]
        for frame in picked:
            sample = {"bone": bone, "frame": frame}
            if len(quat) == 4:
                sample["rotation_quaternion"] = [
                    _round6(quat[i].evaluate(frame)) for i in range(4)
                ]
            if bone == "Hips" and len(loc) == 3:
                sample["location"] = [
                    _round6(loc[i].evaluate(frame)) for i in range(3)
                ]
            if "rotation_quaternion" in sample or "location" in sample:
                samples.append(sample)
    return samples


def main():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=GLB)

    rig = bpy.data.objects[RIG_OBJECT]
    assert rig.type == "ARMATURE"
    shape_keyed = [
        obj.name
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.data.shape_keys is not None
    ]
    assert not shape_keyed, f"model unexpectedly has shape keys: {shape_keyed}"

    gestures = {}
    for name, (clip, frames, stride, bones) in CLIPS.items():
        action = bpy.data.actions.get(clip)
        assert action is not None, f"clip '{clip}' not found in {GLB}"
        gestures[name] = {
            "name_hint": name if name != "wave" else "wave",
            "frames": frames,
            # Half a second back to the clip's entry pose after it
            # ends, so gestures release instead of freezing on their
            # last frame (embodiment mapper blend-out).
            "blend_out_frames": 12,
            "source_clip": clip,
            "samples": sample_clip(action, frames, stride, bones),
        }

    profile = {
        "kind": "animus_embodiment_profile",
        "schema_version": "0.1",
        "profile": "kvrc",
        "profile_version": "0.1.0",
        "source": {
            "authored_by": "fable-embody",
            "license": "project-internal fan asset: public/models/kvrc.glb "
            "(K-VRC robot from the webapp, Mixamo-baked clips); gesture "
            "samples extracted from those baked clips by "
            "scripts/extract_kvrc_profile.py",
            "rig": "KVRCArmature, 27 bones, rigid per-part armature "
            "skinning, no shape keys",
            "note": "face and speech are mechanical: Head-bone tilt for "
            "mood, viseme-driven ear motion for speech, because the "
            "model has no morph targets",
        },
        "fps": 24,
        "rig": {
            "object": RIG_OBJECT,
            "bones": sorted(set(BODY_BONES + HEAD_BONES + EAR_BONES)),
        },
        "default_gesture": "idle",
        "body_action_gestures": {
            "idle": "idle",
            "wait": "idle",
            "walk_to": "idle",
            "turn_to": "idle",
            "interact": "point",
        },
        "gestures": gestures,
        "gaze": {
            "bone": "Neck",
            "name_hint": "gaze",
            "neutral": [1.0, 0.0, 0.0, 0.0],
            "ease_frames": 6,
            "default_target": "camera",
            "targets": {
                "camera": [0.997, 0.076, 0.0, 0.0],
                "viewer": [0.997, 0.076, 0.0, 0.0],
                "left": [0.976, 0.0, 0.216, 0.0],
                "right": [0.976, 0.0, -0.216, 0.0],
                "up": [0.991, -0.131, 0.0, 0.0],
                "down": [0.991, 0.131, 0.0, 0.0],
            },
        },
        "expressions": {
            "mode": "pose",
            "name_hint": "face",
            "bone": "Head",
            "neutral": [1.0, 0.0, 0.0, 0.0],
            "default_expression": "neutral_idle",
            "presets": {
                "neutral_idle": [1.0, 0.0, 0.0, 0.0],
                "warm_amused": [0.9945, 0.0348, 0.0, 0.0958],
                "concerned": [0.9945, 0.1045, 0.0, 0.0],
                "surprised": [0.9962, -0.0872, 0.0, 0.0],
            },
        },
        "speech": {
            "mode": "bone",
            "object": RIG_OBJECT,
            "name_hint": "animus_speech",
            "driver": "mouth_open",
            "bones": [
                {
                    "bone": "LeftEar",
                    "neutral": [1.0, 0.0, 0.0, 0.0],
                    "peak": [0.9877, 0.1564, 0.0, 0.0],
                },
                {
                    "bone": "RightEar",
                    "neutral": [1.0, 0.0, 0.0, 0.0],
                    "peak": [0.9877, 0.1564, 0.0, 0.0],
                },
            ],
            "voice": "af_heart",
            "lang": "a",
            "speed": 1.0,
            "seed": 0,
        },
        "stage": {
            "import": {"format": "glb", "path": "public/models/kvrc.glb"},
            "strip_imported_animation": True,
            "render": {
                "camera_location": [1.15, -3.65, 1.4],
                "camera_target": [0.0, 0.0, 1.0],
                "lens": 40.0,
                "key_energy": 900.0,
                "fill_energy": 260.0,
                "key_location": [2.0, -2.5, 2.8],
                "fill_location": [-2.2, -1.4, 1.2],
                "color_type": "TEXTURE",
                "dress_bones": False,
                "hide_objects": [],
            },
            # The webapp's LED face renders onto this mesh as an
            # emissive image sequence (see animus_actor/face_frames.py
            # and face_material.py). Added by fable-face on the shipped
            # profile; kept here so regeneration does not drop it.
            "face_screen": {"object": "screen", "emission_strength": 3.5},
        },
    }

    checked = validate_embodiment_profile(profile)
    if not checked["ok"]:
        print("PROFILE INVALID:")
        for error in checked["errors"]:
            print("  " + error)
        return 1
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=1)
        handle.write("\n")
    total = sum(len(g["samples"]) for g in gestures.values())
    print(f"[extract_kvrc_profile] wrote {OUT} ({total} gesture samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

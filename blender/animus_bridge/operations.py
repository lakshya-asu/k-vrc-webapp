"""Typed bridge operations.

Every function here runs on Blender's main thread only. The executor is
the single caller. Each mutating operation validates every scene fact it
needs before it touches any datablock. On failure it raises a typed
Refusal and mutates nothing. On success it returns a receipt with the
names it created.

Generated Actions carry the custom property MARKER so the bridge can
tell its own takes apart from the user's work. apply_pose_keys refuses
to write into an Action that lacks the marker. Nothing here ever
overwrites an existing Action or NLA strip. Every take gets fresh names.
"""

import bpy

from .protocol import (
    INTERNAL_ERROR,
    NO_SHAPE_KEYS,
    NOT_AN_ARMATURE,
    PROTECTED_ACTION,
    Refusal,
    UNKNOWN_ACTION,
    UNKNOWN_BONE,
    UNKNOWN_OBJECT,
    UNKNOWN_SHAPE_KEY,
)

MARKER = "animus_bridge"
ACTION_PREFIX = "ANIMUS"


def _get_armature_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise Refusal(UNKNOWN_OBJECT, f"no object named '{name}' in the scene data")
    if getattr(obj, "type", None) != "ARMATURE":
        raise Refusal(NOT_AN_ARMATURE, f"object '{name}' is not an armature")
    return obj


def _bone_names(obj):
    return set(obj.data.bones.keys())


def _require_bones(obj, samples):
    known = _bone_names(obj)
    for sample in samples:
        if sample["bone"] not in known:
            raise Refusal(
                UNKNOWN_BONE,
                f"armature '{obj.name}' has no bone '{sample['bone']}'",
            )


def _get_shape_keyed_object(name):
    """Resolve an object whose data carries shape keys.

    Returns (object, key datablock). Shape-key FCurves and their NLA
    takes live on the Key datablock (object.data.shape_keys), not on
    the object itself, so the Key is what the caller animates.
    """
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise Refusal(UNKNOWN_OBJECT, f"no object named '{name}' in the scene data")
    key = getattr(getattr(obj, "data", None), "shape_keys", None)
    if key is None:
        raise Refusal(NO_SHAPE_KEYS, f"object '{name}' has no shape keys")
    return obj, key


def _require_shape_keys(obj, key, samples):
    known = set(key.key_blocks.keys())
    for sample in samples:
        if sample["shape_key"] not in known:
            raise Refusal(
                UNKNOWN_SHAPE_KEY,
                f"object '{obj.name}' has no shape key '{sample['shape_key']}'",
            )


def _get_marked_action(name):
    action = bpy.data.actions.get(name)
    if action is None:
        raise Refusal(UNKNOWN_ACTION, f"no action named '{name}'")
    return action


def _unique_name(base, taken):
    number = 1
    while True:
        candidate = f"{base}_take{number:03d}"
        if candidate not in taken:
            return candidate
        number += 1


def _unique_action_name(hint):
    return _unique_name(f"{ACTION_PREFIX}_{hint}", set(bpy.data.actions.keys()))


def _unique_nla_names(obj, hint):
    taken = set()
    animation_data = getattr(obj, "animation_data", None)
    if animation_data is not None:
        for track in animation_data.nla_tracks:
            taken.add(track.name)
            for strip in track.strips:
                taken.add(strip.name)
    name = _unique_name(f"{ACTION_PREFIX}_{hint}", taken)
    return f"{name}_track", name


def _new_marked_action(hint):
    action = bpy.data.actions.new(_unique_action_name(hint))
    action[MARKER] = True
    if hasattr(action, "use_fake_user"):
        action.use_fake_user = True
    return action


def _write_pose_keys(action, samples):
    """Write validated samples into the action's FCurves.

    Uses the data API directly. The target object's active Action is
    never touched.
    """
    key_count = 0
    bones = set()
    for sample in samples:
        bone = sample["bone"]
        bones.add(bone)
        channels = []
        if "location" in sample:
            channels.append((f'pose.bones["{bone}"].location', sample["location"]))
        if "rotation_quaternion" in sample:
            channels.append(
                (
                    f'pose.bones["{bone}"].rotation_quaternion',
                    sample["rotation_quaternion"],
                )
            )
        for data_path, values in channels:
            for index, value in enumerate(values):
                fcurve = action.fcurves.find(data_path, index=index)
                if fcurve is None:
                    fcurve = action.fcurves.new(data_path, index=index)
                fcurve.keyframe_points.insert(sample["frame"], value)
                key_count += 1
    return key_count, sorted(bones)


def _new_strip(obj, action, hint, frame_start):
    # 'obj' is any animatable ID datablock: an armature object for pose
    # takes, or the shape-key Key datablock for face takes.
    animation_data = getattr(obj, "animation_data", None)
    if animation_data is None:
        animation_data = obj.animation_data_create()
    track_name, strip_name = _unique_nla_names(obj, hint)
    track = animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(strip_name, int(frame_start), action)
    return track, strip


def inspect_rig(params):
    """Read-only. Reports the armature, its bones, all actions, and NLA."""
    obj = _get_armature_object(params["object"])
    tracks = []
    animation_data = getattr(obj, "animation_data", None)
    if animation_data is not None:
        for track in animation_data.nla_tracks:
            tracks.append(
                {
                    "name": track.name,
                    "strips": [
                        {
                            "name": strip.name,
                            "action": strip.action.name if strip.action else None,
                            "frame_start": strip.frame_start,
                            "frame_end": strip.frame_end,
                        }
                        for strip in track.strips
                    ],
                }
            )
    return {
        "object": obj.name,
        "armature": obj.data.name,
        "bones": sorted(_bone_names(obj)),
        "actions": sorted(bpy.data.actions.keys()),
        "nla_tracks": tracks,
    }


def create_action(params):
    action = _new_marked_action(params["name_hint"])
    return {"action": action.name}


def apply_pose_keys(params):
    obj = _get_armature_object(params["object"])
    action = _get_marked_action(params["action"])
    if not action.get(MARKER):
        raise Refusal(
            PROTECTED_ACTION,
            f"action '{action.name}' was not created by the bridge; refusing to edit it",
        )
    _require_bones(obj, params["samples"])
    key_count, bones = _write_pose_keys(action, params["samples"])
    return {
        "action": action.name,
        "bones": bones,
        "sample_count": len(params["samples"]),
        "key_count": key_count,
        "frame_start": params["frame_start"],
        "frame_end": params["frame_end"],
    }


def push_to_nla(params):
    obj = _get_armature_object(params["object"])
    action = _get_marked_action(params["action"])
    track, strip = _new_strip(obj, action, params["name_hint"], params["frame_start"])
    return {
        "action": action.name,
        "track": track.name,
        "strip": strip.name,
        "frame_start": strip.frame_start,
        "frame_end": strip.frame_end,
    }


def perform_take(params):
    """One atomic take: new Action, validated pose keys, new NLA strip.

    All scene checks run before the first mutation. A repeated identical
    request produces new names. It never overwrites an earlier take.
    """
    obj = _get_armature_object(params["object"])
    _require_bones(obj, params["samples"])

    action = None
    try:
        action = _new_marked_action(params["name_hint"])
        key_count, bones = _write_pose_keys(action, params["samples"])
        track, strip = _new_strip(
            obj, action, params["name_hint"], params["frame_start"]
        )
    except Refusal:
        if action is not None:
            bpy.data.actions.remove(action)
        raise
    except Exception as error:
        if action is not None:
            bpy.data.actions.remove(action)
        raise Refusal(INTERNAL_ERROR, f"take failed and was rolled back: {error}")

    return {
        "action": action.name,
        "track": track.name,
        "strip": strip.name,
        "bones": bones,
        "sample_count": len(params["samples"]),
        "key_count": key_count,
        "frame_start": params["frame_start"],
        "frame_end": params["frame_end"],
    }


def _write_shape_keys(action, samples):
    """Write validated shape-key samples into the action's FCurves.

    One FCurve per shape key, data path key_blocks["name"].value, the
    same legacy action.fcurves API apply_pose_keys uses for bones.
    """
    key_count = 0
    shape_keys = set()
    for sample in samples:
        name = sample["shape_key"]
        shape_keys.add(name)
        data_path = f'key_blocks["{name}"].value'
        fcurve = action.fcurves.find(data_path, index=0)
        if fcurve is None:
            fcurve = action.fcurves.new(data_path, index=0)
        fcurve.keyframe_points.insert(sample["frame"], sample["weight"])
        key_count += 1
    return key_count, sorted(shape_keys)


def apply_shape_keys(params):
    """One atomic shape-key take: new Action, weight keys, new NLA strip.

    The voice pipeline's viseme take maps onto this directly: object,
    name_hint, frame_start, frame_end, and the artifact's samples list
    (extra at_ms/viseme fields are accepted and dropped by validation).
    All scene checks run before the first mutation. The Action and the
    NLA strip land on the Key datablock of the named object. A repeated
    identical request produces new names; nothing is overwritten.
    """
    obj, key = _get_shape_keyed_object(params["object"])
    _require_shape_keys(obj, key, params["samples"])

    action = None
    try:
        action = _new_marked_action(params["name_hint"])
        key_count, shape_keys = _write_shape_keys(action, params["samples"])
        track, strip = _new_strip(
            key, action, params["name_hint"], params["frame_start"]
        )
    except Refusal:
        if action is not None:
            bpy.data.actions.remove(action)
        raise
    except Exception as error:
        if action is not None:
            bpy.data.actions.remove(action)
        raise Refusal(INTERNAL_ERROR, f"take failed and was rolled back: {error}")

    return {
        "action": action.name,
        "track": track.name,
        "strip": strip.name,
        "shape_keys": shape_keys,
        "sample_count": len(params["samples"]),
        "key_count": key_count,
        "frame_start": params["frame_start"],
        "frame_end": params["frame_end"],
    }


HANDLERS = {
    "inspect_rig": inspect_rig,
    "create_action": create_action,
    "apply_pose_keys": apply_pose_keys,
    "apply_shape_keys": apply_shape_keys,
    "push_to_nla": push_to_nla,
    "perform_take": perform_take,
}

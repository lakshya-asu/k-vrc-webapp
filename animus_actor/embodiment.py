"""Embodiment mapping, Python port of src/animus/embodiment.js.

The only place semantic actor intent becomes numeric motion. Every
quaternion, frame, and shape-key weight the bridge receives comes from a
validated profile or from the voice pipeline's viseme artifact; the plan
never supplies numbers (decisions A-005, A-006, A-008).

Number parity with the JS mapper is deliberate: rounding uses the
JS Math.round rule (half away from zero toward positive infinity), so a
plan mapped here produces the same requests byte for byte as the Node
director. The cross-language fixture test enforces this.
"""

import json
import math
import re

PROFILE_KIND = "animus_embodiment_profile"
PROFILE_SCHEMA_VERSION = "0.1"

# Mirrors blender/animus_bridge/protocol.py limits so a mapped request
# can never be structurally refused by the bridge.
MAX_FRAME = 100000
MAX_FRAME_SPAN = 10000
MAX_SAMPLES = 2000
_NAME_HINT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,59}$")
_HINT_CLEAN_RE = re.compile(r"[^A-Za-z0-9_.\-]")
_HINT_START_RE = re.compile(r"^[A-Za-z0-9_]")


def _is_object(value):
    return isinstance(value, dict)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_frame(value):
    return _is_integer(value) and 1 <= value <= MAX_FRAME


def _is_quaternion(value):
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_is_finite_number(item) for item in value)
    )


def _js_round(value):
    """JS Math.round: half rounds toward positive infinity."""
    return math.floor(value + 0.5)


def _round6(value):
    return _js_round(value * 1e6) / 1e6


def sanitize_name_hint(text, fallback="take"):
    hint = _HINT_CLEAN_RE.sub("_", str(text if text is not None else ""))[:60]
    if hint and not _HINT_START_RE.match(hint):
        hint = (f"x{hint}")[:60]
    if not hint or not _NAME_HINT_RE.match(hint):
        return fallback
    return hint


def _validate_gesture_samples(gesture, name, bones, errors):
    samples = gesture.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append(f"gestures.{name}.samples must be a non-empty list")
        return
    if len(samples) > MAX_SAMPLES:
        errors.append(f"gestures.{name}.samples exceeds {MAX_SAMPLES} items")
    for index, sample in enumerate(samples):
        path = f"gestures.{name}.samples[{index}]"
        if not _is_object(sample):
            errors.append(f"{path} must be an object")
            continue
        if sample.get("bone") not in bones:
            errors.append(f"{path}.bone '{sample.get('bone')}' is not a rig bone")
        frame = sample.get("frame")
        if not _is_frame(frame) or frame > gesture["frames"]:
            errors.append(
                f"{path}.frame must be an integer from 1 to gestures.{name}.frames"
            )
        has_rotation = "rotation_quaternion" in sample
        has_location = "location" in sample
        if not has_rotation and not has_location:
            errors.append(f"{path} needs rotation_quaternion or location")
        if has_rotation and not _is_quaternion(sample["rotation_quaternion"]):
            errors.append(f"{path}.rotation_quaternion must be 4 finite numbers")
        if has_location and not (
            isinstance(sample["location"], list)
            and len(sample["location"]) == 3
            and all(_is_finite_number(item) for item in sample["location"])
        ):
            errors.append(f"{path}.location must be 3 finite numbers")


def _channel_mode(section, key, allowed, default, errors):
    """Read an optional mode switch ('shape_keys' by default)."""
    if not _is_object(section) or "mode" not in section:
        return default
    mode = section.get("mode")
    if mode not in allowed:
        errors.append(f"{key}.mode must be one of {sorted(allowed)}")
        return default
    return mode


def validate_embodiment_profile(profile):
    """Full structural validation, port of validateEmbodimentProfile."""
    errors = []
    if not _is_object(profile):
        return {"ok": False, "errors": ["profile must be an object"], "value": None}
    if profile.get("kind") != PROFILE_KIND:
        errors.append(f"profile.kind must be '{PROFILE_KIND}'")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        errors.append(f"profile.schema_version must be '{PROFILE_SCHEMA_VERSION}'")
    if not isinstance(profile.get("profile"), str) or not profile.get("profile"):
        errors.append("profile.profile must name the profile")
    if not isinstance(profile.get("profile_version"), str) or not profile.get(
        "profile_version"
    ):
        errors.append("profile.profile_version is required")
    source = profile.get("source")
    if not _is_object(source) or not isinstance(source.get("license"), str):
        errors.append("profile.source.license is required")
    fps = profile.get("fps")
    if not _is_integer(fps) or fps < 1 or fps > 240:
        errors.append("profile.fps must be an integer from 1 to 240")

    # Mechanical rigs (rigid robots with no morph targets) may map the
    # face channel to a bone pose and speech to viseme-driven bone
    # motion. Shape-key structures are then optional; nothing is faked.
    face_mode = _channel_mode(
        profile.get("expressions"), "expressions", ("shape_keys", "pose"),
        "shape_keys", errors,
    )
    speech_mode = _channel_mode(
        profile.get("speech"), "speech", ("shape_keys", "bone"),
        "shape_keys", errors,
    )
    needs_shape_keys = face_mode == "shape_keys" or speech_mode == "shape_keys"

    rig = profile.get("rig")
    bones = set()
    shape_keys = set()
    if not _is_object(rig):
        errors.append("profile.rig must be an object")
    else:
        if not isinstance(rig.get("object"), str) or not rig.get("object"):
            errors.append("rig.object is required")
        rig_bones = rig.get("bones")
        if (
            not isinstance(rig_bones, list)
            or not rig_bones
            or any(not isinstance(bone, str) or not bone for bone in rig_bones)
        ):
            errors.append("rig.bones must be a non-empty list of bone names")
        else:
            bones = set(rig_bones)
        if needs_shape_keys or rig.get("face_object") is not None:
            if not isinstance(rig.get("face_object"), str) or not rig.get(
                "face_object"
            ):
                errors.append("rig.face_object is required")
        rig_keys = rig.get("shape_keys")
        if needs_shape_keys or rig_keys is not None:
            if (
                not isinstance(rig_keys, list)
                or not rig_keys
                or any(not isinstance(key, str) or not key for key in rig_keys)
            ):
                errors.append(
                    "rig.shape_keys must be a non-empty list of shape key names"
                )
            else:
                shape_keys = set(rig_keys)

    gestures = profile.get("gestures")
    if not _is_object(gestures) or not gestures:
        errors.append("profile.gestures must be a non-empty object")
    else:
        for name, gesture in gestures.items():
            if not _is_object(gesture):
                errors.append(f"gestures.{name} must be an object")
                continue
            if not _NAME_HINT_RE.match(str(gesture.get("name_hint") or "")):
                errors.append(
                    f"gestures.{name}.name_hint must be a valid bridge name hint"
                )
            frames = gesture.get("frames")
            if not _is_frame(frames) or frames > MAX_FRAME_SPAN:
                errors.append(
                    f"gestures.{name}.frames must be an integer from 1 to "
                    f"{MAX_FRAME_SPAN}"
                )
            else:
                _validate_gesture_samples(gesture, name, bones, errors)
        default_gesture = profile.get("default_gesture")
        if not isinstance(default_gesture, str) or default_gesture not in gestures:
            errors.append("profile.default_gesture must name a defined gesture")
        action_gestures = profile.get("body_action_gestures")
        if not _is_object(action_gestures):
            errors.append("profile.body_action_gestures must be an object")
        else:
            for action, name in action_gestures.items():
                if name not in gestures:
                    errors.append(
                        f"body_action_gestures.{action} names unknown gesture '{name}'"
                    )

    gaze = profile.get("gaze")
    if not _is_object(gaze):
        errors.append("profile.gaze must be an object")
    else:
        if gaze.get("bone") not in bones:
            errors.append(f"gaze.bone '{gaze.get('bone')}' is not a rig bone")
        if not _NAME_HINT_RE.match(str(gaze.get("name_hint") or "")):
            errors.append("gaze.name_hint must be a valid bridge name hint")
        if not _is_quaternion(gaze.get("neutral")):
            errors.append("gaze.neutral must be 4 finite numbers")
        ease = gaze.get("ease_frames")
        if not _is_integer(ease) or ease < 1 or ease > 240:
            errors.append("gaze.ease_frames must be an integer from 1 to 240")
        targets = gaze.get("targets")
        if not _is_object(targets) or not targets:
            errors.append("gaze.targets must be a non-empty object")
        else:
            for name, quat in targets.items():
                if not _is_quaternion(quat):
                    errors.append(f"gaze.targets.{name} must be 4 finite numbers")
            if gaze.get("default_target") not in targets:
                errors.append("gaze.default_target must name a defined target")

    expressions = profile.get("expressions")
    if (
        not _is_object(expressions)
        or not _is_object(expressions.get("presets"))
        or not expressions.get("presets")
    ):
        errors.append("profile.expressions.presets must be a non-empty object")
    else:
        if not _NAME_HINT_RE.match(str(expressions.get("name_hint") or "")):
            errors.append("expressions.name_hint must be a valid bridge name hint")
        if face_mode == "pose":
            if expressions.get("bone") not in bones:
                errors.append(
                    f"expressions.bone '{expressions.get('bone')}' is not a rig bone"
                )
            if not _is_quaternion(expressions.get("neutral")):
                errors.append("expressions.neutral must be 4 finite numbers")
            for name, preset in expressions["presets"].items():
                if not _is_quaternion(preset):
                    errors.append(
                        f"expressions.presets.{name} must be 4 finite numbers "
                        "in pose mode"
                    )
        else:
            for name, preset in expressions["presets"].items():
                if not _is_object(preset) or not preset:
                    errors.append(
                        f"expressions.presets.{name} must be a non-empty object"
                    )
                    continue
                for key, weight in preset.items():
                    if key not in shape_keys:
                        errors.append(
                            f"expressions.presets.{name}.{key} is not a rig shape key"
                        )
                    if not _is_finite_number(weight) or weight < 0 or weight > 1:
                        errors.append(
                            f"expressions.presets.{name}.{key} must be a number "
                            "from 0 to 1"
                        )
        if expressions.get("default_expression") not in expressions["presets"]:
            errors.append("expressions.default_expression must name a defined preset")

    speech = profile.get("speech")
    if not _is_object(speech):
        errors.append("profile.speech must be an object")
    else:
        if speech_mode == "bone":
            if _is_object(rig) and speech.get("object") != rig.get("object"):
                errors.append("speech.object must equal rig.object in bone mode")
            speech_bones = speech.get("bones")
            if not isinstance(speech_bones, list) or not speech_bones:
                errors.append(
                    "speech.bones must be a non-empty list in bone mode"
                )
            else:
                for index, entry in enumerate(speech_bones):
                    path = f"speech.bones[{index}]"
                    if not _is_object(entry):
                        errors.append(f"{path} must be an object")
                        continue
                    if entry.get("bone") not in bones:
                        errors.append(
                            f"{path}.bone '{entry.get('bone')}' is not a rig bone"
                        )
                    if not _is_quaternion(entry.get("neutral")):
                        errors.append(f"{path}.neutral must be 4 finite numbers")
                    if not _is_quaternion(entry.get("peak")):
                        errors.append(f"{path}.peak must be 4 finite numbers")
            if "driver" in speech and (
                not isinstance(speech.get("driver"), str) or not speech.get("driver")
            ):
                errors.append("speech.driver must be a viseme weight name")
        elif _is_object(rig) and speech.get("object") != rig.get("face_object"):
            errors.append("speech.object must equal rig.face_object")
        if not _NAME_HINT_RE.match(str(speech.get("name_hint") or "")):
            errors.append("speech.name_hint must be a valid bridge name hint")
        if not isinstance(speech.get("voice"), str) or not speech.get("voice"):
            errors.append("speech.voice is required")

    if errors:
        return {"ok": False, "errors": errors, "value": None}
    return {"ok": True, "errors": [], "value": profile}


def load_profile(path):
    """Load and validate a profile file. Raises ValueError on refusal."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    checked = validate_embodiment_profile(raw)
    if not checked["ok"]:
        raise ValueError(
            "embodiment profile is invalid:\n" + "\n".join(checked["errors"])
        )
    return checked["value"]


def _ms_to_frame(at_ms, fps):
    return 1 + _js_round((at_ms * fps) / 1000)


def _duration_to_frames(duration_ms, fps):
    return max(2, _js_round((duration_ms * fps) / 1000))


def _nlerp(from_quat, to_quat, t):
    mixed = [
        value + (to_quat[index] - value) * t for index, value in enumerate(from_quat)
    ]
    length = math.hypot(*mixed)
    if length == 0:
        return [1, 0, 0, 0]
    return [_round6(value / length) for value in mixed]


def _request_id(beat_id, channel):
    return f"act-{sanitize_name_hint(beat_id, 'beat')}-{channel}"[:120]


def _resolve_gesture_name(body, profile):
    if body["action"] == "gesture":
        if body.get("gesture") in profile["gestures"]:
            return body["gesture"]
        return profile["default_gesture"]
    return profile["body_action_gestures"].get(
        body["action"], profile["default_gesture"]
    )


def _map_body_beat(beat, profile):
    gesture_name = _resolve_gesture_name(beat["body"], profile)
    gesture = profile["gestures"][gesture_name]
    base = _ms_to_frame(beat["at_ms"], profile["fps"])
    offset = base - 1
    samples = []
    for sample in gesture["samples"]:
        moved = {"bone": sample["bone"], "frame": sample["frame"] + offset}
        if "rotation_quaternion" in sample:
            moved["rotation_quaternion"] = sample["rotation_quaternion"]
        if "location" in sample:
            moved["location"] = sample["location"]
        samples.append(moved)
    return {
        "beat_id": beat["id"],
        "channel": "body",
        "gesture": gesture_name,
        "request": {
            "id": _request_id(beat["id"], "body"),
            "op": "perform_take",
            "params": {
                "object": profile["rig"]["object"],
                "name_hint": sanitize_name_hint(gesture["name_hint"], "gesture"),
                "frame_start": base,
                "frame_end": offset + gesture["frames"],
                "samples": samples,
            },
        },
    }


def _map_gaze_beat(beat, profile):
    gaze = profile["gaze"]
    target_name = (
        beat["gaze"]["target"]
        if beat["gaze"]["target"] in gaze["targets"]
        else gaze["default_target"]
    )
    intensity = beat["gaze"].get("intensity", 1)
    aimed = _nlerp(gaze["neutral"], gaze["targets"][target_name], intensity)
    base = _ms_to_frame(beat["at_ms"], profile["fps"])
    dur_frames = _duration_to_frames(beat["duration_ms"], profile["fps"])
    end = base + dur_frames - 1
    ease = min(gaze["ease_frames"], dur_frames - 1)
    frames = {}
    frames[base] = [_round6(value) for value in gaze["neutral"]]
    frames[min(base + ease, end)] = aimed
    frames[end] = aimed
    samples = [
        {"bone": gaze["bone"], "frame": frame, "rotation_quaternion": quat}
        for frame, quat in frames.items()
    ]
    return {
        "beat_id": beat["id"],
        "channel": "gaze",
        "target": target_name,
        "request": {
            "id": _request_id(beat["id"], "gaze"),
            "op": "perform_take",
            "params": {
                "object": profile["rig"]["object"],
                "name_hint": sanitize_name_hint(
                    f"{gaze['name_hint']}_{target_name}", "gaze"
                ),
                "frame_start": base,
                "frame_end": end,
                "samples": samples,
            },
        },
    }


def _map_face_beat(beat, profile):
    expressions = profile["expressions"]
    name = (
        beat["face"]["expression"]
        if beat["face"]["expression"] in expressions["presets"]
        else expressions["default_expression"]
    )
    preset = expressions["presets"][name]
    intensity = beat["face"].get("intensity", 1)
    base = _ms_to_frame(beat["at_ms"], profile["fps"])
    dur_frames = _duration_to_frames(beat["duration_ms"], profile["fps"])
    end = base + dur_frames - 1
    ramp = max(1, min(4, dur_frames // 3))

    # Ramp in, hold, ramp out; a beat too short for the ramp collapses to
    # peak-at-start, zero-at-end. Later writes win on frame collisions.
    frame_weights = {}
    if end - base >= 2 * ramp + 1:
        frame_weights[base] = 0
        frame_weights[base + ramp] = 1
        frame_weights[end - ramp] = 1
        frame_weights[end] = 0
    else:
        frame_weights[base] = 1
        frame_weights[end] = 0

    name_hint = sanitize_name_hint(f"{expressions['name_hint']}_{name}", "face")

    if expressions.get("mode") == "pose":
        # Mechanical face: the mood is a bone pose (head tilt), eased by
        # the same envelope shape-key expressions use.
        neutral = expressions["neutral"]
        samples = [
            {
                "bone": expressions["bone"],
                "frame": frame,
                "rotation_quaternion": _nlerp(
                    neutral, preset, min(1, max(0, intensity * envelope))
                ),
            }
            for frame, envelope in frame_weights.items()
        ]
        return {
            "beat_id": beat["id"],
            "channel": "face",
            "expression": name,
            "request": {
                "id": _request_id(beat["id"], "face"),
                "op": "perform_take",
                "params": {
                    "object": profile["rig"]["object"],
                    "name_hint": name_hint,
                    "frame_start": base,
                    "frame_end": end,
                    "samples": samples,
                },
            },
        }

    samples = []
    for frame, envelope in frame_weights.items():
        for shape_key, weight in preset.items():
            value = min(1, max(0, weight * intensity * envelope))
            samples.append(
                {"shape_key": shape_key, "frame": frame, "weight": _round6(value)}
            )
    return {
        "beat_id": beat["id"],
        "channel": "face",
        "expression": name,
        "request": {
            "id": _request_id(beat["id"], "face"),
            "op": "apply_shape_keys",
            "params": {
                "object": profile["rig"]["face_object"],
                "name_hint": name_hint,
                "frame_start": base,
                "frame_end": end,
                "samples": samples,
            },
        },
    }


def map_plan_to_bridge_jobs(plan, profile):
    """A validated actor plan in, bridge work out.

    Returns pose and shape-key layers ready to send, plus voice jobs the
    loop feeds to the voice pipeline (speech becomes a bridge request
    only after the pipeline produces a viseme artifact).
    """
    layers = []
    voice_jobs = []
    for beat in plan["beats"]:
        if beat.get("body") is not None:
            layers.append(_map_body_beat(beat, profile))
        if beat.get("gaze") is not None:
            layers.append(_map_gaze_beat(beat, profile))
        if beat.get("face") is not None:
            layers.append(_map_face_beat(beat, profile))
        if beat.get("speech") is not None:
            speech = profile["speech"]
            voice_jobs.append(
                {
                    "beat_id": beat["id"],
                    "text": beat["speech"]["text"],
                    "delivery": beat["speech"].get("delivery", "neutral"),
                    "frame_start": _ms_to_frame(beat["at_ms"], profile["fps"]),
                    "fps": profile["fps"],
                    "object": speech["object"],
                    "name_hint": sanitize_name_hint(speech["name_hint"], "speech"),
                    "voice": speech["voice"],
                    "lang": speech.get("lang", "a"),
                    "speed": speech.get("speed", 1.0),
                    "seed": speech.get("seed", 0),
                    "stem": sanitize_name_hint(f"{beat['id']}_speech", "speech"),
                }
            )
    return {
        "profile": {
            "name": profile["profile"],
            "version": profile["profile_version"],
        },
        "layers": layers,
        "voice_jobs": voice_jobs,
    }


def viseme_artifact_to_request(artifact, request_id=None, obj=None):
    """A viseme take artifact becomes one atomic apply_shape_keys request.

    Same field-for-field mapping speech_client.py and the JS mapper use.
    """
    if not _is_object(artifact) or artifact.get("kind") != "animus_viseme_take":
        kind = artifact.get("kind") if _is_object(artifact) else artifact
        raise ValueError(
            f"artifact kind is '{kind}', expected 'animus_viseme_take'"
        )
    return {
        "id": request_id or "speech-take",
        "op": "apply_shape_keys",
        "params": {
            "object": obj or artifact["object"],
            "name_hint": artifact["name_hint"],
            "frame_start": artifact["frame_start"],
            "frame_end": artifact["frame_end"],
            "samples": artifact["samples"],
        },
    }


def viseme_artifact_to_pose_request(artifact, speech, request_id=None, obj=None):
    """A viseme take artifact becomes one atomic perform_take request.

    For rigs with no shape keys (speech.mode 'bone'): the driver viseme
    weight (default 'mouth_open') becomes, frame for frame, an nlerp
    between each configured bone's neutral and peak pose. The voice
    pipeline still owns the timing; the profile still owns every number.
    """
    if not _is_object(artifact) or artifact.get("kind") != "animus_viseme_take":
        kind = artifact.get("kind") if _is_object(artifact) else artifact
        raise ValueError(
            f"artifact kind is '{kind}', expected 'animus_viseme_take'"
        )
    driver = speech.get("driver", "mouth_open")
    driven = {}
    loudest = {}
    for sample in artifact["samples"]:
        frame = sample["frame"]
        weight = sample["weight"]
        loudest[frame] = max(loudest.get(frame, 0.0), weight)
        if sample["shape_key"] == driver:
            driven[frame] = weight
    frames = sorted(loudest)
    bones = speech["bones"]
    stride = 1
    while frames and (len(frames) + stride - 1) // stride * len(bones) > MAX_SAMPLES:
        stride += 1
    samples = []
    for index, frame in enumerate(frames):
        if index % stride and frame != frames[-1]:
            continue
        weight = min(1, max(0, driven.get(frame, loudest[frame])))
        for bone in bones:
            samples.append(
                {
                    "bone": bone["bone"],
                    "frame": frame,
                    "rotation_quaternion": _nlerp(
                        bone["neutral"], bone["peak"], weight
                    ),
                }
            )
    return {
        "id": request_id or "speech-take",
        "op": "perform_take",
        "params": {
            "object": obj or speech["object"],
            "name_hint": artifact["name_hint"],
            "frame_start": artifact["frame_start"],
            "frame_end": artifact["frame_end"],
            "samples": samples,
        },
    }

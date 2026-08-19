"""Wire protocol for the Animus Blender bridge.

Requests are newline-delimited JSON objects:

    {"id": "req-1", "op": "inspect_rig", "params": {"object": "KVRC"}}

Responses are newline-delimited JSON objects:

    {"id": "req-1", "ok": true, "op": "inspect_rig", "result": {...}}
    {"id": "req-1", "ok": false, "op": "inspect_rig",
     "error": {"code": "unknown_object", "message": "..."}}

Everything in this module is pure Python. It never imports bpy, so the
socket thread can run it safely. Scene-dependent checks live in
operations.py and run on Blender's main thread.
"""

import math
import re

PROTOCOL_VERSION = "0.1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_LINE_BYTES = 262144

MAX_SAMPLES = 2000
MAX_FRAME = 100000
MAX_FRAME_SPAN = 10000

# Typed refusal codes. Every failed request maps to exactly one of these.
BAD_JSON = "bad_json"
BAD_REQUEST = "bad_request"
UNKNOWN_OPERATION = "unknown_operation"
INVALID_PARAMS = "invalid_params"
UNKNOWN_OBJECT = "unknown_object"
NOT_AN_ARMATURE = "not_an_armature"
UNKNOWN_BONE = "unknown_bone"
UNKNOWN_ACTION = "unknown_action"
PROTECTED_ACTION = "protected_action"
FRAME_OUT_OF_RANGE = "frame_out_of_range"
INTERNAL_ERROR = "internal_error"

_NAME_HINT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,59}$")

_SAMPLE_KEYS = frozenset({"bone", "frame", "location", "rotation_quaternion"})


class Refusal(Exception):
    """A typed refusal. Raising this mutates nothing."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_error(self):
        return {"code": self.code, "message": self.message}


def _require_keys(params, required, optional, op):
    allowed = set(required) | set(optional)
    for key in params:
        if key not in allowed:
            raise Refusal(INVALID_PARAMS, f"{op}: unknown param '{key}'")
    for key in required:
        if key not in params:
            raise Refusal(INVALID_PARAMS, f"{op}: missing param '{key}'")


def _require_name(params, key, op, max_length=120):
    value = params.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise Refusal(
            INVALID_PARAMS,
            f"{op}: '{key}' must be a non-empty string of at most {max_length} chars",
        )
    return value


def _require_name_hint(params, op):
    value = params.get("name_hint")
    if not isinstance(value, str) or not _NAME_HINT_RE.match(value):
        raise Refusal(
            INVALID_PARAMS,
            f"{op}: 'name_hint' must match [A-Za-z0-9_][A-Za-z0-9_.-]*, max 60 chars",
        )
    return value


def _require_frame(value, op, label):
    if not isinstance(value, int) or isinstance(value, bool):
        raise Refusal(INVALID_PARAMS, f"{op}: '{label}' must be an integer")
    if value < 0 or value > MAX_FRAME:
        raise Refusal(
            FRAME_OUT_OF_RANGE, f"{op}: '{label}' must be from 0 to {MAX_FRAME}"
        )
    return value


def _require_frame_range(params, op):
    start = _require_frame(params.get("frame_start"), op, "frame_start")
    end = _require_frame(params.get("frame_end"), op, "frame_end")
    if start > end:
        raise Refusal(
            FRAME_OUT_OF_RANGE, f"{op}: frame_start must not exceed frame_end"
        )
    if end - start > MAX_FRAME_SPAN:
        raise Refusal(
            FRAME_OUT_OF_RANGE,
            f"{op}: frame range must span at most {MAX_FRAME_SPAN} frames",
        )
    return start, end


def _require_finite_vector(value, size, op, label):
    if not isinstance(value, list) or len(value) != size:
        raise Refusal(
            INVALID_PARAMS, f"{op}: '{label}' must be a list of {size} numbers"
        )
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Refusal(INVALID_PARAMS, f"{op}: '{label}' must contain only numbers")
        if not math.isfinite(item):
            raise Refusal(INVALID_PARAMS, f"{op}: '{label}' must be finite")
        out.append(float(item))
    return out


def _require_samples(params, frame_start, frame_end, op):
    samples = params.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Refusal(INVALID_PARAMS, f"{op}: 'samples' must be a non-empty list")
    if len(samples) > MAX_SAMPLES:
        raise Refusal(
            INVALID_PARAMS, f"{op}: 'samples' may contain at most {MAX_SAMPLES} items"
        )
    normalized = []
    for index, sample in enumerate(samples):
        label = f"samples[{index}]"
        if not isinstance(sample, dict):
            raise Refusal(INVALID_PARAMS, f"{op}: {label} must be an object")
        for key in sample:
            if key not in _SAMPLE_KEYS:
                raise Refusal(INVALID_PARAMS, f"{op}: {label} has unknown key '{key}'")
        bone = sample.get("bone")
        if not isinstance(bone, str) or not bone.strip() or len(bone) > 120:
            raise Refusal(INVALID_PARAMS, f"{op}: {label}.bone must be a bone name")
        frame = sample.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise Refusal(INVALID_PARAMS, f"{op}: {label}.frame must be an integer")
        if frame < frame_start or frame > frame_end:
            raise Refusal(
                FRAME_OUT_OF_RANGE,
                f"{op}: {label}.frame {frame} is outside the declared range "
                f"{frame_start} to {frame_end}",
            )
        clean = {"bone": bone, "frame": frame}
        has_channel = False
        if "location" in sample:
            clean["location"] = _require_finite_vector(
                sample["location"], 3, op, f"{label}.location"
            )
            has_channel = True
        if "rotation_quaternion" in sample:
            clean["rotation_quaternion"] = _require_finite_vector(
                sample["rotation_quaternion"], 4, op, f"{label}.rotation_quaternion"
            )
            has_channel = True
        if not has_channel:
            raise Refusal(
                INVALID_PARAMS,
                f"{op}: {label} needs 'location' or 'rotation_quaternion'",
            )
        normalized.append(clean)
    return normalized


def _validate_inspect_rig(params):
    _require_keys(params, ["object"], [], "inspect_rig")
    return {"object": _require_name(params, "object", "inspect_rig")}


def _validate_create_action(params):
    _require_keys(params, ["name_hint"], [], "create_action")
    return {"name_hint": _require_name_hint(params, "create_action")}


def _validate_apply_pose_keys(params):
    op = "apply_pose_keys"
    _require_keys(params, ["object", "action", "frame_start", "frame_end", "samples"], [], op)
    obj = _require_name(params, "object", op)
    action = _require_name(params, "action", op)
    start, end = _require_frame_range(params, op)
    samples = _require_samples(params, start, end, op)
    return {
        "object": obj,
        "action": action,
        "frame_start": start,
        "frame_end": end,
        "samples": samples,
    }


def _validate_push_to_nla(params):
    op = "push_to_nla"
    _require_keys(params, ["object", "action"], ["name_hint", "frame_start"], op)
    obj = _require_name(params, "object", op)
    action = _require_name(params, "action", op)
    if "name_hint" in params:
        hint = _require_name_hint(params, op)
    else:
        hint = "take"
    if "frame_start" in params:
        start = _require_frame(params.get("frame_start"), op, "frame_start")
    else:
        start = 1
    return {"object": obj, "action": action, "name_hint": hint, "frame_start": start}


def _validate_perform_take(params):
    op = "perform_take"
    _require_keys(
        params, ["object", "name_hint", "frame_start", "frame_end", "samples"], [], op
    )
    obj = _require_name(params, "object", op)
    hint = _require_name_hint(params, op)
    start, end = _require_frame_range(params, op)
    samples = _require_samples(params, start, end, op)
    return {
        "object": obj,
        "name_hint": hint,
        "frame_start": start,
        "frame_end": end,
        "samples": samples,
    }


VALIDATORS = {
    "inspect_rig": _validate_inspect_rig,
    "create_action": _validate_create_action,
    "apply_pose_keys": _validate_apply_pose_keys,
    "push_to_nla": _validate_push_to_nla,
    "perform_take": _validate_perform_take,
}

OPERATIONS = frozenset(VALIDATORS)

MUTATING_OPERATIONS = frozenset(
    {"create_action", "apply_pose_keys", "push_to_nla", "perform_take"}
)


def validate_envelope(obj):
    """Check the request envelope. Returns (request_id, op, raw_params)."""
    if not isinstance(obj, dict):
        raise Refusal(BAD_REQUEST, "request must be a JSON object")
    request_id = obj.get("id")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 120:
        raise Refusal(BAD_REQUEST, "request 'id' must be a short non-empty string")
    for key in obj:
        if key not in ("id", "op", "params"):
            raise Refusal(BAD_REQUEST, f"request has unknown key '{key}'")
    op = obj.get("op")
    if not isinstance(op, str):
        raise Refusal(BAD_REQUEST, "request 'op' must be a string")
    if op not in OPERATIONS:
        raise Refusal(
            UNKNOWN_OPERATION,
            f"unknown operation '{op}'; known: {', '.join(sorted(OPERATIONS))}",
        )
    params = obj.get("params", {})
    if not isinstance(params, dict):
        raise Refusal(BAD_REQUEST, "request 'params' must be an object")
    return request_id, op, params


def validate_params(op, params):
    """Structural validation only. No bpy, no scene access."""
    return VALIDATORS[op](params)


def success(request_id, op, result):
    return {"id": request_id, "ok": True, "op": op, "result": result}


def failure(request_id, op, refusal):
    return {"id": request_id, "ok": False, "op": op, "error": refusal.to_error()}

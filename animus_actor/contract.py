"""Actor plan contract, Python port.

The reference validator is src/animus/contract.js (decision A-004: scene
state plus instruction in, semantic action beats out, never raw
keyframes). This port must stay behavior-identical to the JS validator;
tests/animus_actor replays the committed cross-language fixture to hold
the two together. One deliberate hardening over the JS source: a plan
with no summary at all is refused here with a typed error (the schema
marks summary required; the JS validator would throw on .trim()
instead). Nothing in this module talks to the bridge; the bridge's own
protocol.py stays the fixed side of every disagreement.
"""

SCHEMA_VERSION = "0.1"

CONTROL_LEVELS = ("suggest", "preview", "perform")

BODY_ACTIONS = ("idle", "walk_to", "turn_to", "gesture", "interact", "wait")

BODY_STYLES = (
    "neutral",
    "warm",
    "cold",
    "confident",
    "careful",
    "energetic",
    "tired",
)

_TOP_LEVEL_KEYS = {"schema_version", "summary", "beats"}
_BEAT_KEYS = {
    "id",
    "at_ms",
    "duration_ms",
    "body",
    "gaze",
    "face",
    "face_glyph",
    "speech",
}
_BODY_KEYS = {"action", "target", "gesture", "style", "intensity"}
_GAZE_KEYS = {"target", "intensity"}
_FACE_KEYS = {"expression", "intensity"}
_SPEECH_KEYS = {"text", "delivery"}

# The face_glyph channel (reel-polish brief, directive 3): a composed
# visor face instead of a library expression. Mirrors the JS contract
# and src/animus/face/glyphComposer.js exactly.
FACE_GLYPH_EYES = (
    "round",
    "oval",
    "bar",
    "happy_arc",
    "half_lidded",
    "closed",
    "wide",
    "x_cross",
)
FACE_GLYPH_BROWS = ("none", "flat", "raised", "angry_in", "sad_out")
FACE_GLYPH_MOUTHS = (
    "none",
    "flat",
    "smile",
    "frown",
    "o_small",
    "grin_rect",
    "gritted",
    "v_smile",
    "wavy",
)
FACE_GLYPH_MOODS = (
    "cold",
    "warm",
    "glitch",
    "static",
    "data",
    "boot",
    "angry",
    "dream",
)
FACE_GLYPH_TEXT_MAX = 6
_FACE_GLYPH_TEXT_ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?%+-*#<>:=._"
)
_FACE_GLYPH_KEYS = {"eyes", "brows", "mouth", "text", "mood", "intensity"}
_FORBIDDEN_KEYS = {
    "code",
    "python",
    "keyframe",
    "keyframes",
    "raw_keyframes",
    "fcurve",
    "fcurves",
}

_MISSING = object()


def _is_object(value):
    return isinstance(value, dict)


def _is_finite_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _add_unknown_key_errors(value, allowed, path, errors):
    for key in value:
        if key not in allowed:
            errors.append(f"{path}.{key} is not allowed")


def _find_forbidden_keys(value, path, errors):
    if isinstance(value, list):
        for index, item in enumerate(value):
            _find_forbidden_keys(item, f"{path}[{index}]", errors)
        return
    if not _is_object(value):
        return
    for key, child in value.items():
        if str(key).lower() in _FORBIDDEN_KEYS:
            errors.append(f"{path}.{key} is forbidden")
        _find_forbidden_keys(child, f"{path}.{key}", errors)


def _validate_intensity(value, path, errors):
    if not _is_finite_number(value) or value < 0 or value > 1:
        errors.append(f"{path} must be a number from 0 to 1")


def _validate_optional_text(value, path, errors, max_length=120):
    if value is _MISSING:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        errors.append(
            f"{path} must be a non-empty string no longer than "
            f"{max_length} characters"
        )


def _validate_body(body, path, errors):
    if not _is_object(body):
        errors.append(f"{path} must be an object or null")
        return
    _add_unknown_key_errors(body, _BODY_KEYS, path, errors)
    if body.get("action") not in BODY_ACTIONS:
        errors.append(f"{path}.action must be one of: {', '.join(BODY_ACTIONS)}")
    _validate_optional_text(body.get("target", _MISSING), f"{path}.target", errors)
    _validate_optional_text(
        body.get("gesture", _MISSING), f"{path}.gesture", errors, 60
    )
    if "style" in body and body["style"] not in BODY_STYLES:
        errors.append(f"{path}.style must be one of: {', '.join(BODY_STYLES)}")
    if "intensity" in body:
        _validate_intensity(body["intensity"], f"{path}.intensity", errors)

    if body.get("action") in ("walk_to", "turn_to", "interact") and not body.get(
        "target"
    ):
        errors.append(f"{path}.target is required for {body.get('action')}")
    if body.get("action") == "gesture" and not body.get("gesture"):
        errors.append(f"{path}.gesture is required for gesture")


def _validate_gaze(gaze, path, errors):
    if not _is_object(gaze):
        errors.append(f"{path} must be an object or null")
        return
    _add_unknown_key_errors(gaze, _GAZE_KEYS, path, errors)
    _validate_optional_text(gaze.get("target", _MISSING), f"{path}.target", errors)
    if not gaze.get("target"):
        errors.append(f"{path}.target is required")
    if "intensity" in gaze:
        _validate_intensity(gaze["intensity"], f"{path}.intensity", errors)


def _validate_face(face, path, errors):
    if not _is_object(face):
        errors.append(f"{path} must be an object or null")
        return
    _add_unknown_key_errors(face, _FACE_KEYS, path, errors)
    _validate_optional_text(
        face.get("expression", _MISSING), f"{path}.expression", errors, 80
    )
    if not face.get("expression"):
        errors.append(f"{path}.expression is required")
    if "intensity" in face:
        _validate_intensity(face["intensity"], f"{path}.intensity", errors)


def _validate_face_glyph(glyph, path, errors):
    if not _is_object(glyph):
        errors.append(f"{path} must be an object or null")
        return
    _add_unknown_key_errors(glyph, _FACE_GLYPH_KEYS, path, errors)
    has_text = glyph.get("text") is not None
    if has_text:
        for key in ("eyes", "brows", "mouth"):
            if key in glyph:
                errors.append(f"{path}.{key} is not allowed in text mode")
        text = glyph["text"]
        if not isinstance(text, str):
            errors.append(f"{path}.text must be a string")
        else:
            normalized = text.strip().upper()
            if not 1 <= len(normalized) <= FACE_GLYPH_TEXT_MAX:
                errors.append(
                    f"{path}.text must be 1 to {FACE_GLYPH_TEXT_MAX} characters"
                )
            elif not all(ch in _FACE_GLYPH_TEXT_ALLOWED for ch in normalized):
                errors.append(
                    f"{path}.text may use only A-Z 0-9 and ! ? % + - * # < > : = . _"
                )
    else:
        if glyph.get("eyes") not in FACE_GLYPH_EYES:
            errors.append(
                f"{path}.eyes must be one of: {', '.join(FACE_GLYPH_EYES)}"
            )
        if "brows" in glyph and glyph["brows"] not in FACE_GLYPH_BROWS:
            errors.append(
                f"{path}.brows must be one of: {', '.join(FACE_GLYPH_BROWS)}"
            )
        if "mouth" in glyph and glyph["mouth"] not in FACE_GLYPH_MOUTHS:
            errors.append(
                f"{path}.mouth must be one of: {', '.join(FACE_GLYPH_MOUTHS)}"
            )
    if "mood" in glyph and glyph["mood"] not in FACE_GLYPH_MOODS:
        errors.append(f"{path}.mood must be one of: {', '.join(FACE_GLYPH_MOODS)}")
    if "intensity" in glyph:
        _validate_intensity(glyph["intensity"], f"{path}.intensity", errors)


def _validate_speech(speech, path, errors):
    if not _is_object(speech):
        errors.append(f"{path} must be an object or null")
        return
    _add_unknown_key_errors(speech, _SPEECH_KEYS, path, errors)
    _validate_optional_text(
        speech.get("text", _MISSING), f"{path}.text", errors, 500
    )
    _validate_optional_text(
        speech.get("delivery", _MISSING), f"{path}.delivery", errors, 80
    )
    if not speech.get("text"):
        errors.append(f"{path}.text is required")


def validate_actor_plan(candidate, authority=None):
    """Validate a candidate plan against the actor contract.

    Returns {"ok": bool, "errors": [str], "value": dict or None}. On
    success value carries actor_id and control_level from the caller's
    authority, never from the model (decision A-004).
    """
    authority = authority or {}
    errors = []
    if not _is_object(candidate):
        return {"ok": False, "errors": ["plan must be an object"], "value": None}

    _find_forbidden_keys(candidate, "plan", errors)
    _add_unknown_key_errors(candidate, _TOP_LEVEL_KEYS, "plan", errors)

    if candidate.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"plan.schema_version must be {SCHEMA_VERSION}")
    if "summary" not in candidate:
        errors.append(
            "plan.summary must be a non-empty string no longer than 200 characters"
        )
    else:
        _validate_optional_text(candidate["summary"], "plan.summary", errors, 200)

    beats = candidate.get("beats")
    if not isinstance(beats, list) or len(beats) == 0:
        errors.append("plan.beats must contain at least one beat")
    elif len(beats) > 8:
        errors.append("plan.beats may contain at most 8 beats")
    else:
        for index, beat in enumerate(beats):
            path = f"plan.beats[{index}]"
            if not _is_object(beat):
                errors.append(f"{path} must be an object")
                continue
            _add_unknown_key_errors(beat, _BEAT_KEYS, path, errors)
            _validate_optional_text(
                beat.get("id", _MISSING), f"{path}.id", errors, 60
            )
            if not beat.get("id"):
                errors.append(f"{path}.id is required")
            at_ms = beat.get("at_ms")
            if not _is_integer(at_ms) or at_ms < 0 or at_ms > 60000:
                errors.append(f"{path}.at_ms must be an integer from 0 to 60000")
            duration_ms = beat.get("duration_ms")
            if (
                not _is_integer(duration_ms)
                or duration_ms < 100
                or duration_ms > 30000
            ):
                errors.append(
                    f"{path}.duration_ms must be an integer from 100 to 30000"
                )

            channels = [
                key
                for key in ("body", "gaze", "face", "face_glyph", "speech")
                if beat.get(key) is not None
            ]
            if not channels:
                errors.append(f"{path} must use at least one actor channel")
            if (
                beat.get("face") is not None
                and beat.get("face_glyph") is not None
            ):
                errors.append(f"{path} may use face or face_glyph, not both")
            if beat.get("body") is not None:
                _validate_body(beat["body"], f"{path}.body", errors)
            if beat.get("gaze") is not None:
                _validate_gaze(beat["gaze"], f"{path}.gaze", errors)
            if beat.get("face") is not None:
                _validate_face(beat["face"], f"{path}.face", errors)
            if beat.get("face_glyph") is not None:
                _validate_face_glyph(
                    beat["face_glyph"], f"{path}.face_glyph", errors
                )
            if beat.get("speech") is not None:
                _validate_speech(beat["speech"], f"{path}.speech", errors)

    control_level = authority.get("control_level", "suggest")
    if control_level not in CONTROL_LEVELS:
        errors.append(
            f"authority.control_level must be one of: {', '.join(CONTROL_LEVELS)}"
        )
    actor_id = authority.get("actor_id", _MISSING)
    _validate_optional_text(actor_id, "authority.actor_id", errors, 80)
    if actor_id is _MISSING or not actor_id:
        errors.append("authority.actor_id is required")

    if errors:
        return {"ok": False, "errors": errors, "value": None}

    return {
        "ok": True,
        "errors": [],
        "value": {
            "schema_version": SCHEMA_VERSION,
            "actor_id": actor_id,
            "control_level": control_level,
            "summary": candidate["summary"].strip(),
            "beats": beats,
        },
    }

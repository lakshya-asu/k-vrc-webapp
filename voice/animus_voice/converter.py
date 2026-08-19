"""Convert Rhubarb mouth cues into an Animus viseme track.

This module is pure Python. It imports no torch, no soundfile, and runs
no binary. It takes Rhubarb Lip Sync JSON output and produces a timed
viseme take the K-VRC embodiment profile can apply as face-screen
shape-key weights.

Artifact shape mirrors the P1 Blender bridge take vocabulary
(blender/animus_bridge/operations.py perform_take) as closely as is
reasonable for a face track:

    bridge take                 viseme take
    -----------                 -----------
    object                      object            target datablock name
    name_hint                   name_hint         base name for the take
    frame_start / frame_end     frame_start / frame_end
    samples[].bone              samples[].shape_key   one animated channel
    samples[].frame             samples[].frame
    samples[].location/quat     samples[].weight      one scalar value

A bone channel in the bridge carries one float per keyframe; a shape key
carries one float per keyframe too, so the mapping is one to one. The
viseme take adds at_ms and viseme fields for readability and a source
provenance block, exactly the provenance the architecture doc requires.

On any malformed cue input the converter raises ConverterRefusal and
produces nothing. It never guesses a mouth shape or invents timing.
"""

import json
import math

from .visemes import (
    KVRC_SHAPE_KEYS,
    PROFILE,
    PROFILE_VERSION,
    RHUBARB_SHAPES,
    VISEME_TO_SHAPES,
    full_weights,
)

ARTIFACT_KIND = "animus_viseme_take"
ARTIFACT_SCHEMA_VERSION = "0.1"

# Guard rails. A single spoken line should never exceed these.
MAX_CUES = 100000
MAX_DURATION_S = 3600.0
DEFAULT_FPS = 24


class ConverterRefusal(Exception):
    """A typed refusal. Raising this produces no artifact.

    Mirrors the P1 bridge Refusal: a code plus a human message.
    """

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_error(self):
        return {"code": self.code, "message": self.message}


# Refusal codes.
BAD_JSON = "bad_json"
BAD_CUE_FILE = "bad_cue_file"
BAD_CUE = "bad_cue"
UNKNOWN_VISEME = "unknown_viseme"
BAD_FPS = "bad_fps"
BAD_PARAM = "bad_param"


def _is_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def load_rhubarb_cues(source):
    """Load and validate Rhubarb cues from a path, string, or parsed dict.

    Returns a normalized list of cues, each a dict with float start, float
    end, and a known viseme value. Also returns the reported duration.
    Raises ConverterRefusal on anything malformed.
    """
    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, bytes)):
        text = source
        # A path is a short string that exists on disk; anything else is
        # treated as raw JSON text.
        if isinstance(source, str) and len(source) < 4096 and "\n" not in source:
            try:
                with open(source, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except (OSError, ValueError):
                text = source
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as error:
            raise ConverterRefusal(BAD_JSON, f"cue input is not valid JSON: {error}")
    else:
        raise ConverterRefusal(
            BAD_CUE_FILE, "cue source must be a path, JSON string, or parsed object"
        )

    if not isinstance(data, dict):
        raise ConverterRefusal(BAD_CUE_FILE, "Rhubarb output must be a JSON object")

    raw_cues = data.get("mouthCues")
    if not isinstance(raw_cues, list):
        raise ConverterRefusal(
            BAD_CUE_FILE, "Rhubarb output must contain a 'mouthCues' list"
        )
    if not raw_cues:
        raise ConverterRefusal(BAD_CUE_FILE, "'mouthCues' is empty; nothing to convert")
    if len(raw_cues) > MAX_CUES:
        raise ConverterRefusal(
            BAD_CUE_FILE, f"'mouthCues' has more than {MAX_CUES} entries"
        )

    metadata = data.get("metadata")
    duration = None
    if isinstance(metadata, dict) and "duration" in metadata:
        candidate = metadata["duration"]
        if _is_number(candidate) and math.isfinite(candidate) and candidate >= 0:
            duration = float(candidate)

    cues = []
    previous_start = -1.0
    for index, cue in enumerate(raw_cues):
        label = f"mouthCues[{index}]"
        if not isinstance(cue, dict):
            raise ConverterRefusal(BAD_CUE, f"{label} must be an object")
        start = cue.get("start")
        end = cue.get("end")
        value = cue.get("value")
        if not _is_number(start) or not math.isfinite(start):
            raise ConverterRefusal(BAD_CUE, f"{label}.start must be a finite number")
        if not _is_number(end) or not math.isfinite(end):
            raise ConverterRefusal(BAD_CUE, f"{label}.end must be a finite number")
        start = float(start)
        end = float(end)
        if start < 0 or end < 0:
            raise ConverterRefusal(BAD_CUE, f"{label} times must not be negative")
        if end < start:
            raise ConverterRefusal(BAD_CUE, f"{label}.end precedes its start")
        if start > MAX_DURATION_S:
            raise ConverterRefusal(
                BAD_CUE, f"{label}.start exceeds {MAX_DURATION_S} seconds"
            )
        if start < previous_start:
            raise ConverterRefusal(
                BAD_CUE, f"{label} starts before the previous cue; cues must be ordered"
            )
        previous_start = start
        if not isinstance(value, str) or value not in RHUBARB_SHAPES:
            known = ", ".join(sorted(RHUBARB_SHAPES))
            raise ConverterRefusal(
                UNKNOWN_VISEME,
                f"{label}.value '{value}' is not a known Rhubarb shape; known: {known}",
            )
        if value not in VISEME_TO_SHAPES:
            raise ConverterRefusal(
                UNKNOWN_VISEME, f"{label}.value '{value}' has no shape-key mapping"
            )
        cues.append({"start": start, "end": end, "value": value})

    return cues, duration


def _validate_fps(fps):
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise ConverterRefusal(BAD_FPS, "fps must be a positive integer")
    if fps < 1 or fps > 240:
        raise ConverterRefusal(BAD_FPS, "fps must be from 1 to 240")
    return fps


def cues_to_viseme_take(
    cues,
    duration=None,
    fps=DEFAULT_FPS,
    frame_start=1,
    obj="KVRC",
    name_hint="animus_speech",
    source=None,
):
    """Build the Animus viseme take artifact from validated cues.

    cues is the list returned by load_rhubarb_cues. Deterministic: the
    same cues, fps, and frame_start always yield byte-identical samples.
    Each cue start becomes a keyframe; every KVRC_SHAPE_KEYS channel is
    written at that frame so the mouth is fully specified and no earlier
    weight leaks across a cue boundary.
    """
    fps = _validate_fps(fps)
    if isinstance(frame_start, bool) or not isinstance(frame_start, int):
        raise ConverterRefusal(BAD_PARAM, "frame_start must be an integer")
    if frame_start < 0:
        raise ConverterRefusal(BAD_PARAM, "frame_start must not be negative")
    if not isinstance(obj, str) or not obj.strip():
        raise ConverterRefusal(BAD_PARAM, "obj must be a non-empty string")
    if not isinstance(name_hint, str) or not name_hint.strip():
        raise ConverterRefusal(BAD_PARAM, "name_hint must be a non-empty string")
    if not isinstance(cues, list) or not cues:
        raise ConverterRefusal(BAD_CUE, "cues must be a non-empty list")

    samples = []
    cue_track = []
    max_frame = frame_start
    for cue in cues:
        start = cue["start"]
        viseme = cue["value"]
        frame = frame_start + int(round(start * fps))
        at_ms = int(round(start * 1000.0))
        weights = full_weights(viseme)
        cue_track.append(
            {"at_ms": at_ms, "frame": frame, "viseme": viseme, "weights": weights}
        )
        for shape_key in KVRC_SHAPE_KEYS:
            samples.append(
                {
                    "frame": frame,
                    "at_ms": at_ms,
                    "viseme": viseme,
                    "shape_key": shape_key,
                    "weight": weights[shape_key],
                }
            )
        if frame > max_frame:
            max_frame = frame

    # frame_end covers the last cue's end, not only its start, so the take
    # spans the whole utterance.
    last_end = cues[-1]["end"]
    end_frame = frame_start + int(round(last_end * fps))
    frame_end = max(max_frame, end_frame)

    if duration is None:
        duration = last_end
    duration_ms = int(round(duration * 1000.0))

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "embodiment": {
            "profile": PROFILE,
            "profile_version": PROFILE_VERSION,
            "target": "shape_keys",
            "shape_keys": list(KVRC_SHAPE_KEYS),
        },
        "object": obj,
        "name_hint": name_hint,
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration_ms": duration_ms,
        "cue_count": len(cues),
        "cues": cue_track,
        "samples": samples,
    }
    if source is not None:
        if not isinstance(source, dict):
            raise ConverterRefusal(BAD_PARAM, "source provenance must be an object")
        artifact["source"] = source
    return artifact

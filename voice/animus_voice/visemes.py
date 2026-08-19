"""Viseme mapping for the K-VRC embodiment profile.

Rhubarb Lip Sync emits mouth cues using an extended Preston Blair set:
the basic shapes A to F plus the idle shape X, and two optional extended
shapes G and H. See https://github.com/DanielSWolf/rhubarb-lip-sync for
the authoritative descriptions. This module records those shapes and maps
each one to a set of K-VRC face-screen shape-key weights.

K-VRC has no physical mouth. Its face is a screen with named float
parameters drawn each frame. The relevant mouth parameters come from
modal_app/heads.py FACE_PARAM_NAMES and the expression library design:

    mouth_open        0..1  how far the mouth is open
    smile_width       0..1  how wide the mouth is stretched
    mouth_curl_left   0..1  left corner lift
    mouth_curl_right  0..1  right corner lift

A shape-key weight is a single float in 0..1. That is the direct analog
of one FCurve channel value in the P1 Blender bridge (operations.py),
where a bone channel carries one float per keyframe. The viseme track
this module feeds keeps the same shape: a flat list of timed samples,
each naming one target (shape_key, the analog of the bridge's bone) and
one scalar value (weight, the analog of a channel value).

The weights below are authored defaults, not measured values. They are
readable, symmetric on the two mouth corners, and land the idle and
closed shapes at a shut mouth so silence reads as a resting face.
"""

# Profile identity travels with every artifact for provenance.
PROFILE = "kvrc"
PROFILE_VERSION = "0.1.0"

# The mouth face-screen parameters this profile drives. Any shape-key
# named in VISEME_TO_SHAPES must appear here. Kept in the FACE_PARAM_NAMES
# order used by modal_app/heads.py so the two stay legible together.
KVRC_SHAPE_KEYS = (
    "mouth_open",
    "smile_width",
    "mouth_curl_left",
    "mouth_curl_right",
)

# Human-readable notes on each Rhubarb shape. The pairs are the mouth
# shape's typical phonemes, not an exhaustive phoneme map.
RHUBARB_SHAPES = {
    "A": "closed mouth for the P, B, and M sounds",
    "B": "slightly open mouth with clenched teeth; many consonants and the EE vowel",
    "C": "open mouth for the EH and AE vowels",
    "D": "wide open mouth for the AA vowel",
    "E": "slightly rounded open mouth for AO and ER",
    "F": "puckered lips for UW, OW, and W",
    "G": "upper teeth on lower lip for F and V (extended shape)",
    "H": "tongue-up L shape (extended shape)",
    "X": "idle or rest position for silence",
}

# Each Rhubarb shape maps to a full set of mouth shape-key weights. A key
# absent from a shape's dict is treated as 0.0 by the converter, so every
# emitted frame writes the complete KVRC_SHAPE_KEYS set and no stale weight
# from an earlier viseme lingers. Weights stay in 0..1.
VISEME_TO_SHAPES = {
    # Silence and closed shapes shut the mouth. A small resting smile keeps
    # the idle face from reading as a flat line.
    "X": {"mouth_open": 0.00, "smile_width": 0.10, "mouth_curl_left": 0.05, "mouth_curl_right": 0.05},
    "A": {"mouth_open": 0.00, "smile_width": 0.15, "mouth_curl_left": 0.05, "mouth_curl_right": 0.05},
    # Clenched-teeth consonants and EE: barely open, wide.
    "B": {"mouth_open": 0.20, "smile_width": 0.55, "mouth_curl_left": 0.10, "mouth_curl_right": 0.10},
    # EH and AE: mid open, mid wide.
    "C": {"mouth_open": 0.45, "smile_width": 0.40, "mouth_curl_left": 0.05, "mouth_curl_right": 0.05},
    # AA: wide open jaw.
    "D": {"mouth_open": 0.85, "smile_width": 0.30, "mouth_curl_left": 0.00, "mouth_curl_right": 0.00},
    # AO and ER: rounded, mid open.
    "E": {"mouth_open": 0.50, "smile_width": 0.15, "mouth_curl_left": 0.00, "mouth_curl_right": 0.00},
    # UW, OW, W: puckered, narrow.
    "F": {"mouth_open": 0.30, "smile_width": 0.00, "mouth_curl_left": 0.00, "mouth_curl_right": 0.00},
    # F and V: teeth on lip, slightly open and wide.
    "G": {"mouth_open": 0.15, "smile_width": 0.35, "mouth_curl_left": 0.05, "mouth_curl_right": 0.05},
    # L: mid open, mid wide.
    "H": {"mouth_open": 0.35, "smile_width": 0.30, "mouth_curl_left": 0.05, "mouth_curl_right": 0.05},
}


def full_weights(viseme):
    """Return the complete shape-key weight set for one viseme.

    Every key in KVRC_SHAPE_KEYS is present. Missing entries fall to 0.0
    so each frame fully specifies the mouth and cannot inherit a stale
    weight from a neighbouring cue.
    """
    mapped = VISEME_TO_SHAPES[viseme]
    return {key: float(mapped.get(key, 0.0)) for key in KVRC_SHAPE_KEYS}

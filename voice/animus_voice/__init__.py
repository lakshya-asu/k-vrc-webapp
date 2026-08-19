"""Animus voice pipeline (P2).

Text to Kokoro TTS audio, to Rhubarb lip-sync cues, to an Animus viseme
track the K-VRC embodiment profile can apply. The converter and artifact
layer are pure Python and run without any model or binary. The TTS and
Rhubarb steps are optional and are skipped cleanly when the tools are
absent.
"""

from .visemes import (
    RHUBARB_SHAPES,
    VISEME_TO_SHAPES,
    KVRC_SHAPE_KEYS,
    PROFILE,
    PROFILE_VERSION,
)
from .converter import (
    ConverterRefusal,
    cues_to_viseme_take,
    load_rhubarb_cues,
)

__all__ = [
    "RHUBARB_SHAPES",
    "VISEME_TO_SHAPES",
    "KVRC_SHAPE_KEYS",
    "PROFILE",
    "PROFILE_VERSION",
    "ConverterRefusal",
    "cues_to_viseme_take",
    "load_rhubarb_cues",
]

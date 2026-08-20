"""Test bootstrap for the actor loop suite. Import before anything else.

Puts the repo root (animus_actor), blender/ (animus_bridge), voice/
(animus_voice), and the bridge test dir (fake_bpy) on sys.path, then
installs fake bpy under the name 'bpy'. Blender itself is exercised by
the acceptance suite, not here.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_BRIDGE_TESTS = os.path.join(_REPO, "tests", "animus_bridge")

for _entry in (
    _REPO,
    os.path.join(_REPO, "blender"),
    os.path.join(_REPO, "voice"),
    _BRIDGE_TESTS,
    _HERE,
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import fake_bpy  # noqa: E402

sys.modules["bpy"] = fake_bpy

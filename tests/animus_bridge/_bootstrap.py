"""Test bootstrap. Import this before any animus_bridge module.

Installs the fake bpy module under the name 'bpy' and puts the blender/
directory on sys.path so 'animus_bridge' imports resolve. Blender is not
installed on this machine; every test runs against the fake.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_BLENDER_DIR = os.path.join(_REPO, "blender")

if _BLENDER_DIR not in sys.path:
    sys.path.insert(0, _BLENDER_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fake_bpy  # noqa: E402

sys.modules["bpy"] = fake_bpy

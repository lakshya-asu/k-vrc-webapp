"""animus_actor: one command from a script line to a performed take.

Python actor loop for the Animus proof:

    script line -> deterministic actor plan (no model, no GPU)
                -> contract validation (port of src/animus/contract.js)
                -> embodiment mapping (port of src/animus/embodiment.js)
                -> voice synthesis + visemes (animus_voice, CPU only)
                -> typed bridge requests (perform_take / apply_shape_keys)
                -> receipts

Run from the repo root:

    python -m animus_actor "Hello, I am K-VRC"

By default this launches headless Blender itself (animus_actor/stage.py
builds the test rig and drains the bridge), performs every mapped layer,
and reports a receipt: voice track path, viseme count, beats executed,
and the bridge op receipts. --attach connects to an already running
bridge instead.

The bridge contract (blender/animus_bridge/protocol.py) is the fixed
side. If this loop and the bridge ever disagree, adapt here, never
there.
"""

__version__ = "0.1.0"

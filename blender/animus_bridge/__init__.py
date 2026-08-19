"""Animus Bridge: a typed localhost animation API for Blender.

P1 of the Animus proof. The add-on runs a socket server on 127.0.0.1.
Socket threads only parse, validate, and queue. A bpy.app.timers
callback drains the queue on Blender's main thread. Operations are
narrow and typed: inspect_rig, create_action, apply_pose_keys,
push_to_nla, and the atomic perform_take. There is no exec, no eval,
and no raw code path. Every mutation returns the names it created.

Port: set the environment variable ANIMUS_BRIDGE_PORT, default 8765.
"""

import os

import bpy

from .executor import Executor
from .protocol import DEFAULT_PORT
from .server import BridgeServer

bl_info = {
    "name": "Animus Bridge",
    "author": "Lakshya Jain",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "Background service, no UI",
    "description": "Typed localhost animation API. Actions and NLA takes only.",
    "category": "Animation",
}

_executor = None
_server = None


def _configured_port():
    raw = os.environ.get("ANIMUS_BRIDGE_PORT", "")
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    if 0 < port < 65536:
        return port
    return DEFAULT_PORT


def start_bridge(port=None):
    """Start the executor timer and the socket server. Idempotent."""
    global _executor, _server
    if _server is not None:
        return _server
    _executor = Executor()
    _executor.install()
    _server = BridgeServer(_executor, port=port or _configured_port())
    _server.start()
    print(f"[animus_bridge] listening on 127.0.0.1:{_server.port}")
    return _server


def stop_bridge():
    global _executor, _server
    if _server is not None:
        _server.stop()
        _server = None
    if _executor is not None:
        _executor.stop()
        _executor = None


def register():
    if bpy.app.background:
        # Queued work would never drain without the event loop.
        # Same guard as the inspected blender-mcp add-on.
        print("[animus_bridge] background mode detected, server not started")
        return
    start_bridge()


def unregister():
    stop_bridge()

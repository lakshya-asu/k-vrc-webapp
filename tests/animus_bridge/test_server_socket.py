"""End-to-end transport tests over a real localhost socket.

The server binds 127.0.0.1 on an ephemeral port inside the test. The
test thread plays the role of Blender's main thread by pumping the fake
timer registry.
"""

import json
import os
import socket
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import fake_bpy  # noqa: E402
from animus_bridge import protocol  # noqa: E402
from animus_bridge.executor import Executor  # noqa: E402
from animus_bridge.server import BridgeServer  # noqa: E402

BONES = ["root", "arm.R", "hand.R"]


def wave_request(request_id="r1"):
    return {
        "id": request_id,
        "op": "perform_take",
        "params": {
            "object": "KVRC",
            "name_hint": "wave",
            "frame_start": 1,
            "frame_end": 48,
            "samples": [
                {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1, 0, 0, 0]},
                {"bone": "arm.R", "frame": 48, "rotation_quaternion": [0.9, 0, 0, 0.4]},
            ],
        },
    }


class SocketServerTests(unittest.TestCase):
    def setUp(self):
        fake_bpy.reset()
        fake_bpy.add_armature_object("KVRC", BONES)
        self.executor = Executor()
        self.executor.install()
        self.server = BridgeServer(self.executor, port=0)
        self.server.start()
        self.addCleanup(self.server.stop)

    def connect(self):
        client = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
        self.addCleanup(client.close)
        return client

    def read_response(self, client, timeout=5.0):
        """Pump the fake timer while waiting for one response line."""
        client.settimeout(0.05)
        buffer = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fake_bpy.app.timers.pump()
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk
            if b"\n" in buffer:
                line, _rest = buffer.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))
        raise AssertionError("no response before timeout")

    def wait_for_queue(self, size, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.executor.queue.qsize() >= size:
                return
            time.sleep(0.01)
        raise AssertionError(f"queue never reached size {size}")

    def test_wave_round_trip(self):
        client = self.connect()
        client.sendall((json.dumps(wave_request()) + "\n").encode("utf-8"))
        response = self.read_response(client)
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["action"], "ANIMUS_wave_take001")
        self.assertEqual(response["result"]["strip"], "ANIMUS_wave_take001")

    def test_bad_json_gets_typed_refusal_and_never_queues(self):
        client = self.connect()
        client.sendall(b"this is not json\n")
        response = self.read_response(client)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.BAD_JSON)
        self.assertEqual(self.executor.queue.qsize(), 0)
        self.assertEqual(len(fake_bpy.data.actions), 0)

    def test_unknown_operation_refused_before_queue(self):
        client = self.connect()
        request = {"id": "r9", "op": "execute_python", "params": {"code": "1"}}
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        response = self.read_response(client)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.UNKNOWN_OPERATION)
        self.assertEqual(response["id"], "r9")
        self.assertEqual(self.executor.queue.qsize(), 0)

    def test_invalid_params_refused_before_queue(self):
        client = self.connect()
        request = wave_request()
        request["params"]["samples"][0]["frame"] = 999
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        response = self.read_response(client)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.FRAME_OUT_OF_RANGE)
        self.assertEqual(self.executor.queue.qsize(), 0)
        self.assertEqual(len(fake_bpy.data.actions), 0)

    def test_partial_line_then_disconnect_leaves_nothing(self):
        client = self.connect()
        payload = json.dumps(wave_request()).encode("utf-8")
        client.sendall(payload[: len(payload) // 2])
        client.close()
        time.sleep(0.2)
        fake_bpy.app.timers.pump()
        self.assertEqual(self.executor.queue.qsize(), 0)
        self.assertEqual(len(fake_bpy.data.actions), 0)
        obj = fake_bpy.data.objects.get("KVRC")
        self.assertIsNone(obj.animation_data)

    def test_disconnect_after_full_request_applies_fully(self):
        client = self.connect()
        client.sendall((json.dumps(wave_request()) + "\n").encode("utf-8"))
        self.wait_for_queue(1)
        client.close()
        fake_bpy.app.timers.pump()
        self.assertEqual(len(fake_bpy.data.actions), 1)
        obj = fake_bpy.data.objects.get("KVRC")
        strips = [s for t in obj.animation_data.nla_tracks for s in t.strips]
        self.assertEqual(len(strips), 1)
        self.assertEqual(strips[0].action.name, "ANIMUS_wave_take001")

    def test_socket_thread_never_touches_bpy(self):
        client = self.connect()
        client.sendall((json.dumps(wave_request()) + "\n").encode("utf-8"))
        self.wait_for_queue(1)
        self.assertEqual(fake_bpy.ACCESS_THREADS, set())
        response = self.read_response(client)
        self.assertTrue(response["ok"])
        self.assertEqual(fake_bpy.ACCESS_THREADS, {__import__("threading").get_ident()})

    def test_two_requests_one_connection(self):
        client = self.connect()
        client.sendall((json.dumps(wave_request("a")) + "\n").encode("utf-8"))
        first = self.read_response(client)
        client.sendall((json.dumps(wave_request("b")) + "\n").encode("utf-8"))
        second = self.read_response(client)
        self.assertTrue(first["ok"] and second["ok"])
        self.assertNotEqual(first["result"]["action"], second["result"]["action"])


if __name__ == "__main__":
    unittest.main()

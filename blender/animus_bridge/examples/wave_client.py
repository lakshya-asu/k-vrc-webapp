"""Wave demo client for the Animus bridge.

Sends one atomic perform_take request over the localhost socket and
prints the receipt. Stdlib only. Run it from any terminal while Blender
has the add-on enabled.

Usage:
    python wave_client.py --object KVRC [--port 8765]
"""

import argparse
import json
import socket


def wave_request(object_name):
    return {
        "id": "wave-demo",
        "op": "perform_take",
        "params": {
            "object": object_name,
            "name_hint": "wave",
            "frame_start": 1,
            "frame_end": 48,
            "samples": [
                {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
                {"bone": "arm.R", "frame": 12, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
                {"bone": "arm.R", "frame": 24, "rotation_quaternion": [0.98, 0.0, 0.0, -0.2]},
                {"bone": "arm.R", "frame": 36, "rotation_quaternion": [0.92, 0.0, 0.0, 0.38]},
                {"bone": "arm.R", "frame": 48, "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Animus bridge wave demo")
    parser.add_argument("--object", default="KVRC", help="armature object name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    request = wave_request(args.object)
    with socket.create_connection((args.host, args.port), timeout=10) as conn:
        conn.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = b""
        while b"\n" not in buffer:
            chunk = conn.recv(4096)
            if not chunk:
                raise SystemExit("connection closed before a response arrived")
            buffer += chunk
    response = json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))
    print(json.dumps(response, indent=2))
    if not response.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

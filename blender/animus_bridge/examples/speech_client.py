"""Speech take client for the Animus bridge.

Reads an animus_voice viseme take artifact (the *.animus.json the voice
pipeline writes) and sends it to the bridge as one atomic
apply_shape_keys request. Stdlib only.

The mapping is direct, field for field:

    artifact                    request params
    --------                    --------------
    object                      object        (or --object override)
    name_hint                   name_hint
    frame_start / frame_end     frame_start / frame_end
    samples                     samples       (sent as-is)

Artifact samples carry extra at_ms and viseme provenance fields; the
bridge protocol accepts and drops them, so the samples list needs no
client-side rewriting.

Usage:
    python speech_client.py --artifact out/hello.animus.json \
        [--object KVRC_face] [--port 8765]
"""

import argparse
import json
import socket


def build_request(artifact, object_override=None, request_id="speech-take"):
    """Map a viseme take artifact onto one apply_shape_keys request."""
    if artifact.get("kind") != "animus_viseme_take":
        raise SystemExit(
            f"artifact kind is '{artifact.get('kind')}', expected 'animus_viseme_take'"
        )
    return {
        "id": request_id,
        "op": "apply_shape_keys",
        "params": {
            "object": object_override or artifact["object"],
            "name_hint": artifact["name_hint"],
            "frame_start": artifact["frame_start"],
            "frame_end": artifact["frame_end"],
            "samples": artifact["samples"],
        },
    }


def send_request(request, host="127.0.0.1", port=8765, timeout=10):
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = b""
        while b"\n" not in buffer:
            chunk = conn.recv(4096)
            if not chunk:
                raise SystemExit("connection closed before a response arrived")
            buffer += chunk
    return json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Animus bridge speech take demo")
    parser.add_argument("--artifact", required=True, help="viseme take JSON path")
    parser.add_argument("--object", default=None, help="override target object name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with open(args.artifact, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    request = build_request(artifact, object_override=args.object)
    response = send_request(request, host=args.host, port=args.port)
    print(json.dumps(response, indent=2))
    if not response.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

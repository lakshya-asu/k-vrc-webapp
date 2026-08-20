"""Acceptance client for the Animus bridge. One step per invocation.

Runs as a separate OS process from Blender, so every request crosses
the real localhost socket. Stdlib only. Steps:

    inspect          inspect_rig on the armature
    wave             one atomic perform_take (pose keys)
    invalid          two refused payloads: a structural refusal the
                     socket thread answers (sample frame outside the
                     declared range) and a scene refusal the main
                     thread answers (unknown bone)
    disconnect       send half a request, close, never read a reply
    shape_keys       one apply_shape_keys take from a voice artifact
                     (--artifact path, sent via speech_client mapping)

Prints one JSON object per response to stdout. Exit 0 when the step
behaved as expected (a refusal is EXPECTED for 'invalid').
"""

import argparse
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
sys.path.insert(0, EXAMPLES)

from speech_client import build_request, send_request  # noqa: E402
from wave_client import wave_request  # noqa: E402


def send(request, port):
    return send_request(request, port=port)


def step_inspect(args):
    response = send(
        {"id": "acc-inspect", "op": "inspect_rig", "params": {"object": args.object}},
        args.port,
    )
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


def step_wave(args):
    request = wave_request(args.object)
    request["id"] = args.request_id
    response = send(request, args.port)
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


def step_invalid(args):
    # Structural refusal: declared range is 1..48 but a sample sits at
    # frame 9000. The socket thread refuses this before anything is
    # queued.
    bad_frame = wave_request(args.object)
    bad_frame["id"] = "acc-bad-frame"
    bad_frame["params"]["samples"][2]["frame"] = 9000
    first = send(bad_frame, args.port)
    print(json.dumps(first))

    # Scene refusal: a bone the armature does not have. Validation
    # passes structurally; the main thread inspects the scene and
    # refuses without mutating.
    bad_bone = wave_request(args.object)
    bad_bone["id"] = "acc-bad-bone"
    bad_bone["params"]["samples"][0]["bone"] = "tentacle.R"
    second = send(bad_bone, args.port)
    print(json.dumps(second))

    ok = (
        not first.get("ok")
        and first.get("error", {}).get("code") == "frame_out_of_range"
        and not second.get("ok")
        and second.get("error", {}).get("code") == "unknown_bone"
    )
    return 0 if ok else 1


def step_disconnect(args):
    request = wave_request(args.object)
    request["id"] = "acc-cut"
    payload = json.dumps(request).encode("utf-8")
    with socket.create_connection(("127.0.0.1", args.port), timeout=10) as conn:
        conn.sendall(payload[: len(payload) // 2])
        # Close with the request half-sent and no newline ever written.
    print(json.dumps({"id": "acc-cut", "sent_bytes": len(payload) // 2, "closed": True}))
    return 0


def step_shape_keys(args):
    with open(args.artifact, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    request = build_request(
        artifact, object_override=args.object, request_id="acc-speech"
    )
    response = send(request, args.port)
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


STEPS = {
    "inspect": step_inspect,
    "wave": step_wave,
    "invalid": step_invalid,
    "disconnect": step_disconnect,
    "shape_keys": step_shape_keys,
}


def main():
    parser = argparse.ArgumentParser(description="Animus bridge acceptance client")
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--object", default="KVRC")
    parser.add_argument("--artifact", default=None, help="viseme take JSON path")
    parser.add_argument("--request-id", dest="request_id", default="acc-wave")
    args = parser.parse_args()
    return STEPS[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())

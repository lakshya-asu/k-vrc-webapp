"""Minimal client for the Animus bridge wire protocol.

Newline-delimited JSON over a localhost socket, one connection per
request, mirroring the Node director runner and the acceptance clients.
The bridge validates everything; this client adds nothing to the
payload.
"""

import json
import socket
import time


def send_request(request, host="127.0.0.1", port=8765, timeout=60.0):
    """Send one request, return the parsed response object."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk
    line, _, _ = buffer.partition(b"\n")
    if not line.strip():
        raise RuntimeError(
            f"bridge closed the connection without answering request "
            f"'{request.get('id')}'"
        )
    return json.loads(line.decode("utf-8"))


def wait_for_bridge(host, port, deadline_s=120.0, interval_s=0.25):
    """Poll until the bridge socket accepts connections. Returns True/False."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(interval_s)
    return False


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

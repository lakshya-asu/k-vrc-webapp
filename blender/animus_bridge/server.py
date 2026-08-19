"""Localhost socket server for the Animus Blender bridge.

Transport rules:

- Bind 127.0.0.1 only. This is a local code boundary, not a network API.
- Requests and responses are newline-delimited JSON.
- Socket threads read, parse, and structurally validate. They enqueue
  validated work and send refusals. They never import or touch bpy.
- A partial line at disconnect is discarded. It is never parsed and
  never enqueued, so it can never become a scene mutation.
- A fully received request is executed even if the client disconnects
  before the reply. The mutation applies fully; only the receipt is lost.
"""

import json
import socket
import threading

from . import protocol
from .executor import WorkItem


class BridgeServer:
    def __init__(self, executor, host=protocol.DEFAULT_HOST, port=protocol.DEFAULT_PORT):
        self.executor = executor
        self.host = host
        self.port = port
        self._listener = None
        self._accept_thread = None
        self._running = False
        self._client_threads = []

    def start(self):
        if self._running:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(4)
        self.port = listener.getsockname()[1]
        self._listener = listener
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="animus-bridge-accept", daemon=True
        )
        self._accept_thread.start()

    def stop(self):
        self._running = False
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        for thread in list(self._client_threads):
            thread.join(timeout=1.0)
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

    def _accept_loop(self):
        while self._running:
            try:
                conn, _addr = self._listener.accept()
            except OSError:
                break
            thread = threading.Thread(
                target=self._client_loop,
                args=(conn,),
                name="animus-bridge-client",
                daemon=True,
            )
            self._client_threads.append(thread)
            thread.start()

    def _client_loop(self, conn):
        send_lock = threading.Lock()

        def reply(response):
            payload = (json.dumps(response) + "\n").encode("utf-8")
            try:
                with send_lock:
                    conn.sendall(payload)
            except OSError:
                pass

        buffer = b""
        try:
            while self._running:
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > protocol.MAX_LINE_BYTES:
                    reply(
                        protocol.failure(
                            "unknown",
                            "unknown",
                            protocol.Refusal(
                                protocol.BAD_REQUEST, "request line too long"
                            ),
                        )
                    )
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self._handle_line(line, reply)
            # Anything left in buffer is an incomplete line. Discard it.
            # It was never parsed, never queued, and never reaches bpy.
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handle_line(self, line, reply):
        """Runs on a socket thread. Parse, validate, enqueue. No bpy."""
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            reply(
                protocol.failure(
                    "unknown",
                    "unknown",
                    protocol.Refusal(protocol.BAD_JSON, f"invalid JSON: {error}"),
                )
            )
            return

        request_id = "unknown"
        op = "unknown"
        try:
            request_id, op, raw_params = protocol.validate_envelope(obj)
            params = protocol.validate_params(op, raw_params)
        except protocol.Refusal as refusal:
            if isinstance(obj, dict):
                candidate_id = obj.get("id")
                if isinstance(candidate_id, str) and candidate_id.strip():
                    request_id = candidate_id[:120]
                candidate_op = obj.get("op")
                if isinstance(candidate_op, str):
                    op = candidate_op[:60]
            reply(protocol.failure(request_id, op, refusal))
            return

        self.executor.submit(
            WorkItem({"id": request_id, "op": op, "params": params}, reply)
        )

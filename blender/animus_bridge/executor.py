"""Main-thread executor for queued bridge work.

Socket threads enqueue WorkItem objects and nothing else. A callback
registered through bpy.app.timers drains the queue on Blender's main
thread. That callback is the only code path that reaches operations.py,
so every bpy access happens on the main thread. This follows Blender's
threading rules and the pattern documented in the Animus research brief.
"""

import queue

import bpy

from . import operations
from .protocol import INTERNAL_ERROR, Refusal, failure, success


class WorkItem:
    """One fully validated request plus a way to answer it.

    The request dict holds 'id', 'op', and structurally validated
    'params'. reply is a callable taking the response dict. A reply
    failure (client already gone) is swallowed; the scene result stands.
    """

    __slots__ = ("request", "reply")

    def __init__(self, request, reply):
        self.request = request
        self.reply = reply


class Executor:
    def __init__(self, interval=0.05, max_items_per_tick=8):
        self.interval = interval
        self.max_items_per_tick = max_items_per_tick
        self.queue = queue.Queue()
        self._stopped = False
        self._installed = False

    def submit(self, item):
        """Called from socket threads. Touches only the queue, never bpy."""
        self.queue.put(item)

    def install(self):
        if self._installed:
            return
        self._stopped = False
        self._installed = True
        bpy.app.timers.register(self.drain, first_interval=self.interval)

    def stop(self):
        """The next drain tick unregisters the timer by returning None."""
        self._stopped = True

    def drain(self):
        """Timer callback. Runs on Blender's main thread."""
        if self._stopped:
            self._installed = False
            return None
        for _ in range(self.max_items_per_tick):
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            response = self.execute(item.request)
            try:
                item.reply(response)
            except Exception:
                pass
        return self.interval

    def execute(self, request):
        request_id = request["id"]
        op = request["op"]
        handler = operations.HANDLERS[op]
        try:
            result = handler(request["params"])
        except Refusal as refusal:
            return failure(request_id, op, refusal)
        except Exception as error:
            return failure(
                request_id,
                op,
                Refusal(INTERNAL_ERROR, f"unexpected failure: {error}"),
            )
        return success(request_id, op, result)

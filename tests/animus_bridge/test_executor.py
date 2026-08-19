"""Queue semantics: submit from any thread, execute only on the drain."""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import fake_bpy  # noqa: E402
from animus_bridge import protocol  # noqa: E402
from animus_bridge.executor import Executor, WorkItem  # noqa: E402

BONES = ["root", "arm.R", "hand.R"]


def take_request(request_id="r1"):
    return {
        "id": request_id,
        "op": "perform_take",
        "params": protocol.validate_params(
            "perform_take",
            {
                "object": "KVRC",
                "name_hint": "wave",
                "frame_start": 1,
                "frame_end": 48,
                "samples": [
                    {"bone": "arm.R", "frame": 1, "rotation_quaternion": [1, 0, 0, 0]},
                    {"bone": "arm.R", "frame": 48, "rotation_quaternion": [0.9, 0, 0, 0.4]},
                ],
            },
        ),
    }


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        fake_bpy.reset()
        fake_bpy.add_armature_object("KVRC", BONES)
        self.executor = Executor()
        self.responses = []

    def submit(self, request):
        self.executor.submit(WorkItem(request, self.responses.append))

    def test_install_registers_timer(self):
        self.executor.install()
        self.assertTrue(fake_bpy.app.timers.is_registered(self.executor.drain))

    def test_nothing_mutates_before_drain(self):
        self.executor.install()
        self.submit(take_request())
        self.assertEqual(len(fake_bpy.data.actions), 0)
        self.assertEqual(self.responses, [])

    def test_drain_executes_and_replies(self):
        self.executor.install()
        self.submit(take_request())
        fake_bpy.app.timers.pump()
        self.assertEqual(len(self.responses), 1)
        response = self.responses[0]
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["action"], "ANIMUS_wave_take001")
        self.assertEqual(len(fake_bpy.data.actions), 1)

    def test_submit_from_worker_thread_touches_no_bpy(self):
        self.executor.install()
        worker = threading.Thread(target=self.submit, args=(take_request(),))
        worker.start()
        worker.join()
        worker_ident = worker.ident
        self.assertEqual(fake_bpy.ACCESS_THREADS, set())
        fake_bpy.app.timers.pump()
        self.assertEqual(fake_bpy.ACCESS_THREADS, {threading.get_ident()})
        self.assertNotIn(worker_ident, fake_bpy.ACCESS_THREADS)

    def test_refusal_becomes_typed_response(self):
        self.executor.install()
        request = take_request()
        request["params"]["object"] = "Ghost"
        self.submit(request)
        fake_bpy.app.timers.pump()
        response = self.responses[0]
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], protocol.UNKNOWN_OBJECT)
        self.assertEqual(len(fake_bpy.data.actions), 0)

    def test_reply_failure_does_not_undo_the_mutation(self):
        self.executor.install()

        def broken_reply(_response):
            raise OSError("client is gone")

        self.executor.submit(WorkItem(take_request(), broken_reply))
        fake_bpy.app.timers.pump()
        self.assertEqual(len(fake_bpy.data.actions), 1)

    def test_stop_unregisters_timer(self):
        self.executor.install()
        self.executor.stop()
        fake_bpy.app.timers.pump()
        self.assertFalse(fake_bpy.app.timers.is_registered(self.executor.drain))

    def test_fifo_order(self):
        self.executor.install()
        self.submit(take_request("a"))
        self.submit(take_request("b"))
        fake_bpy.app.timers.pump()
        self.assertEqual([r["id"] for r in self.responses], ["a", "b"])
        names = {r["result"]["action"] for r in self.responses}
        self.assertEqual(names, {"ANIMUS_wave_take001", "ANIMUS_wave_take002"})


if __name__ == "__main__":
    unittest.main()

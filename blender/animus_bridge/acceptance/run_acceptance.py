"""Real-Blender acceptance for the Animus bridge. Headless only.

Run from the repo root:

    blender --background --factory-startup \
        --python blender/animus_bridge/acceptance/run_acceptance.py

Environment:
    ANIMUS_ACCEPTANCE_PORT      socket port (default 8791)
    ANIMUS_VOICE_ARTIFACT       path to a viseme take JSON for the
                                shape-key step (required)
    ANIMUS_ACCEPTANCE_OUT       where to write the results JSON
                                (default: results.json next to this file)

How this differs from the GUI wiring, and why it is still a fair test:

- In --background, Blender runs the --python script and exits; the
  event loop that fires bpy.app.timers never runs. The add-on's
  register() detects this and refuses to start the server (by design).
  The harness therefore calls start_bridge() explicitly and drains the
  executor queue ITSELF on the main thread while it waits for the
  client process. The threading contract is unchanged: socket threads
  parse and enqueue, only the main thread touches bpy. The drain call
  is the same Executor.drain the timer would invoke.
- The scene is built directly through bpy data and operator calls (a
  plain armature with named bones plus a shape-keyed mesh). Rigify is
  not used: generating a Rigify rig headless needs the addon enabled
  plus a metarig generate step, and P1 only requires named bones
  (manual_install.md allows exactly this substitution).
- The client runs as a separate OS process started with Blender's own
  bundled Python (sys.executable), so requests cross a real socket
  between two processes.

The harness checks the acceptance invariants INSIDE Blender after each
step and prints one machine-readable JSON block between the markers
ANIMUS_ACCEPTANCE_RESULTS_BEGIN / ANIMUS_ACCEPTANCE_RESULTS_END.
Exit code 0 only if every check passed.
"""

import json
import os
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER_DIR = os.path.dirname(os.path.dirname(HERE))
for entry in (BLENDER_DIR, HERE):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import animus_bridge  # noqa: E402

PORT = int(os.environ.get("ANIMUS_ACCEPTANCE_PORT", "8791"))
ARTIFACT = os.environ.get("ANIMUS_VOICE_ARTIFACT", "")
OUT_PATH = os.environ.get(
    "ANIMUS_ACCEPTANCE_OUT", os.path.join(HERE, "results.json")
)
CLIENT = os.path.join(HERE, "client_steps.py")

from biped import BIPED_BONE_NAMES, build_biped_armature  # noqa: E402

BONES = list(BIPED_BONE_NAMES)
SHAPE_KEYS = ["mouth_open", "smile_width", "mouth_curl_left", "mouth_curl_right"]

RESULTS = {"steps": [], "checks": [], "environment": {}}


def check(name, passed, detail=""):
    RESULTS["checks"].append(
        {"name": name, "passed": bool(passed), "detail": str(detail)}
    )
    status = "PASS" if passed else "FAIL"
    print(f"[acceptance] {status} {name} {detail}")
    return bool(passed)


def build_scene():
    obj = build_biped_armature("KVRC")

    mesh = bpy.data.meshes.new("KVRC_face_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)]
    )
    face = bpy.data.objects.new("KVRC_face", mesh)
    bpy.context.scene.collection.objects.link(face)
    face.shape_key_add(name="Basis")
    for name in SHAPE_KEYS:
        face.shape_key_add(name=name)
    return obj, face


def strip_names(animation_data):
    if animation_data is None:
        return []
    return sorted(
        strip.name for track in animation_data.nla_tracks for strip in track.strips
    )


def snapshot():
    kvrc = bpy.data.objects["KVRC"]
    key = bpy.data.objects["KVRC_face"].data.shape_keys
    return {
        "actions": sorted(action.name for action in bpy.data.actions),
        "kvrc_strips": strip_names(kvrc.animation_data),
        "face_strips": strip_names(key.animation_data),
    }


def drain():
    executor = animus_bridge._executor
    if executor is not None:
        executor.drain()


def run_client(step, extra=None):
    command = [sys.executable, CLIENT, "--step", step, "--port", str(PORT)]
    command += extra or []
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + 60.0
    while process.poll() is None:
        drain()
        if time.monotonic() > deadline:
            process.kill()
            break
        time.sleep(0.01)
    # A request fully received right at client exit still executes.
    for _ in range(20):
        drain()
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    responses = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                responses.append(json.loads(line))
            except ValueError:
                pass
    record = {
        "step": step,
        "exit_code": process.returncode,
        "responses": responses,
        "stderr": stderr.strip()[-500:],
    }
    RESULTS["steps"].append(record)
    return record


def action_api_report(action_name):
    """Record how a bridge Action looks under the 4.4+ slotted model."""
    action = bpy.data.actions.get(action_name)
    if action is None:
        return {"found": False}
    report = {
        "found": True,
        "fcurve_count": len(action.fcurves),
        "fcurve_paths": sorted({fc.data_path for fc in action.fcurves}),
    }
    slots = getattr(action, "slots", None)
    if slots is not None:
        report["slot_count"] = len(slots)
        report["slots"] = [
            {
                "identifier": getattr(slot, "identifier", None),
                "target_id_type": getattr(slot, "target_id_type", None),
            }
            for slot in slots
        ]
    return report


def main():
    RESULTS["environment"] = {
        "blender_version": bpy.app.version_string,
        "background": bpy.app.background,
        "python": sys.version.split()[0],
        "port": PORT,
        "artifact": ARTIFACT,
        "rig_note": "plain bpy armature with named bones; Rigify not used headless",
    }

    build_scene()
    check("scene_built", "KVRC" in bpy.data.objects and "KVRC_face" in bpy.data.objects)

    # register() must refuse to start the server in background mode.
    animus_bridge.register()
    check(
        "register_guard_in_background",
        animus_bridge._server is None,
        "register() started no server under --background, as designed",
    )
    server = animus_bridge.start_bridge(port=PORT)
    check("server_started", server is not None and server.port == PORT, f"port {server.port}")

    base = snapshot()
    check("scene_starts_clean", base == {"actions": [], "kvrc_strips": [], "face_strips": []}, base)

    # Step 1: inspect_rig.
    record = run_client("inspect")
    response = record["responses"][0] if record["responses"] else {}
    result = response.get("result", {})
    check("inspect_ok", response.get("ok") is True, response.get("error"))
    check("inspect_bones_match", result.get("bones") == sorted(BONES), result.get("bones"))
    check("inspect_mutated_nothing", snapshot() == base)

    # Step 2: one wave perform_take.
    before = snapshot()
    record = run_client("wave", ["--request-id", "acc-wave-1"])
    response = record["responses"][0] if record["responses"] else {}
    check("wave_ok", response.get("ok") is True, response.get("error"))
    receipt_one = response.get("result", {})
    after_one = snapshot()
    new_actions = sorted(set(after_one["actions"]) - set(before["actions"]))
    new_strips = sorted(set(after_one["kvrc_strips"]) - set(before["kvrc_strips"]))
    check("wave_one_new_action", len(new_actions) == 1, new_actions)
    check("wave_one_new_strip", len(new_strips) == 1, new_strips)
    check("wave_receipt_names_action", receipt_one.get("action") in new_actions, receipt_one)
    check("wave_face_untouched", after_one["face_strips"] == before["face_strips"])
    RESULTS["wave_action_api"] = action_api_report(receipt_one.get("action", ""))

    # Step 3: repeat take gets fresh names, overwrites nothing.
    record = run_client("wave", ["--request-id", "acc-wave-2"])
    response = record["responses"][0] if record["responses"] else {}
    check("repeat_ok", response.get("ok") is True, response.get("error"))
    receipt_two = response.get("result", {})
    after_two = snapshot()
    check(
        "repeat_new_names",
        receipt_two.get("action") != receipt_one.get("action")
        and receipt_two.get("strip") != receipt_one.get("strip"),
        {"first": receipt_one.get("action"), "second": receipt_two.get("action")},
    )
    check(
        "repeat_added_exactly_one_of_each",
        len(after_two["actions"]) == len(after_one["actions"]) + 1
        and len(after_two["kvrc_strips"]) == len(after_one["kvrc_strips"]) + 1,
        after_two,
    )
    check(
        "repeat_kept_first_take",
        receipt_one.get("action") in after_two["actions"],
    )

    # Step 4: invalid payloads are refused and mutate nothing.
    before = snapshot()
    record = run_client("invalid")
    check("invalid_refused", record["exit_code"] == 0, record["responses"])
    check("invalid_mutated_nothing", snapshot() == before)

    # Step 5: disconnect mid-request leaves no partial artifact.
    before = snapshot()
    record = run_client("disconnect")
    check("disconnect_client_ran", record["exit_code"] == 0, record["responses"])
    check("disconnect_mutated_nothing", snapshot() == before)

    # Step 6: apply_shape_keys from the real voice artifact.
    check("artifact_present", bool(ARTIFACT) and os.path.exists(ARTIFACT), ARTIFACT)
    before = snapshot()
    record = run_client(
        "shape_keys", ["--object", "KVRC_face", "--artifact", ARTIFACT]
    )
    response = record["responses"][0] if record["responses"] else {}
    check("shape_keys_ok", response.get("ok") is True, response.get("error"))
    receipt_face = response.get("result", {})
    after_face = snapshot()
    new_actions = sorted(set(after_face["actions"]) - set(before["actions"]))
    new_face_strips = sorted(set(after_face["face_strips"]) - set(before["face_strips"]))
    check("shape_keys_one_new_action", len(new_actions) == 1, new_actions)
    check("shape_keys_one_new_face_strip", len(new_face_strips) == 1, new_face_strips)
    check("shape_keys_receipt_names_action", receipt_face.get("action") in new_actions, receipt_face)
    check("shape_keys_kvrc_untouched", after_face["kvrc_strips"] == before["kvrc_strips"])
    RESULTS["face_action_api"] = action_api_report(receipt_face.get("action", ""))

    # Verify weights actually landed: evaluate one fcurve inside Blender.
    face_action = bpy.data.actions.get(receipt_face.get("action", ""))
    if face_action is not None and len(face_action.fcurves):
        curve = face_action.fcurves[0]
        check(
            "shape_keys_fcurves_have_keyframes",
            len(curve.keyframe_points) > 0,
            f"{curve.data_path} has {len(curve.keyframe_points)} keys",
        )
    else:
        check("shape_keys_fcurves_have_keyframes", False, "no fcurves found")

    animus_bridge.stop_bridge()

    RESULTS["final_scene"] = snapshot()
    RESULTS["passed"] = all(item["passed"] for item in RESULTS["checks"])
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, indent=2)
    print("ANIMUS_ACCEPTANCE_RESULTS_BEGIN")
    print(json.dumps(RESULTS, indent=2))
    print("ANIMUS_ACCEPTANCE_RESULTS_END")
    return 0 if RESULTS["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

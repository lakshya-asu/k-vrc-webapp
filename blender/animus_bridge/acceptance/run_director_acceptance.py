"""Director-loop acceptance: one script line becomes performed takes.

Headless only. Run from the repo root:

    blender --background --factory-startup \
        --python blender/animus_bridge/acceptance/run_director_acceptance.py

Environment:
    ANIMUS_DIRECTOR_PORT        bridge socket port (default 8792)
    ANIMUS_DIRECTOR_LIVE        "1" uses the local LLM on port 8081;
                                anything else runs --fallback-only
    ANIMUS_DIRECTOR_VOICE_MODE  live | convert (default live)
    ANIMUS_DIRECTOR_OUT         results JSON path (default
                                results-director.json next to this file)
    ANIMUS_DIRECTOR_DEADLINE    seconds per director run (default 420)

The scene, threading, and drain mechanics are identical to
run_acceptance.py (imported from it). The client here is the actual
director runner, scripts/animus-director.mjs, spawned as a separate
Node process, so the whole loop crosses real process and socket
boundaries: plan -> contract -> embodiment mapping -> voice pipeline ->
typed bridge requests.

Three runs, each checked inside Blender:

    suggest   authority stays with the caller: no socket, no mutation
    perform   every mapped layer lands as its own Action + NLA strip,
              including the voiced mouth take from the voice pipeline
    fallback  an invalid provider plan is rejected by the contract and
              the deterministic fallback still performs
"""

import json
import os
import shutil
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER_DIR = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(BLENDER_DIR)
for entry in (BLENDER_DIR, HERE):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import animus_bridge  # noqa: E402
from run_acceptance import build_scene, strip_names  # noqa: E402

PORT = int(os.environ.get("ANIMUS_DIRECTOR_PORT", "8792"))
LIVE = os.environ.get("ANIMUS_DIRECTOR_LIVE", "") == "1"
VOICE_MODE = os.environ.get("ANIMUS_DIRECTOR_VOICE_MODE", "live")
OUT_PATH = os.environ.get(
    "ANIMUS_DIRECTOR_OUT", os.path.join(HERE, "results-director.json")
)
DEADLINE = float(os.environ.get("ANIMUS_DIRECTOR_DEADLINE", "420"))
DIRECTOR = os.path.join(REPO, "scripts", "animus-director.mjs")
LINE = "Wave to the viewer and say hello."
SPEECH = "Hello, I am K-VRC."

RESULTS = {"runs": [], "checks": [], "environment": {}}


def check(name, passed, detail=""):
    RESULTS["checks"].append(
        {"name": name, "passed": bool(passed), "detail": str(detail)}
    )
    status = "PASS" if passed else "FAIL"
    print(f"[director-acceptance] {status} {name} {detail}")
    return bool(passed)


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


def run_director(label, extra_args):
    node = shutil.which("node")
    command = [
        node, DIRECTOR,
        "--line", LINE,
        "--speech", SPEECH,
        "--profile", os.path.join(REPO, "src", "animus", "embodiment",
                                  "kvrc-testrig.profile.json"),
        "--port", str(PORT),
    ] + extra_args
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=REPO,
    )
    deadline = time.monotonic() + DEADLINE
    while process.poll() is None:
        drain()
        if time.monotonic() > deadline:
            process.kill()
            break
        time.sleep(0.01)
    for _ in range(20):
        drain()
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    summary = None
    text = stdout.strip()
    if text.startswith("{"):
        try:
            summary = json.loads(text)
        except ValueError:
            pass
    record = {
        "run": label,
        "exit_code": process.returncode,
        "summary": summary,
        "stderr": stderr.strip()[-800:],
    }
    RESULTS["runs"].append(record)
    return record


def load_receipt(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def keyed_action_names(names):
    report = {}
    for name in names:
        action = bpy.data.actions.get(name)
        if action is None:
            report[name] = "missing"
            continue
        keys = sum(len(fc.keyframe_points) for fc in action.fcurves)
        report[name] = keys
    return report


def main():
    RESULTS["environment"] = {
        "blender_version": bpy.app.version_string,
        "background": bpy.app.background,
        "port": PORT,
        "live_llm": LIVE,
        "voice_mode": VOICE_MODE,
        "node": shutil.which("node"),
        "line": LINE,
    }

    build_scene()
    check("scene_built", "KVRC" in bpy.data.objects and "KVRC_face" in bpy.data.objects)
    animus_bridge.register()
    server = animus_bridge.start_bridge(port=PORT)
    check("server_started", server is not None and server.port == PORT)

    plan_source = [] if LIVE else ["--fallback-only"]

    # Run 1: suggest maps the plan but never opens the socket.
    before = snapshot()
    record = run_director(
        "suggest",
        plan_source + ["--control-level", "suggest", "--voice-mode", "skip"],
    )
    check("suggest_exit_0", record["exit_code"] == 0, record["stderr"])
    check("suggest_mutated_nothing", snapshot() == before, snapshot())
    summary = record["summary"] or {}
    check("suggest_not_performed", summary.get("performed") is False, summary)

    # Run 2: perform. Every mapped layer must land, including the
    # voiced mouth take built live by the voice pipeline.
    receipt_path = os.path.join(HERE, "director-receipt-perform.json")
    before = snapshot()
    record = run_director(
        "perform",
        plan_source + [
            "--control-level", "perform",
            "--voice-mode", VOICE_MODE,
            "--receipt", receipt_path,
        ],
    )
    check("perform_exit_0", record["exit_code"] == 0, record["stderr"])
    receipt = load_receipt(receipt_path)
    check("perform_receipt_written", receipt is not None, receipt_path)
    after = snapshot()

    if receipt is not None:
        layers = receipt.get("layers", [])
        channels = [layer.get("channel") for layer in layers]
        pose_layers = [c for c in channels if c in ("body", "gaze")]
        face_layers = [c for c in channels if c in ("face", "speech")]
        check("perform_has_pose_layer", len(pose_layers) >= 1, channels)
        check("perform_has_voiced_take", "speech" in channels, channels)
        check(
            "perform_every_layer_ok",
            all(layer.get("response", {}).get("ok") is True for layer in layers),
            [layer.get("response") for layer in layers],
        )
        new_actions = sorted(set(after["actions"]) - set(before["actions"]))
        new_kvrc = sorted(set(after["kvrc_strips"]) - set(before["kvrc_strips"]))
        new_face = sorted(set(after["face_strips"]) - set(before["face_strips"]))
        check(
            "perform_one_action_per_layer",
            len(new_actions) == len(layers),
            {"layers": len(layers), "new_actions": new_actions},
        )
        check("perform_pose_strip_counts", len(new_kvrc) == len(pose_layers), new_kvrc)
        check("perform_face_strip_counts", len(new_face) == len(face_layers), new_face)
        receipt_actions = [
            layer.get("response", {}).get("result", {}).get("action")
            for layer in layers
        ]
        check(
            "perform_receipts_name_real_actions",
            all(name in new_actions for name in receipt_actions),
            receipt_actions,
        )
        keyed = keyed_action_names(receipt_actions)
        check(
            "perform_actions_have_keyframes",
            all(isinstance(keys, int) and keys > 0 for keys in keyed.values()),
            keyed,
        )
        provenance = receipt.get("plan", {}).get("provenance", {})
        check(
            "perform_provenance_recorded",
            isinstance(provenance.get("operator"), str)
            and isinstance(provenance.get("fallback"), bool),
            provenance,
        )
        RESULTS["perform_provenance"] = provenance
        RESULTS["perform_actions"] = keyed

    # Run 3: an invalid provider plan must fall back deterministically
    # and still perform.
    receipt_path = os.path.join(HERE, "director-receipt-fallback.json")
    before = snapshot()
    record = run_director(
        "fallback-from-invalid-plan",
        [
            "--provider", "stub-invalid",
            "--control-level", "perform",
            "--voice-mode", "skip",
            "--receipt", receipt_path,
        ],
    )
    check("fallback_exit_0", record["exit_code"] == 0, record["stderr"])
    receipt = load_receipt(receipt_path)
    check("fallback_receipt_written", receipt is not None, receipt_path)
    if receipt is not None:
        provenance = receipt.get("plan", {}).get("provenance", {})
        failures = provenance.get("prior_failures", [])
        check("fallback_used_deterministic", provenance.get("operator") == "deterministic", provenance)
        check("fallback_flag_true", provenance.get("fallback") is True, provenance)
        check(
            "fallback_names_rejected_provider",
            bool(failures) and failures[0].get("provider") == "stub-invalid",
            failures,
        )
        after = snapshot()
        new_actions = sorted(set(after["actions"]) - set(before["actions"]))
        check(
            "fallback_still_performed",
            len(new_actions) == len(receipt.get("layers", [])) and new_actions,
            new_actions,
        )

    animus_bridge.stop_bridge()
    RESULTS["final_scene"] = snapshot()
    RESULTS["passed"] = all(item["passed"] for item in RESULTS["checks"])
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, indent=2)
    print("ANIMUS_DIRECTOR_ACCEPTANCE_RESULTS_BEGIN")
    print(json.dumps(RESULTS, indent=2))
    print("ANIMUS_DIRECTOR_ACCEPTANCE_RESULTS_END")
    return 0 if RESULTS["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

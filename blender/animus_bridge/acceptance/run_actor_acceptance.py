"""Actor-loop acceptance: python -m animus_actor performs a full take.

Headless only. Run from the repo root:

    blender --background --factory-startup \
        --python blender/animus_bridge/acceptance/run_actor_acceptance.py

Environment:
    ANIMUS_ACTOR_PORT        bridge socket port (default 8793)
    ANIMUS_ACTOR_VOICE_MODE  live | convert (default live)
    ANIMUS_ACTOR_OUT         results JSON path (default
                             results-actor.json next to this file)
    ANIMUS_ACTOR_DEADLINE    seconds per client run (default 420)

The scene, threading, and drain mechanics are identical to
run_acceptance.py (imported from it). The client is the Python actor
loop itself, python -m animus_actor, spawned with Blender's own bundled
interpreter in --attach mode, so the loop crosses a real process and
socket boundary: line -> deterministic plan -> contract -> embodiment
mapping -> voice pipeline -> typed bridge requests -> receipts.

Two runs, each checked inside Blender:

    suggest   authority stays with the caller: no socket, no mutation
    perform   every mapped layer lands as its own Action + NLA strip,
              including the voiced mouth take from the voice pipeline
"""

import json
import os
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

PORT = int(os.environ.get("ANIMUS_ACTOR_PORT", "8793"))
VOICE_MODE = os.environ.get("ANIMUS_ACTOR_VOICE_MODE", "live")
OUT_PATH = os.environ.get(
    "ANIMUS_ACTOR_OUT", os.path.join(HERE, "results-actor.json")
)
DEADLINE = float(os.environ.get("ANIMUS_ACTOR_DEADLINE", "420"))
LINE = "Hello, I am K-VRC"

RESULTS = {"runs": [], "checks": [], "environment": {}}


def check(name, passed, detail=""):
    RESULTS["checks"].append(
        {"name": name, "passed": bool(passed), "detail": str(detail)}
    )
    status = "PASS" if passed else "FAIL"
    print(f"[actor-acceptance] {status} {name} {detail}")
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


def run_actor(label, extra_args):
    command = [
        sys.executable, "-m", "animus_actor", LINE,
        "--attach", "--host", "127.0.0.1", "--port", str(PORT),
        "--out", os.path.join(REPO, "voice", "animus_voice", "out", "actor"),
    ] + extra_args
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=REPO, env=env,
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
    start = stdout.find("{")
    if start >= 0:
        try:
            summary = json.loads(stdout[start:])
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
        report[name] = sum(len(fc.keyframe_points) for fc in action.fcurves)
    return report


def main():
    RESULTS["environment"] = {
        "blender_version": bpy.app.version_string,
        "background": bpy.app.background,
        "python": sys.version.split()[0],
        "port": PORT,
        "voice_mode": VOICE_MODE,
        "line": LINE,
    }

    build_scene()
    check("actor_scene_built", "KVRC" in bpy.data.objects and "KVRC_face" in bpy.data.objects)
    animus_bridge.register()
    server = animus_bridge.start_bridge(port=PORT)
    check("actor_server_started", server is not None and server.port == PORT)

    # Run 1: suggest maps the plan but never opens the socket.
    before = snapshot()
    record = run_actor(
        "suggest", ["--control-level", "suggest", "--voice-mode", "skip"]
    )
    check("actor_suggest_exit_0", record["exit_code"] == 0, record["stderr"])
    check("actor_suggest_mutated_nothing", snapshot() == before, snapshot())
    summary = record["summary"] or {}
    check("actor_suggest_not_performed", summary.get("performed") is False, summary)

    # Run 2: perform. One command, one full take: plan, contract,
    # embodiment, voice, and every layer landing as Action + NLA strip.
    receipt_path = os.path.join(HERE, "actor-receipt-perform.json")
    before = snapshot()
    record = run_actor(
        "perform",
        ["--control-level", "perform", "--voice-mode", VOICE_MODE,
         "--receipt", receipt_path],
    )
    check("actor_perform_exit_0", record["exit_code"] == 0, record["stderr"])
    receipt = load_receipt(receipt_path)
    check("actor_receipt_written", receipt is not None, receipt_path)
    after = snapshot()

    if receipt is not None:
        provenance = receipt.get("plan", {}).get("provenance", {})
        check(
            "actor_used_deterministic_brain",
            provenance.get("operator") == "deterministic"
            and provenance.get("fallback") is True
            and provenance.get("model") is None,
            provenance,
        )
        layers = receipt.get("layers", [])
        channels = [layer.get("channel") for layer in layers]
        check(
            "actor_channels_complete",
            channels == ["body", "gaze", "face", "speech"],
            channels,
        )
        check(
            "actor_every_layer_ok",
            layers != [] and all(
                layer.get("response", {}).get("ok") is True for layer in layers
            ),
            [layer.get("response") for layer in layers],
        )
        new_actions = sorted(set(after["actions"]) - set(before["actions"]))
        new_kvrc = sorted(set(after["kvrc_strips"]) - set(before["kvrc_strips"]))
        new_face = sorted(set(after["face_strips"]) - set(before["face_strips"]))
        pose_layers = [c for c in channels if c in ("body", "gaze")]
        face_layers = [c for c in channels if c in ("face", "speech")]
        check(
            "actor_one_action_per_layer",
            len(new_actions) == len(layers),
            {"layers": len(layers), "new_actions": new_actions},
        )
        check("actor_pose_strip_counts", len(new_kvrc) == len(pose_layers), new_kvrc)
        check("actor_face_strip_counts", len(new_face) == len(face_layers), new_face)
        receipt_actions = [
            layer.get("response", {}).get("result", {}).get("action")
            for layer in layers
        ]
        check(
            "actor_receipts_name_real_actions",
            all(name in new_actions for name in receipt_actions),
            receipt_actions,
        )
        keyed = keyed_action_names(receipt_actions)
        check(
            "actor_actions_have_keyframes",
            all(isinstance(keys, int) and keys > 0 for keys in keyed.values()),
            keyed,
        )
        voice = receipt.get("voice", [])
        voice_track = voice[0] if voice else {}
        take_json = voice_track.get("take_json")
        wav = voice_track.get("wav")
        track_ok = bool(take_json) and os.path.exists(take_json)
        if VOICE_MODE == "live":
            track_ok = track_ok and bool(wav) and os.path.exists(wav)
        check("actor_voice_track_exists", track_ok, {"wav": wav, "take": take_json})
        check(
            "actor_viseme_count_positive",
            voice_track.get("sample_count", 0) > 0
            and voice_track.get("cue_count", 0) > 0,
            voice_track,
        )
        RESULTS["perform_actions"] = keyed
        RESULTS["perform_voice"] = {
            "wav": wav,
            "take_json": take_json,
            "sample_count": voice_track.get("sample_count"),
            "cue_count": voice_track.get("cue_count"),
        }

    # Run 3: the real K-VRC character. The GLB is imported into the same
    # scene (its baked clips stripped, exactly what the stage does) and
    # the loop performs the same line through the kvrc profile: gestures
    # sampled from the model's own clips, gaze on the Neck, face as a
    # Head pose, visemes as ear motion. No shape keys exist or are faked.
    kvrc_profile_path = os.path.join(
        REPO, "src", "animus", "embodiment", "kvrc.profile.json"
    )
    with open(kvrc_profile_path, "r", encoding="utf-8-sig") as handle:
        kvrc_profile = json.load(handle)
    bpy.ops.import_scene.gltf(
        filepath=os.path.join(REPO, "public", "models", "kvrc.glb")
    )
    for action in list(bpy.data.actions):
        if action.name not in RESULTS.get("perform_actions", {}):
            bpy.data.actions.remove(action)
    for holder in bpy.data.objects:
        if holder.animation_data is not None and holder.name == "KVRCArmature":
            holder.animation_data_clear()
    armature = bpy.data.objects.get("KVRCArmature")
    check(
        "kvrc_scene_imported",
        armature is not None
        and armature.type == "ARMATURE"
        and all(
            bone in armature.data.bones for bone in kvrc_profile["rig"]["bones"]
        ),
        sorted(kvrc_profile["rig"]["bones"]),
    )

    kvrc_receipt_path = os.path.join(HERE, "actor-receipt-kvrc.json")
    before = snapshot()
    before_kvrc = strip_names(armature.animation_data) if armature else []
    record = run_actor(
        "kvrc-perform",
        ["--control-level", "perform", "--voice-mode", VOICE_MODE,
         "--profile", kvrc_profile_path, "--receipt", kvrc_receipt_path],
    )
    check("kvrc_perform_exit_0", record["exit_code"] == 0, record["stderr"])
    kvrc_receipt = load_receipt(kvrc_receipt_path)
    after = snapshot()
    after_kvrc = strip_names(armature.animation_data) if armature else []

    if kvrc_receipt is not None:
        layers = kvrc_receipt.get("layers", [])
        channels = [layer.get("channel") for layer in layers]
        check(
            "kvrc_channels_complete",
            channels == ["body", "gaze", "face", "speech"],
            channels,
        )
        check(
            "kvrc_every_layer_ok",
            layers != [] and all(
                layer.get("response", {}).get("ok") is True for layer in layers
            ),
            [layer.get("response") for layer in layers],
        )
        new_actions = sorted(set(after["actions"]) - set(before["actions"]))
        new_strips = sorted(set(after_kvrc) - set(before_kvrc))
        check(
            "kvrc_all_layers_land_on_the_armature",
            len(new_actions) == len(layers) and len(new_strips) == len(layers)
            and after["kvrc_strips"] == before["kvrc_strips"]
            and after["face_strips"] == before["face_strips"],
            {"new_actions": new_actions, "new_strips": new_strips},
        )
        keyed = keyed_action_names(
            layer.get("response", {}).get("result", {}).get("action")
            for layer in layers
        )
        check(
            "kvrc_actions_have_keyframes",
            all(isinstance(keys, int) and keys > 0 for keys in keyed.values()),
            keyed,
        )
        RESULTS["kvrc_actions"] = keyed

    animus_bridge.stop_bridge()
    RESULTS["final_scene"] = snapshot()
    RESULTS["passed"] = all(item["passed"] for item in RESULTS["checks"])
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, indent=2)
    print("ANIMUS_ACTOR_ACCEPTANCE_RESULTS_BEGIN")
    print(json.dumps(RESULTS, indent=2))
    print("ANIMUS_ACTOR_ACCEPTANCE_RESULTS_END")
    return 0 if RESULTS["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

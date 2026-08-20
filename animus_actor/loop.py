"""The actor loop: one line of dialogue to one performed take.

    line -> deterministic plan -> contract validation -> embodiment
    mapping -> voice synthesis + visemes -> typed bridge requests ->
    receipts

Authority stays caller-owned: only control level 'perform' opens a
socket (decision A-009). The default plan source is the deterministic
fallback: no model, no GPU. --brain model asks the same local
OpenAI-compatible endpoint the Node director uses (model_brain.py);
a model plan that fails the strict contract falls back loudly, never
silently.

By default a perform run launches its own headless Blender with
animus_actor/stage.py and tears it down afterward. --attach targets a
bridge that is already listening (the acceptance harness does this).
"""

import glob
import json
import os
import shutil
import subprocess

from .bridge_client import pick_free_port, send_request, wait_for_bridge
from .contract import validate_actor_plan
from .embodiment import (
    load_profile,
    map_plan_to_bridge_jobs,
    viseme_artifact_to_pose_request,
    viseme_artifact_to_request,
)
from .face_frames import (
    build_face_job,
    face_screen_config,
    render_face_frames,
)
from .fallback import deterministic_actor_plan
from .model_brain import model_actor_plan
from .voice import run_voice_job

BRAINS = ("fallback", "model")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILE = os.path.join(
    REPO, "src", "animus", "embodiment", "kvrc-testrig.profile.json"
)
DEFAULT_OUT_DIR = os.path.join(REPO, "voice", "animus_voice", "out", "actor")
STAGE_SCRIPT = os.path.join(REPO, "animus_actor", "stage.py")
KNOWN_BLENDER = os.path.join(
    os.path.expanduser("~"), "tools", "blender-4.5", "blender.exe"
)


class ActorLoopError(RuntimeError):
    pass


def find_blender(explicit=None):
    """Blender binary: explicit arg, ANIMUS_BLENDER, PATH, known install."""
    known = sorted(
        glob.glob(os.path.join(os.path.dirname(KNOWN_BLENDER), "*", "blender.exe"))
    )
    candidates = [
        explicit,
        os.environ.get("ANIMUS_BLENDER"),
        shutil.which("blender"),
        KNOWN_BLENDER if os.path.exists(KNOWN_BLENDER) else None,
    ] + known
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
        if candidate and shutil.which(candidate):
            return candidate
    raise ActorLoopError(
        "Blender not found. Pass --blender, set ANIMUS_BLENDER, or put "
        "blender on PATH."
    )


def build_plan(line, instruction=None, speech=None, target="camera",
               actor_id="kvrc", control_level="perform", brain="fallback"):
    """Plan plus contract validation plus provenance.

    brain 'fallback' is the deterministic table; brain 'model' calls the
    director's local endpoint and keeps the deterministic plan as a loud
    fallback (model_brain.model_actor_plan owns that policy).
    """
    if brain not in BRAINS:
        raise ActorLoopError(f"unknown brain '{brain}'; choose from {BRAINS}")
    request = {
        "instruction": instruction if instruction is not None else line,
        "target": target,
        "speech": speech if speech is not None else line,
        "capabilities": {"body": True, "gaze": True, "face": True, "speech": True},
    }
    authority = {"actor_id": actor_id, "control_level": control_level}
    if brain == "model":
        return model_actor_plan(request, authority)
    candidate = deterministic_actor_plan(request)
    checked = validate_actor_plan(candidate, authority)
    if not checked["ok"]:
        raise ActorLoopError(
            "deterministic plan violated the actor contract:\n"
            + "\n".join(checked["errors"])
        )
    plan = dict(checked["value"])
    plan["provenance"] = {"operator": "deterministic", "model": None, "fallback": True}
    return plan


def _voice_layers(jobs, voice_mode, out_dir, profile):
    """Run every speech beat through the voice pipeline; return receipts."""
    voice_receipts = []
    if voice_mode == "skip":
        return voice_receipts
    speech_cfg = profile["speech"]
    for job in jobs["voice_jobs"]:
        receipt = run_voice_job(job, out_dir, voice_mode)
        if speech_cfg.get("mode") == "bone":
            request = viseme_artifact_to_pose_request(
                receipt["artifact"],
                speech_cfg,
                request_id=f"act-{job['stem']}",
                obj=job["object"],
            )
        else:
            request = viseme_artifact_to_request(
                receipt["artifact"],
                request_id=f"act-{job['stem']}",
                obj=job["object"],
            )
        jobs["layers"].append(
            {
                "beat_id": job["beat_id"],
                "channel": "speech",
                "request": request,
            }
        )
        slim = dict(receipt)
        slim.pop("artifact", None)
        voice_receipts.append(slim)
    return voice_receipts


def load_plan_file(path, actor_id="kvrc", control_level="perform"):
    """Replay a saved plan (or a full receipt) without any brain.

    The file may be a bare actor plan or an actor receipt whose 'plan'
    key holds one. The plan is re-validated against the strict contract
    with the caller's authority, and its provenance is kept with a
    'replayed_from' marker so a re-render never masquerades as a fresh
    model authorship.
    """
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    candidate = data.get("plan") if isinstance(data, dict) and "plan" in data else data
    if not isinstance(candidate, dict):
        raise ActorLoopError(f"no actor plan found in '{path}'")
    original_provenance = candidate.get("provenance") or {}
    # provenance is re-stamped below; actor_id and control_level are
    # authority-owned, the validator stamps them from the caller.
    candidate = {
        key: value
        for key, value in candidate.items()
        if key not in ("provenance", "actor_id", "control_level")
    }
    authority = {"actor_id": actor_id, "control_level": control_level}
    checked = validate_actor_plan(candidate, authority)
    if not checked["ok"]:
        raise ActorLoopError(
            f"replayed plan from '{path}' violated the actor contract:\n"
            + "\n".join(checked["errors"])
        )
    plan = dict(checked["value"])
    provenance = dict(original_provenance)
    provenance["replayed_from"] = os.path.basename(path)
    plan["provenance"] = provenance
    return plan


def _launch_stage(port, profile_path, run_dir, blender, deadline,
                  render_dir=None, render_audio=None, render_audio_start=None,
                  face_frames=None, render_engine=None, render_size=None):
    done_file = os.path.join(run_dir, "stage.done")
    report_path = os.path.join(run_dir, "stage-report.json")
    log_path = os.path.join(run_dir, "stage.log")
    for path in (done_file, report_path):
        if os.path.exists(path):
            os.remove(path)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        STAGE_SCRIPT,
        "--",
        "--port",
        str(port),
        "--profile",
        profile_path,
        "--done-file",
        done_file,
        "--report",
        report_path,
        "--deadline",
        str(deadline),
    ]
    if render_dir:
        command += ["--render-dir", render_dir]
        if render_size:
            command += ["--render-size", render_size]
        if render_audio:
            command += ["--render-audio", render_audio]
            if render_audio_start is not None:
                command += ["--render-audio-start", str(render_audio_start)]
        if face_frames:
            command += ["--face-frames", face_frames]
        if render_engine:
            command += ["--render-engine", render_engine]
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=REPO, stdout=log_handle, stderr=subprocess.STDOUT
    )
    return process, done_file, report_path, log_path, log_handle


def _stage_log_tail(log_path, limit=800):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:]
    except OSError:
        return ""


def run_actor_loop(
    line,
    instruction=None,
    speech=None,
    target="camera",
    actor_id="kvrc",
    control_level="perform",
    brain="fallback",
    profile_path=DEFAULT_PROFILE,
    voice_mode="live",
    out_dir=DEFAULT_OUT_DIR,
    receipt_path=None,
    attach=False,
    host="127.0.0.1",
    port=None,
    blender=None,
    stage_deadline=300.0,
    sender=None,
    render_dir=None,
    render_size=None,
    plan_file=None,
):
    """Run the whole chain. Returns the full receipt dict.

    sender: optional callable(request) -> response used instead of any
    socket or Blender process; tests route it through fake bpy.
    """
    profile = load_profile(profile_path)
    if plan_file:
        plan = load_plan_file(
            plan_file, actor_id=actor_id, control_level=control_level
        )
        if not line:
            speech_texts = [
                beat["speech"]["text"]
                for beat in plan["beats"]
                if beat.get("speech")
            ]
            line = speech_texts[0] if speech_texts else plan.get("summary", "")
    else:
        plan = build_plan(
            line,
            instruction=instruction,
            speech=speech,
            target=target,
            actor_id=actor_id,
            control_level=control_level,
            brain=brain,
        )
    jobs = map_plan_to_bridge_jobs(plan, profile)

    os.makedirs(out_dir, exist_ok=True)
    voice_receipts = _voice_layers(jobs, voice_mode, out_dir, profile)

    performed = plan["control_level"] == "perform"
    stage_report = None
    bridge_info = None
    face_report = None

    if performed and sender is not None:
        for layer in jobs["layers"]:
            layer["response"] = sender(layer["request"])
        bridge_info = {"mode": "injected-sender"}
    elif performed and attach:
        if not wait_for_bridge(host, port, deadline_s=10.0):
            raise ActorLoopError(f"no bridge listening on {host}:{port}")
        for layer in jobs["layers"]:
            layer["response"] = send_request(layer["request"], host=host, port=port)
        bridge_info = {"mode": "attach", "host": host, "port": port}
    elif performed:
        blender_bin = find_blender(blender)
        stage_port = port or pick_free_port()
        render_audio = None
        render_audio_start = None
        render_engine = None
        if render_dir:
            for item in voice_receipts:
                if item.get("wav"):
                    render_audio = item["wav"]
                    break
            # The wav must land on the speech beat's first frame, not
            # the take's: a plan whose speech starts after 0 ms would
            # otherwise mux the voice ahead of the animated mouth.
            if render_audio:
                for layer in jobs["layers"]:
                    if layer["channel"] == "speech":
                        render_audio_start = layer["request"]["params"][
                            "frame_start"
                        ]
                        break
            # The visor face: profiles with a screen object get the
            # webapp's face rendered as an image sequence first, then
            # the stage binds it as the screen's emissive texture.
            # Real materials need EEVEE, so a face take renders there.
            if face_screen_config(profile) is not None:
                face_job = build_face_job(
                    plan, profile, voice_receipts, jobs["layers"]
                )
                face_report = render_face_frames(
                    face_job, os.path.join(render_dir, "face-frames")
                )
                render_engine = "BLENDER_EEVEE_NEXT"
        process, done_file, report_path, log_path, log_handle = _launch_stage(
            stage_port, profile_path, out_dir, blender_bin, stage_deadline,
            render_dir=render_dir, render_audio=render_audio,
            render_audio_start=render_audio_start,
            face_frames=face_report["dir"] if face_report else None,
            render_engine=render_engine, render_size=render_size,
        )
        try:
            if not wait_for_bridge("127.0.0.1", stage_port, deadline_s=120.0):
                raise ActorLoopError(
                    "the stage bridge never came up on port "
                    f"{stage_port}. Stage log tail:\n{_stage_log_tail(log_path)}"
                )
            for layer in jobs["layers"]:
                layer["response"] = send_request(
                    layer["request"], host="127.0.0.1", port=stage_port
                )
        finally:
            # Always release the stage so Blender exits, even on error.
            # A render run keeps encoding after the done-file, so give it
            # a much longer exit window before killing anything.
            with open(done_file, "w", encoding="utf-8") as handle:
                handle.write("done\n")
            try:
                # 1080p EEVEE renders of the longer takes need well over
                # ten minutes; the encode keeps running after done-file.
                process.wait(timeout=2400 if render_dir else 60)
            except subprocess.TimeoutExpired:
                process.kill()
            log_handle.close()
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as handle:
                stage_report = json.load(handle)
        bridge_info = {
            "mode": "stage",
            "host": "127.0.0.1",
            "port": stage_port,
            "blender": blender_bin,
            "stage_exit_code": process.returncode,
            "stage_log": log_path,
        }

    receipt = {
        "kind": "animus_actor_receipt",
        "line": line,
        "instruction": instruction if instruction is not None else line,
        "requested_speech": speech if speech is not None else line,
        "control_level": plan["control_level"],
        "performed": performed,
        "brain": brain,
        "voice_mode": voice_mode,
        "plan": plan,
        "profile": jobs["profile"],
        "layers": jobs["layers"],
        "voice": voice_receipts,
        "bridge": bridge_info,
        "stage": stage_report,
        "face_frames": face_report,
    }
    if receipt_path:
        os.makedirs(os.path.dirname(os.path.abspath(receipt_path)), exist_ok=True)
        with open(receipt_path, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
        receipt["receipt_path"] = receipt_path
    return receipt


def summarize(receipt):
    """The compact report the CLI prints: voice track, visemes, beats, ops."""
    performed = receipt["performed"]
    layers = []
    for layer in receipt["layers"]:
        response = layer.get("response") or {}
        result = response.get("result") or {}
        layers.append(
            {
                "beat_id": layer["beat_id"],
                "channel": layer["channel"],
                "request_id": layer["request"]["id"],
                "ok": (response.get("ok") is True) if performed else None,
                "action": result.get("action"),
                "strip": result.get("strip"),
                "key_count": result.get("key_count"),
                "error": response.get("error"),
            }
        )
    voice = receipt["voice"]
    return {
        "line": receipt["line"],
        "control_level": receipt["control_level"],
        "performed": performed,
        "operator": receipt["plan"]["provenance"]["operator"],
        "fallback": receipt["plan"]["provenance"]["fallback"],
        "beats": len(receipt["plan"]["beats"]),
        "beats_executed": len(receipt["plan"]["beats"]) if performed else 0,
        "layers": layers,
        "voice_track": (voice[0].get("wav") or voice[0].get("take_json"))
        if voice
        else None,
        "viseme_count": sum(item.get("sample_count", 0) for item in voice),
        "stage_timed_out": (receipt.get("stage") or {}).get("timed_out"),
        "render": (receipt.get("stage") or {}).get("render"),
        "face_frames": (receipt.get("face_frames") or {}).get("frame_count"),
        "receipt_path": receipt.get("receipt_path"),
    }


def loop_succeeded(receipt):
    if not receipt["performed"]:
        return True
    stage = receipt.get("stage")
    if stage is not None and stage.get("timed_out"):
        return False
    return all(
        (layer.get("response") or {}).get("ok") is True
        for layer in receipt["layers"]
    )

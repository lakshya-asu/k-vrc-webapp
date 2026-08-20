"""Visor face frames for a rendered take.

Builds the compact face job (fps, frame range, face beats, viseme
samples, seed) from the same data the receipt records, then runs the
Node renderer (scripts/render-face-frames.mjs), which draws the
webapp's own face -- ported drawing code plus the verbatim expression
library -- under @napi-rs/canvas. Deterministic: one job and seed, one
set of PNG bytes; the manifest carries a sha256 per frame.

The face job is what feeds the Blender stage's screen material. The
ear-antenna speech proxy still animates; the visor is the primary
speech read.
"""

import json
import os
import shutil
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_SCRIPT = os.path.join(REPO, "scripts", "render-face-frames.mjs")

FACE_JOB_KIND = "animus_face_job"
MANIFEST_NAME = "face-frames-manifest.json"


class FaceFrameError(RuntimeError):
    pass


def find_node(explicit=None):
    """Node binary: explicit arg, ANIMUS_NODE, then PATH."""
    for candidate in (explicit, os.environ.get("ANIMUS_NODE"), shutil.which("node")):
        if candidate and (os.path.exists(candidate) or shutil.which(candidate)):
            return candidate
    raise FaceFrameError(
        "node not found. Set ANIMUS_NODE or put node on PATH; the visor "
        "face renderer runs the webapp's drawing code under Node."
    )


def face_screen_config(profile):
    """The profile's stage.face_screen section, or None.

    Only profiles that name a screen object get face frames; the test
    rig has no visor to texture.
    """
    stage = profile.get("stage") or {}
    config = stage.get("face_screen")
    if not isinstance(config, dict) or not config.get("object"):
        return None
    return config


def build_face_job(plan, profile, voice_receipts, layers, seed=None):
    """The compact, self-contained input for the face renderer.

    Everything comes from take data: face beats from the validated
    plan, viseme samples from the voice pipeline's take artifacts, the
    frame range from the mapped bridge requests, fps and seed from the
    profile. No number is invented here.
    """
    fps = profile["fps"]
    if seed is None:
        seed = int((profile.get("speech") or {}).get("seed", 0))

    face_beats = []
    for beat in plan["beats"]:
        face = beat.get("face")
        if face is not None:
            face_beats.append(
                {
                    "expression": face["expression"],
                    "intensity": face.get("intensity", 1),
                    "at_ms": beat["at_ms"],
                    "duration_ms": beat["duration_ms"],
                }
            )
        glyph = beat.get("face_glyph")
        if glyph is not None:
            # The composed-glyph channel: passed through whole; the
            # timeline runs the strict glyph validator again on render.
            face_beats.append(
                {
                    "glyph": glyph,
                    "at_ms": beat["at_ms"],
                    "duration_ms": beat["duration_ms"],
                }
            )

    viseme_samples = []
    frame_end = 1
    for receipt in voice_receipts:
        take_json = receipt.get("take_json")
        if not take_json or not os.path.exists(take_json):
            continue
        with open(take_json, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        viseme_samples.extend(artifact.get("samples") or [])
        frame_end = max(frame_end, int(artifact.get("frame_end") or 1))
    for layer in layers:
        end = ((layer.get("request") or {}).get("params") or {}).get("frame_end")
        if isinstance(end, int):
            frame_end = max(frame_end, end)

    job = {
        "kind": FACE_JOB_KIND,
        "fps": fps,
        "frame_end": frame_end,
        "face_beats": face_beats,
        "viseme_samples": viseme_samples,
        "seed": seed,
    }
    # Optional deterministic glitch windows: the library's 8-20 s glitch
    # timer can never fire inside a short take, so a profile may force
    # bursts at fixed times (stage.face_screen.force_glitches:
    # [{at_ms, duration_ms}]). The timeline validates the entries.
    screen = face_screen_config(profile) or {}
    forced = screen.get("force_glitches")
    if isinstance(forced, list) and forced:
        job["force_glitches"] = forced
    return job


def render_face_frames(job, out_dir, node=None, timeout=300):
    """Run the Node renderer on one face job. Returns a report dict."""
    os.makedirs(out_dir, exist_ok=True)
    job_path = os.path.join(out_dir, "face-job.json")
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)
        handle.write("\n")

    command = [
        find_node(node),
        RENDER_SCRIPT,
        "--receipt",
        job_path,
        "--out",
        out_dir,
        "--seed",
        str(job.get("seed", 0)),
    ]
    result = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise FaceFrameError(
            "face frame renderer failed "
            f"(exit {result.returncode}): {result.stderr[-800:]}"
        )

    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    count = manifest.get("frame_count", 0)
    frames = manifest.get("frames", [])
    if count < 1 or len(frames) != count:
        raise FaceFrameError(f"face frame manifest is inconsistent: {manifest_path}")
    for entry in frames:
        frame_path = os.path.join(out_dir, entry["file"])
        if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            raise FaceFrameError(f"face frame missing or empty: {frame_path}")

    return {
        "dir": out_dir,
        "job": job_path,
        "manifest": manifest_path,
        "frame_count": count,
        "frame_start": manifest.get("frame_start", 1),
        "frame_end": manifest.get("frame_end"),
        "seed": manifest.get("seed"),
        "moods": sorted({entry.get("mood") for entry in frames}),
    }

"""Command-line entry for the actor loop.

Examples, all from the repo root:

    # The whole chain in one command: deterministic plan, live CPU
    # voice, headless Blender launched and torn down automatically.
    python -m animus_actor "Hello, I am K-VRC"

    # Target a bridge that is already running (acceptance harness).
    python -m animus_actor "Hello, I am K-VRC" --attach --port 8791

    # Map everything but perform nothing.
    python -m animus_actor "Hello, I am K-VRC" --control-level suggest

    # No voice tools installed: rebuild visemes from the fixture cues.
    python -m animus_actor "Hello, I am K-VRC" --voice-mode convert

    # Model-authored plan from the director's local endpoint (needs the
    # llama server from scripts/start-animus-model.ps1 on port 8081).
    python -m animus_actor "Hello, I am K-VRC" --brain model
"""

import argparse
import json
import sys

from .contract import CONTROL_LEVELS
from .loop import (
    BRAINS,
    DEFAULT_OUT_DIR,
    DEFAULT_PROFILE,
    ActorLoopError,
    loop_succeeded,
    run_actor_loop,
    summarize,
)
from .voice import VOICE_MODES


def build_parser():
    parser = argparse.ArgumentParser(prog="animus_actor", description=__doc__)
    parser.add_argument(
        "line",
        nargs="?",
        default=None,
        help="the line of dialogue to perform (optional with --plan-file)",
    )
    parser.add_argument(
        "--plan-file",
        dest="plan_file",
        default=None,
        help="replay a saved plan or receipt JSON instead of asking a "
        "brain; the plan is re-validated and its provenance is kept "
        "with a replayed_from marker",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="stage direction for the plan (default: the line itself)",
    )
    parser.add_argument(
        "--speech",
        default=None,
        help="spoken text if different from the line",
    )
    parser.add_argument("--target", default="camera")
    parser.add_argument("--actor-id", dest="actor_id", default="kvrc")
    parser.add_argument(
        "--control-level",
        dest="control_level",
        default="perform",
        choices=CONTROL_LEVELS,
    )
    parser.add_argument(
        "--brain",
        default="fallback",
        choices=BRAINS,
        help="plan source: deterministic fallback (default) or the local "
        "model endpoint the director uses (ANIMUS_LLM_BASE_URL, "
        "ANIMUS_MODEL, ANIMUS_LLM_ATTEMPTS)",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--voice-mode", dest="voice_mode", default="live", choices=VOICE_MODES
    )
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument(
        "--receipt", default=None, help="write the full receipt JSON here"
    )
    parser.add_argument(
        "--attach",
        action="store_true",
        help="connect to a running bridge instead of launching Blender",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--blender", default=None, help="Blender binary for the stage"
    )
    parser.add_argument(
        "--stage-deadline", dest="stage_deadline", type=float, default=300.0
    )
    parser.add_argument(
        "--render-dir",
        dest="render_dir",
        default=None,
        help="after the take lands, render it there: animus-take.mp4 "
        "(H.264, AAC voice when a live wav exists) plus four still PNGs. "
        "Stage runs only, ignored with --attach or an injected sender.",
    )
    parser.add_argument(
        "--render-size",
        dest="render_size",
        default=None,
        help="render resolution WxH (stage default 960x540; use "
        "1920x1080 for full HD)",
    )
    parser.add_argument(
        "--render-engine",
        dest="render_engine",
        default=None,
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"),
        help="stage render engine (default: Workbench, or EEVEE Next "
        "when the profile has a visor face screen; CYCLES is the "
        "path-traced quality option, GPU when available)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.attach and args.port is None:
        print(
            json.dumps({"error": "--attach requires --port"}), file=sys.stderr
        )
        return 2
    if not args.line and not args.plan_file:
        print(
            json.dumps({"error": "a line of dialogue or --plan-file is required"}),
            file=sys.stderr,
        )
        return 2
    try:
        receipt = run_actor_loop(
            args.line,
            instruction=args.instruction,
            speech=args.speech,
            target=args.target,
            actor_id=args.actor_id,
            control_level=args.control_level,
            brain=args.brain,
            profile_path=args.profile,
            voice_mode=args.voice_mode,
            out_dir=args.out,
            receipt_path=args.receipt,
            attach=args.attach,
            host=args.host,
            port=args.port,
            blender=args.blender,
            stage_deadline=args.stage_deadline,
            render_dir=args.render_dir,
            render_size=args.render_size,
            plan_file=args.plan_file,
            render_engine=args.render_engine,
        )
    except (ActorLoopError, ValueError, RuntimeError) as error:
        print(
            json.dumps({"error": {"code": "actor_loop", "message": str(error)}}),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(summarize(receipt), indent=2))
    return 0 if loop_succeeded(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())

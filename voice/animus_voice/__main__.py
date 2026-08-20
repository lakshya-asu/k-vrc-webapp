"""Command-line entry for the Animus voice pipeline.

Examples:

    # Offline: rebuild the viseme take from a Rhubarb cue file.
    python -m animus_voice convert --cues fixtures/hello.rhubarb.json \
        --out out --stem hello

    # Live: text -> wav -> cues -> viseme take (needs Kokoro + Rhubarb).
    python -m animus_voice speak --text "Hello, I am K-VRC." \
        --out out --stem hello --voice af_heart

    # Report which live tools are installed.
    python -m animus_voice doctor
"""

import argparse
import json
import sys

from .converter import ConverterRefusal
from .pipeline import convert_only, run_pipeline


def _cmd_doctor(_args):
    from .rhubarb import find_rhubarb, rhubarb_available
    from .tts_chatterbox import tts_available as chatterbox_available
    from .tts_kokoro import tts_available
    from .tts_xtts import tts_available as xtts_available

    report = {
        "kokoro_tts": tts_available(),
        "chatterbox_tts": chatterbox_available(),
        "xtts_tts": xtts_available(),
        "rhubarb": rhubarb_available(),
        "rhubarb_path": find_rhubarb(),
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_convert(args):
    artifact, take_path = convert_only(
        args.cues,
        args.out,
        stem=args.stem,
        fps=args.fps,
        frame_start=args.frame_start,
        obj=args.object,
        name_hint=args.name_hint,
    )
    print(
        json.dumps(
            {
                "take_json": take_path,
                "cue_count": artifact["cue_count"],
                "sample_count": len(artifact["samples"]),
                "frame_start": artifact["frame_start"],
                "frame_end": artifact["frame_end"],
                "duration_ms": artifact["duration_ms"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_speak(args):
    tts_opts = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
        "device": args.device,
        "pitch_semitones": args.pitch_semitones,
        "speaker": args.speaker,
        "tempo": args.tempo,
    }
    receipt = run_pipeline(
        args.text,
        args.out,
        stem=args.stem,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        seed=args.seed,
        fps=args.fps,
        frame_start=args.frame_start,
        obj=args.object,
        name_hint=args.name_hint,
        backend=args.backend,
        tts_opts=tts_opts,
    )
    receipt = dict(receipt)
    receipt.pop("artifact", None)
    print(json.dumps(receipt, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="animus_voice", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out", help="output directory")
    common.add_argument("--stem", default="line", help="base name for output files")
    common.add_argument("--fps", type=int, default=24, help="frames per second")
    common.add_argument("--frame-start", dest="frame_start", type=int, default=1)
    common.add_argument("--object", default="KVRC", help="target object name")
    common.add_argument("--name-hint", dest="name_hint", default="animus_speech")

    p_convert = sub.add_parser(
        "convert", parents=[common], help="rebuild a viseme take from cue JSON"
    )
    p_convert.add_argument("--cues", required=True, help="Rhubarb cue JSON path")
    p_convert.set_defaults(func=_cmd_convert)

    p_speak = sub.add_parser(
        "speak", parents=[common], help="full live pipeline (needs Kokoro + Rhubarb)"
    )
    p_speak.add_argument("--text", required=True, help="line of speech to render")
    p_speak.add_argument(
        "--backend",
        default="kokoro",
        choices=("kokoro", "chatterbox", "xtts"),
        help="TTS engine: kokoro (deterministic CPU), chatterbox "
        "(expressive, built-in voice only), or xtts (built-in studio "
        "speakers, CPML non-commercial license)",
    )
    p_speak.add_argument("--voice", default="af_heart", help="kokoro voice id")
    p_speak.add_argument("--lang", default="a")
    p_speak.add_argument("--speed", type=float, default=1.0)
    p_speak.add_argument("--seed", type=int, default=0)
    p_speak.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="chatterbox emotion intensity (0.5 neutral)",
    )
    p_speak.add_argument(
        "--cfg-weight",
        dest="cfg_weight",
        type=float,
        default=0.5,
        help="chatterbox pacing; lower is slower, more deliberate",
    )
    p_speak.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="chatterbox sampling temperature",
    )
    p_speak.add_argument(
        "--device",
        default="auto",
        help="chatterbox device: auto, cuda, or cpu",
    )
    p_speak.add_argument(
        "--speaker",
        default="Torcull Diarmuid",
        help="xtts: built-in studio speaker name",
    )
    p_speak.add_argument(
        "--tempo",
        type=float,
        default=1.0,
        help="xtts: pitch-preserving time compression of the rendered "
        "wav (1.2 = 20 percent faster; ffmpeg rubberband)",
    )
    p_speak.add_argument(
        "--pitch-semitones",
        dest="pitch_semitones",
        type=float,
        default=0.0,
        help="chatterbox: formant-preserving pitch shift applied to the "
        "rendered wav (negative = deeper; ffmpeg rubberband)",
    )
    p_speak.set_defaults(func=_cmd_speak)

    p_doctor = sub.add_parser("doctor", help="report installed live tools")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConverterRefusal as refusal:
        print(json.dumps({"error": refusal.to_error()}, indent=2), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(json.dumps({"error": {"code": "tool_error", "message": str(error)}}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

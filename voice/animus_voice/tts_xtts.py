"""XTTS-v2 wrapper (coqui-tts fork), built-in studio speakers only.

Third voice backend beside Kokoro and Chatterbox. Uses ONLY the
model's built-in licensed speaker set (speaker names like "Torcull
Diarmuid"); no reference audio is accepted, so no real person's voice
is cloned or conditioned on.

LICENSE NOTE, deliberately loud: the XTTS-v2 weights are distributed
under the Coqui Public Model License (CPML), which is NON-COMMERCIAL.
Every receipt this wrapper emits carries that license string so the
restriction travels with the asset. The model owner accepted CPML for
this project knowingly (recorded on the Animus board, 2026-08-20).

Knobs:
    speaker        built-in speaker name (required)
    speed          XTTS's own pacing control
    temperature    sampling temperature (XTTS default 0.65)
    tempo          pitch-preserving time compression applied to the
                   rendered wav with ffmpeg rubberband (1.2 = 20%
                   faster, same pitch and formants). LESSON from reel
                   v2 (2026-08-20): compression at 1.15-1.25 makes the
                   voice audibly robotic. Ship at 1.0 and fit the video
                   to the audio instead; 1.05 is the ceiling for a line
                   that truly drags.

Output is a 24kHz mono WAV, same contract as the other backends. The
torch seed is set from seed for repeatability on one device/version.
"""

import os
import shutil
import subprocess
import tempfile
import wave

DEFAULT_SPEAKER = "Torcull Diarmuid"
DEFAULT_TEMPERATURE = 0.65
XTTS_LICENSE = (
    "Coqui Public Model License (CPML), NON-COMMERCIAL; "
    "accepted knowingly by the project owner 2026-08-20"
)

_MODEL = None
_MODEL_DEVICE = None


def tts_available():
    """True when coqui-tts and torch can be imported."""
    try:
        import torch  # noqa: F401
        from TTS.api import TTS  # noqa: F401
    except Exception:
        return False
    return True


def _pick_device(device):
    import torch

    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(device):
    """Load once per process; ~2GB of weights from Hugging Face."""
    global _MODEL, _MODEL_DEVICE
    if _MODEL is None or _MODEL_DEVICE != device:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        _MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        _MODEL_DEVICE = device
    return _MODEL


def tempo_compress_wav(path, tempo):
    """Pitch- and formant-preserving time compression in place."""
    if not tempo or float(tempo) == 1.0:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "tempo compression needs ffmpeg (with rubberband) on PATH"
        )
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-af",
                f"rubberband=tempo={float(tempo):.4f}",
                handle.name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg rubberband tempo failed: {result.stderr[-300:]}"
            )
        shutil.copyfile(handle.name, path)
    finally:
        os.unlink(handle.name)


def render_to_wav(
    text,
    out_wav,
    speaker=DEFAULT_SPEAKER,
    speed=1.0,
    temperature=DEFAULT_TEMPERATURE,
    tempo=1.0,
    seed=0,
    device="auto",
):
    """Render text to out_wav with a built-in speaker. Returns a receipt."""
    if not tts_available():
        raise RuntimeError(
            "XTTS is not installed in this environment; install 'coqui-tts' "
            "plus torch/torchaudio/torchcodec (the .venv-voice3 env)"
        )

    import torch

    resolved_device = _pick_device(device)
    torch.manual_seed(seed)
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = _get_model(resolved_device)
    names = model.synthesizer.tts_model.speaker_manager.speaker_names
    if speaker not in names:
        raise RuntimeError(f"'{speaker}' is not a built-in XTTS speaker")
    model.tts_to_file(
        text=text,
        speaker=speaker,
        language="en",
        file_path=out_wav,
        speed=float(speed),
        temperature=float(temperature),
    )
    tempo_compress_wav(out_wav, tempo)

    with wave.open(out_wav) as handle:
        sample_rate = handle.getframerate()
        duration_s = handle.getnframes() / float(sample_rate)

    return {
        "engine": "xtts",
        "model": "coqui/XTTS-v2",
        "license": XTTS_LICENSE,
        "voice": f"builtin:{speaker}",
        "speed": float(speed),
        "temperature": float(temperature),
        "tempo": float(tempo),
        "seed": seed,
        "device": resolved_device,
        "sample_rate": sample_rate,
        "duration_s": round(duration_s, 3),
        "wav": out_wav,
    }

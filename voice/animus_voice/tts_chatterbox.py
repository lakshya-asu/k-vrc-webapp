"""Chatterbox TTS wrapper (resemble-ai/chatterbox, MIT).

Second voice backend beside Kokoro: a 0.5B expressive TTS with built-in
emotion controls. This wrapper uses ONLY the model's built-in default
voice conditionals; it never loads reference audio, so no real person's
voice is cloned or conditioned on (audio_prompt_path is deliberately not
exposed). Expressiveness is shaped with the model's own knobs:

    exaggeration  0..1+, emotion intensity (0.5 neutral)
    cfg_weight    0..1, lower = slower, more deliberate pacing
    temperature   sampling temperature

The model runs on CUDA when available, else CPU. Sampling is stochastic,
so the torch seed is set from the seed argument for repeatability on the
same device and versions. Output is a 24kHz mono WAV, same contract as
tts_kokoro.render_to_wav. Chatterbox watermarks its audio (Perth); that
is upstream behavior and is left on.
"""

import os
import shutil
import subprocess
import tempfile
import wave

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8

_MODEL = None
_MODEL_DEVICE = None


def tts_available():
    """True when chatterbox and torch can be imported."""
    try:
        import torch  # noqa: F401
        from chatterbox.tts import ChatterboxTTS  # noqa: F401
    except Exception:
        return False
    return True


def _pick_device(device):
    import torch

    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(device):
    """Load once per process; the weights are ~2 GB from Hugging Face."""
    global _MODEL, _MODEL_DEVICE
    if _MODEL is None or _MODEL_DEVICE != device:
        from chatterbox.tts import ChatterboxTTS

        _MODEL = ChatterboxTTS.from_pretrained(device=device)
        _MODEL_DEVICE = device
    return _MODEL


def _write_wav(path, samples, sample_rate):
    """Write float samples in -1..1 to a 16-bit mono WAV using stdlib."""
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def pitch_shift_wav(path, semitones):
    """Formant-preserving pitch shift in place (ffmpeg rubberband).

    Duration is unchanged: this is a real pitch shift, not a resample
    slowdown. Raises RuntimeError when ffmpeg or its rubberband filter
    is unavailable.
    """
    if not semitones:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "pitch_semitones needs ffmpeg (with the rubberband filter) on PATH"
        )
    ratio = 2.0 ** (float(semitones) / 12.0)
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
                f"rubberband=pitch={ratio:.6f}:formant=preserved",
                handle.name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg rubberband pitch shift failed: {result.stderr[-300:]}"
            )
        shutil.copyfile(handle.name, path)
    finally:
        os.unlink(handle.name)


def render_to_wav(
    text,
    out_wav,
    exaggeration=DEFAULT_EXAGGERATION,
    cfg_weight=DEFAULT_CFG_WEIGHT,
    temperature=DEFAULT_TEMPERATURE,
    seed=0,
    device="auto",
    pitch_semitones=0.0,
):
    """Render text to out_wav with the built-in voice. Returns a receipt.

    Raises RuntimeError with a clear message when Chatterbox is missing,
    so callers can decide to skip rather than crash opaquely.
    """
    if not tts_available():
        raise RuntimeError(
            "Chatterbox TTS is not installed in this environment; "
            "install 'chatterbox-tts' (the .venv-voice2 env) to use "
            "the chatterbox backend"
        )

    import torch

    resolved_device = _pick_device(device)
    torch.manual_seed(seed)
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = _get_model(resolved_device)
    audio = model.generate(
        text,
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
        temperature=float(temperature),
    )
    samples = audio.squeeze(0).detach().cpu().numpy()
    _write_wav(out_wav, samples, model.sr)
    pitch_shift_wav(out_wav, pitch_semitones)

    duration_s = float(samples.shape[-1]) / model.sr
    return {
        "engine": "chatterbox",
        "model": "resemble-ai/chatterbox",
        "license": "MIT",
        "voice": "builtin",
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "seed": seed,
        "device": resolved_device,
        "pitch_semitones": float(pitch_semitones),
        "sample_rate": model.sr,
        "duration_s": round(duration_s, 3),
        "wav": out_wav,
    }

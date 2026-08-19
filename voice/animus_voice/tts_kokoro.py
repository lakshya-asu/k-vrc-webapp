"""Kokoro-82M TTS wrapper, CPU only.

Kokoro is Apache-2.0 (https://huggingface.co/hexgrad/Kokoro-82M). This
wrapper renders one line of text to a 24kHz mono WAV on CPU. It imports
torch and the kokoro package lazily so the converter and tests never pull
them in. When the package is missing, tts_available() returns False and
callers skip the live render cleanly.

Determinism note: Kokoro's forward pass is deterministic for a fixed
text, voice, and speed on CPU. We also set the torch seed defensively.
The same text and voice therefore render to the same audio.
"""

import wave

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "a"  # American English in Kokoro's language codes.


def tts_available():
    """True when kokoro, torch, and numpy can be imported."""
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from kokoro import KPipeline  # noqa: F401
    except Exception:
        return False
    return True


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


def render_to_wav(
    text,
    out_wav,
    voice=DEFAULT_VOICE,
    lang=DEFAULT_LANG,
    speed=1.0,
    seed=0,
):
    """Render text to out_wav on CPU. Returns a small render receipt.

    Raises RuntimeError with a clear message when Kokoro is not installed,
    so the caller can decide to skip rather than crash opaquely.
    """
    if not tts_available():
        raise RuntimeError(
            "Kokoro TTS is not installed in this environment; "
            "install 'kokoro' and CPU 'torch' to render live audio"
        )

    import numpy as np
    import torch
    from kokoro import KPipeline

    torch.manual_seed(seed)
    # Force CPU. Never touch CUDA in this proof.
    pipeline = KPipeline(lang_code=lang, device="cpu")

    chunks = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype="float32"))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the given text")

    samples = np.concatenate(chunks)
    _write_wav(out_wav, samples, SAMPLE_RATE)

    duration_s = float(len(samples)) / SAMPLE_RATE
    return {
        "engine": "kokoro",
        "model": "hexgrad/Kokoro-82M",
        "license": "Apache-2.0",
        "voice": voice,
        "lang": lang,
        "speed": speed,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration_s, 3),
        "wav": out_wav,
    }

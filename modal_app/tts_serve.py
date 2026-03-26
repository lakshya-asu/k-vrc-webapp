# modal_app/tts_serve.py
import os

# Must be set before any TTS import so coqui uses the volume path for caching.
# COQUI_TOS_AGREED skips the interactive license prompt (required in headless containers).
# Note: the actual cache lands at /data/tts_cache/tts/ (coqui appends "tts" internally).
os.environ.setdefault("TTS_HOME", "/data/tts_cache")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import modal
from fastapi import FastAPI, Response
from pydantic import BaseModel

# Reuse shared volume — same one used by kvrc-animation
volume = modal.Volume.from_name("kvrc-data", create_if_missing=True)
VOLUME_PATH = "/data"
VOICE_SAMPLE = f"{VOLUME_PATH}/voice/kvrc-voice-sample.mp3"

tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "TTS>=0.22.0",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "numpy>=1.26.0",
        "fastapi>=0.110.0",
        "pydantic>=2.0",
    )
    .env({"COQUI_TOS_AGREED": "1", "TTS_HOME": "/data/tts_cache"})
)

app = modal.App("kvrc-tts", image=tts_image)

web_app = FastAPI()

class TTSRequest(BaseModel):
    text: str

_model = None

def _get_model():
    global _model
    if _model is None:
        from TTS.api import TTS  # noqa: PLC0415 — deferred to avoid import at build time
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    return _model


@web_app.post("/tts")
def synthesize(req: TTSRequest):
    import tempfile
    import subprocess

    if not os.path.exists(VOICE_SAMPLE):
        return Response(
            content=b'{"error":"voice sample missing - run: modal volume put kvrc-data tools/kvrc-voice-sample.mp3 voice/kvrc-voice-sample.mp3"}',
            status_code=500,
            media_type="application/json",
        )

    tts = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_wav = f.name
    out_mp3 = out_wav.replace(".wav", ".mp3")

    try:
        tts.tts_to_file(
            text=req.text,
            speaker_wav=VOICE_SAMPLE,
            language="en",
            file_path=out_wav,
        )
        subprocess.run(
            ["ffmpeg", "-i", out_wav, "-q:a", "5", "-y", out_mp3],
            check=True,
            capture_output=True,
        )
        with open(out_mp3, "rb") as f:
            audio_bytes = f.read()
    finally:
        for p in (out_wav, out_mp3):
            if os.path.exists(p):
                os.unlink(p)

    return Response(content=audio_bytes, media_type="audio/mpeg")


@web_app.get("/health")
def health():
    return {"status": "ok"}


@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=120,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    return web_app

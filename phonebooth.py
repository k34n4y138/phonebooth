from __future__ import annotations

"""Phonebooth – centralised STT/TTS service.

This module loads heavy GPU models once per Python process and exposes them
through convenient utility functions **and** a small FastAPI application so they
can be consumed either via direct import **or** over HTTP.  Importing this
module is therefore enough to have the Whisper and XTTS models loaded and ready
for use.

Usage from code (no network overhead):

    import phonebooth as pb

    text = await pb.transcribe_audio(raw_wav_bytes)
    wav_np = await pb.synthesize(text="hello", speaker="0")

Running as a standalone micro-service:

    uvicorn phonebooth:app --host 0.0.0.0 --port 8000

Keeping the two access patterns together guarantees that every place in the
codebase re-uses the **same** model instances and therefore the same GPU
memory.
"""
__author__: str = "Zakaria Moumen <keanay@1337.ma>"

from fastapi import FastAPI, UploadFile, File, Query, Header, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from TTS.api import TTS
import torch
import tempfile
import asyncio
import io
import torchaudio
import numpy as np
import logging
import py3langid as langid
import os
from typing import Callable, Any, cast, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=False,  # respect any existing configuration
)

# ---------------------------------------------------------------------------
# Device & model initialisation – executed once at import time
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Whisper STT -------------------------------------------------------------
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "float32"
logger.info("Loading Whisper model (%s, compute=%s) on %s", WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, DEVICE)
WHISPER_MODEL = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

# 2) XTTS TTS ----------------------------------------------------------------
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
logger.info("Loading XTTS model '%s' on %s", TTS_MODEL_NAME, DEVICE)
XTTS = TTS(model_name=TTS_MODEL_NAME)
XTTS.to(DEVICE)

# Speaker utilities ----------------------------------------------------------
XTTS_SPEAKERS: dict[str, str] = {f"{idx}": name for idx, name in enumerate(XTTS.speakers)}
XTTS_DEFAULT_SPEAKER_IDX: int = 52
_XTTS_LOCK = asyncio.Lock()
_XTTS_EXECUTOR = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Language detection (small and cheap, can stay here)
# ---------------------------------------------------------------------------
_LANG_MAP: dict[str, str] = {
    "en": "en",
    "ar": "ar",
    "fr": "fr",
    "es": "es",
    "de": "de",
    "it": "it",
    "pt": "pt",
}

def detect_language(text: str, default: str = "en") -> str:  # noqa: D401
    """Return short language code best matching *text*.

    This uses *py3langid* under the hood but only allows a controlled set of
    languages to avoid extra dependencies for rarely used languages.
    """
    code, _ = langid.classify(text)
    lang = _LANG_MAP.get(code, default)
    logger.info("Detected language '%s' for text starting with %.30r", lang, text)
    return lang

# ---------------------------------------------------------------------------
# Core public async helpers
# ---------------------------------------------------------------------------

async def transcribe_audio(raw_audio: bytes) -> str:
    """Return text transcription of *raw_audio* (WAV bytes)."""
    # Write bytes to a temp file – faster-whisper handles resampling internally.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_audio)
        tmp.flush()
        tmp_path = tmp.name
    logger.info("Starting Whisper transcription of %s (%.1f kB)", tmp_path, len(raw_audio) / 1024)
    segments, _ = WHISPER_MODEL.transcribe(tmp_path, beam_size=5)
    transcription = " ".join(seg.text for seg in segments)
    logger.info("Whisper transcription finished (len=%d chars) %s", len(transcription), transcription)
    return transcription


async def synthesize(text: str, speaker: str | None = None) -> np.ndarray:  # noqa: D401
    """Return a 1-D *numpy* waveform for *text* using XTTS.

    The language is auto-detected.  *speaker* can be either a numeric index
    (string) or a direct speaker name.
    """
    if speaker is None:
        speaker = str(XTTS_DEFAULT_SPEAKER_IDX)
    # Resolve numeric index to actual name if necessary
    speaker_name = XTTS_SPEAKERS.get(speaker, speaker)
    lang = detect_language(text)
    logger.info("Synthesizing TTS for %s with speaker %s, lang %s", text, speaker_name, lang)
    # XTTS is blocking – offload to a thread to keep AsyncIO loop free
    # XTTS is NOT thread-safe – run calls sequentially behind an async lock and in a dedicated executor
    async with _XTTS_LOCK:
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(
            _XTTS_EXECUTOR,
            cast(Callable[..., Any], XTTS.tts),
            text,
            speaker_name,
            lang,
        )
    if isinstance(wav, (list, tuple)):
        wav = np.array(wav)
    logger.info("XTTS synthesis finished (%.2f s @24kHz)", len(wav) / 24000)
    return wav  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------
API_KEY: Optional[str] = os.getenv("API_KEY")
# Normalize empty string to None
if API_KEY == "":
    API_KEY = None
if API_KEY:
    logger.info("API key authentication enabled")
else:
    logger.info("API key authentication disabled (API_KEY env var not set)")

async def verify_api_key(request: Request) -> None:
    """Verify API key from Authorization Bearer token header.
    
    If API_KEY environment variable is set, authentication is required.
    If not set, authentication is optional and all requests are allowed.
    """
    if not API_KEY:
        # Authentication disabled - allow all requests
        return
    
    authorization = request.headers.get("Authorization")
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header. Please provide Authorization: Bearer <token> header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Parse Bearer token
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization[7:].strip()  # Remove "Bearer " prefix
    
    if token != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

# ---------------------------------------------------------------------------
# FastAPI application – optional, provides micro-service interface
# ---------------------------------------------------------------------------

async def _wav_chunks(
    wav: np.ndarray,
    *,
    sample_rate: int = 24_000,
    chunk_seconds: int = 3,
):
    """Yield *chunk_seconds*-second WAV chunks for *wav* (NumPy array).

    The caller can adjust *chunk_seconds* to control latency vs. overhead in
    the streaming endpoint.  Defaults to **2** seconds.
    """

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")

    samples_per_chunk = int(sample_rate * chunk_seconds)
    total = len(wav)
    offset = 0
    while offset < total:
        chunk = wav[offset : offset + samples_per_chunk]
        offset += samples_per_chunk
        buf = io.BytesIO()
        tensor_chunk = torch.tensor(chunk, dtype=torch.float32)
        torchaudio.save(buf, tensor_chunk.unsqueeze(0).cpu(), sample_rate, format="wav", bits_per_sample=16)
        yield buf.getvalue()

# ---------------------------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------------------------
# Proxy subroute support - when served behind nginx at a subroute
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")
if ROOT_PATH:
    logger.info("Proxy subroute configured: %s", ROOT_PATH)
    # Ensure root_path starts with / for FastAPI
    if not ROOT_PATH.startswith("/"):
        ROOT_PATH = "/" + ROOT_PATH
else:
    ROOT_PATH = ""

ENABLE_DOCS = os.getenv("ENABLE_DOCS", "").lower() not in ("false", "0", "no")
docs_url = ROOT_PATH + "/" if ENABLE_DOCS else None
redoc_url = None
if ENABLE_DOCS:
    logger.info("OpenAPI docs enabled at %s", docs_url or "/")
else:
    logger.info("OpenAPI docs disabled (set ENABLE_DOCS=true to enable)")

app = FastAPI(
    title="Phonebooth – STT/TTS Service",
    docs_url=docs_url,
    redoc_url=redoc_url,
    root_path=ROOT_PATH,
)

# Add middleware to trust proxy headers (X-Forwarded-*)
# This is important when running behind nginx or other reverse proxies
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Allow all hosts when behind proxy
)

class TranscriptionOut(BaseModel):
    text: str


@app.post("/transcribe", response_model=TranscriptionOut, dependencies=[Depends(verify_api_key)])
async def transcribe_endpoint(file: UploadFile = File(...)) -> TranscriptionOut:  # noqa: D401
    """Transcribe the uploaded WAV and return detected text."""
    raw = await file.read()
    text = await transcribe_audio(raw)
    logger.info("Whisper transcription finished (len=%d chars) %s", len(text), text)
    return TranscriptionOut(text=text)


class TTSRequest(BaseModel):
    text: str
    speaker: str = Query(str(XTTS_DEFAULT_SPEAKER_IDX), description="Speaker id or name")
    chunk_seconds: int = Query(2, ge=1, le=30, description="Seconds per audio chunk for streaming endpoint (default 2)")


@app.post("/tts", dependencies=[Depends(verify_api_key)])
async def tts_endpoint(req: TTSRequest) -> StreamingResponse:  # noqa: D401
    """Return synthesised WAV for *req.text* using speaker *req.speaker*."""
    wav_np = await synthesize(req.text, req.speaker)

    wav_tensor = torch.tensor(wav_np, dtype=torch.float32)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_tensor.unsqueeze(0).cpu(), 24_000, format="wav", bits_per_sample=16)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Streaming TTS – chunked WAV pieces (1 s) for real-time playback
# ---------------------------------------------------------------------------


@app.post("/tts_stream", dependencies=[Depends(verify_api_key)])
async def tts_stream_endpoint(req: TTSRequest) -> StreamingResponse:  # noqa: D401
    """Stream WAV chunks for *req.text* using speaker *req.speaker*.

    The chunk duration is controlled by *req.chunk_seconds* (default 2 s).
    """
    wav_np = await synthesize(req.text, req.speaker)

    async def streamer():
        async for chunk in _wav_chunks(wav_np, chunk_seconds=req.chunk_seconds):
            yield chunk

    return StreamingResponse(streamer(), media_type="application/octet-stream")


@app.get("/speakers", dependencies=[Depends(verify_api_key)])
async def speakers_endpoint() -> dict[str, str]:  # noqa: D401
    """Return mapping of numeric speaker indices to names."""
    return XTTS_SPEAKERS


# ---------------------------------------------------------------------------
# Demo page – simple browser client (phonebooth/demo.html)
# ---------------------------------------------------------------------------
ENABLE_DEMO = os.getenv("ENABLE_DEMO", "").lower() in ("true", "1", "yes")
if ENABLE_DEMO:
    logger.info("Demo page enabled at /demo")
    
    @app.get("/demo", response_class=FileResponse)
    async def demo_page() -> FileResponse:  # noqa: D401
        """Return the static demo HTML page bundled with the package."""
        return FileResponse(Path(__file__).with_name("demo.html"), media_type="text/html")
else:
    logger.info("Demo page disabled (set ENABLE_DEMO=true to enable)")

__all__ = [
    "DEVICE",
    "XTTS_SPEAKERS",
    "XTTS_DEFAULT_SPEAKER_IDX",
    "detect_language",
    "transcribe_audio",
    "synthesize",
    "app",
]

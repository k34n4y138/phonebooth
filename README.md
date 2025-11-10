# Phonebooth – Centralised STT/TTS Service

Phonebooth wraps two heavy-weight speech models – [Whisper](https://github.com/openai/whisper) for speech-to-text (STT) and [XTTS-v2](https://github.com/idiap/coqui-ai-TTS) for multilingual text-to-speech (TTS) – in a single Python module **and** a small FastAPI application.

Importing the module automatically initialises both models **once** per Python process so that every part of your codebase can share GPU memory.  If you prefer network isolation or a polyglot environment, the same object is also exposed through an HTTP micro-service.


---

## Features

* 📝 **Transcription** – whisper-large-v3 running through *faster-whisper* for blazingly fast inference.
* 🌐 **Language Support** – supports multiple languages for both transcription and synthesis.
* 🔊 **Synthesis** – XTTS-v2 multilingual TTS with 100+ speakers.
* 🔄 **Single initialisation** – heavy models are loaded once and re-used everywhere.
* ⚡ **Async helpers** – `await transcribe_audio()` & `await synthesize()`.
* 🛰️ **Micro-service** – optional FastAPI server with streaming responses.
* 📦 **Docker image** – builds models at image-creation time for near-instant container start-up.

---

## Quick start

### 1.  Install (local GPU or CPU)

```bash
# Clone this repository or copy phonebooth/ into your project
pip install -r phonebooth/requirements.txt
```

PyTorch wheels are **not** pinned in `requirements.txt` – install the flavour that matches your hardware & CUDA version _before_ the step above, e.g.

```bash
pip install torch==2.7.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2.  Use as a Python library

```python
import phonebooth as pb

wav_bytes = Path("example.wav").read_bytes()
text = await pb.transcribe_audio(wav_bytes)
print(text)

wav_np = await pb.synthesize("Bonjour le monde!", speaker="1")
# save with torchaudio, sounddevice, etc.
```

### 3.  Run as a micro-service

```bash
uvicorn phonebooth:app --host 0.0.0.0 --port 8000
```

*Swagger UI* will be available on the root path `/` by default. Set `ENABLE_DOCS=false` to disable it.

**Proxy subroute support**: When serving behind nginx or another reverse proxy at a subroute (e.g., `https://example.com/api/phonebooth`), set the `ROOT_PATH` environment variable to the subroute path (e.g., `/api/phonebooth`). This ensures URLs in OpenAPI docs and responses are generated correctly.

```text
POST /transcribe   → { "text": "…" }
POST /tts          → audio/wav (full file)
POST /tts_stream   → application/octet-stream (chunked WAV)
GET  /speakers      → { "0": "Speaker A", "1": "Speaker B", … }
```

### 4.  Docker (GPU)

A ready-to-run Dockerfile is provided:

```bash
# Build (models are downloaded & initialised during build)
docker build -t phonebooth .

# Run with NVIDIA runtime
docker run --gpus all -p 8000:8000 \
           -e NVIDIA_VISIBLE_DEVICES=0 \  # optional, limit GPU
           -e COQUI_TOS_AGREED=1 \         # agree to XTTS terms
           -e API_KEY=your-secret-key \    # optional, enable API authentication
           phonebooth
```

Container start-up is now instant because the heavy models were already loaded at build time (`RUN python phonebooth.py`).

**Docker Compose**: A `compose.yml` file is included for easy deployment. Copy `env.example` to `.env` and update the values:

```bash
cp env.example .env
# Edit .env with your preferred values
```

The compose file will automatically load variables from `.env` (the file is gitignored for security). See `env.example` for all available configuration options.

---

## API reference

### Async helpers (Python)

```python
await transcribe_audio(raw_audio: bytes) -> str
await synthesize(text: str, speaker: str | None = None) -> numpy.ndarray
```

* `raw_audio` must be 16-bit PCM WAV bytes (any sampling rate, mono/stereo).
* `speaker` may be either a numeric index **as string** or an exact speaker name.
* The language for TTS is auto-detected via *py3langid* (configurable mapping inside the module).

### HTTP endpoints (FastAPI)

| Method | Path          | Body                               | Response                |
|--------|---------------|------------------------------------|-------------------------|
| POST   | `/transcribe` | `multipart/form-data` file=`wav`   | `{ "text": "…" }`      |
| POST   | `/tts`        | JSON: `{text, speaker}`            | `audio/wav`             |
| POST   | `/tts_stream` | JSON: `{text, speaker, chunk_seconds}` | binary stream *(2 s chunks by default)* |
| GET    | `/speakers`   | –                                  | `{ "id": "name", … }` |
| GET    | `/demo`       | –                                  | HTML demo page *(requires `ENABLE_DEMO=true`)* |

**Authentication**: When the `API_KEY` environment variable is set, all API endpoints (except `/demo`) require Bearer token authentication. Include the header `Authorization: Bearer <your-api-key>` in requests. The `/demo` page is publicly accessible (when enabled) and includes an optional API key input field for authenticated API calls.

Streaming TTS chunks let you start playback while the rest of the sentence is still being synthesised – ideal for real-time chat assistants.

---

## Configuration

Variable | Purpose | Default
---------|----------|--------
`API_KEY` | API key for Bearer token authentication. When set, all endpoints (except `/demo`) require `Authorization: Bearer <key>` header | –
`ENABLE_DEMO` | Enable the `/demo` endpoint. Set to `true`, `1`, or `yes` to enable the demo page | `false`
`ENABLE_DOCS` | Enable OpenAPI documentation (Swagger UI). Set to `false`, `0`, or `no` to disable | `true`
`ROOT_PATH` | Root path when served behind a reverse proxy at a subroute (e.g., `/api/phonebooth`). Leave empty for root deployment | –
`COQUI_TOS_AGREED` | Must be set to `1` to silence the XTTS licence prompt | –
`GLOG_minloglevel` | Reduce TensorFlow/Whisper spam (`export GLOG_minloglevel=2`) | –

Edit `phonebooth.py` if you want to:

* change `WHISPER_MODEL_SIZE` (tiny, small, medium, large-v3, …)
* pin a different compute type (`int8_float16`, `float16`, …)
* replace the XTTS speaker list mapping

---

## Troubleshooting

* **CUDA out of memory** – lower the compute type (e.g. float16 → int8_float16) or run on CPU.
* **Libtorch/ONNX warnings** – harmless; silence them with `TORCH_WARNINGS=ignore`.
* **Slow startup** – first run downloads ~3 GB of model weights; subsequent runs are instant.
---

## Licence

This repository only contains glue code; it relies on third-party models licensed under their respective terms.  See the original projects for details.  All original code in *phonebooth* is released under the MIT licence © 2024.

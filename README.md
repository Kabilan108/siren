# siren

A simple server using OpenAI's Whisper model for my personal audio transcription needs.

[![Tests](https://github.com/Kabilan108/whisper-server/actions/workflows/tests.yml/badge.svg)](https://github.com/Kabilan108/whisper-server/actions/workflows/tests.yml)

## About

This project sets up a lightweight server to process audio files using Whisper. It’s designed for personal use and includes tests to ensure reliability. It's meant to be used with [dictator] - a local vocie typing app for linux.

## Setup

### Running Locally

1. Install dependencies with `uv sycn --frozen`.
2. Run the server with `uv run fastapi run server.py`.

### Running via Docker

Run this if your machine has nvidia gpu (make sure you have the Nvidia Container Toolkit installed):

```bash
docker compose up -d siren-gpu
```

If your machine does not have a GPU:

```bash
docker compose up -d siren-cpu
```

## Running Tests

Tests are run automatically via GitHub Actions. To run locally:
```bash
uv run pytest
```

## Timestamped transcriptions

`POST /v1/audio/transcriptions` keeps the OpenAI-compatible text response by
default. Request `verbose_json` to include sentence-level timestamps:

```bash
curl -H "Authorization: Bearer $SIREN_API_KEY" \
  -F "file=@meeting.flac" \
  -F "model=nvidia/parakeet-tdt-0.6b-v2" \
  -F "response_format=verbose_json" \
  https://siren.example/v1/audio/transcriptions
```

The verbose response contains `text`, `language`, `duration`, and `segments`
with `start`, `end`, and `text`. Other response-format values retain the legacy
JSON `{ "text": ... }` response. Multipart uploads are streamed to a temporary
file in 1 MiB blocks, and model inference is serialized to keep concurrent
requests from exhausting GPU memory.

For long recordings on the current 24 GB GPU host, clients should submit
bounded chunks. A representative meeting benchmark completed 5-minute and
10-minute Parakeet chunks successfully; a 20-minute chunk exhausted GPU memory.
The meeting pipeline therefore uses 10 minutes as its retry and progress unit.

## Whisper recognition defaults

Whisper backends share a fixed recognition recipe for dictation and durable
transcript jobs: English unless a language is explicitly supplied, beam size 1,
temperature 0, previous-text conditioning off, and no VAD trimming. English
requests use the bounded terminology prompt in `siren/config.py`. Explicit
non-English requests omit that English prompt. Word timestamps remain opt-in
for ordinary transcription and enabled by the long-form job worker.

`SIREN_WHISPER_SPEECH_GATE=true` enables a whole-input speech check. If the first
check finds no speech, a second check uses a peak-normalized copy to rescue quiet
speech. Only two negative checks suppress transcription. When speech is found,
Whisper receives the complete original samples, without trimming or amplification.
Empty responses preserve the audio duration and contain no segments.

The gate defaults to off pending the real-recording review. Setting it to `false`
disables only this check; the Whisper recognition recipe stays in place. This
change does not switch a saved server model or load a second resident model.

To share the API process's resident ASR model with durable jobs, set
`SIREN_JOB_ASR_URL=http://127.0.0.1:8301` in the server service environment,
using its actual listening port. Job workers inherit it and send each chunk to
the authenticated local transcription endpoint, requesting word timestamps.
They do not load a second ASR backend. The URL must be an HTTP loopback origin;
proxies and redirects are disabled. Without the setting, workers retain their
standalone model-loading behavior.

Use the same model ID in every client to avoid model reloads. Foreground and
job transcription share the existing inference semaphore; a foreground request
may wait for the current chunk. Conversion, job isolation, diarization and
alignment remain in the worker. Speaker diarization still loads a separate,
temporary model after ASR. Parakeet support can remain installed without loading
it. Changing these defaults requires updating clients and the server's persisted
model selection; changing the Python default alone does not override saved state.

## Health check

`GET /health` is unauthenticated and returns the running service version:

```json
{"status":"ok","version":"1.2.0"}
```

### Parakeet terminology hint

Set `SIREN_PARAKEET_CLAUDE_HINT=true` to enable the evaluated single-term
`Claude` hint at strength 0.125 for `nvidia/parakeet-tdt-0.6b-v2`. It is
disabled by default and does not affect other Parakeet models. Restart the
service after changing it. This improves selected name spellings, not
sentence-level recognition or negations. On NixOS set
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` for Triton CUDA discovery.
User-configurable request vocabularies remain separate future work.

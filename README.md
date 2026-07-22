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

## Health check

`GET /health` is unauthenticated and returns the running service version:

```json
{"status":"ok","version":"1.1.0"}
```

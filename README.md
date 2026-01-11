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

## Streaming Transcription

Siren supports real-time streaming transcription via WebSocket.

### Endpoint

```
ws://<host>:8000/ws/transcribe?token=<api_key>
```

### Protocol

Send audio chunks as base64-encoded 16kHz mono PCM:

```json
{"type": "audio", "data": "<base64 PCM>", "seq": 1}
```

Receive partial transcriptions:

```json
{"type": "partial", "text": "hello world", "stable_len": 6, "seq": 1}
```

The `stable_len` field indicates how many characters from the start are finalized.

Signal end of audio:

```json
{"type": "end"}
```

Receive final transcription:

```json
{"type": "final", "text": "Hello world.", "seq": 2}
```

### Latency Configuration

Send a config message before audio to adjust latency:

```json
{"type": "config", "chunk_frames": 7}
```

| chunk_frames | Latency |
|--------------|---------|
| 2 | 160ms |
| 7 | 560ms (default) |
| 14 | 1.12s |

Lower latency = faster response but potentially less accurate.

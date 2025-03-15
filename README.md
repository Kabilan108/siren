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

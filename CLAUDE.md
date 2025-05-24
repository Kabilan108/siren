# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Siren is an OpenAI-compatible Whisper server using Faster Whisper backend for audio transcription. It exposes `/v1/models` and `/v1/audio/transcriptions` endpoints with Bearer token authentication. The server supports dynamic model switching with persistence via config.json.

## Build & Run Commands
- Install dependencies: `uv sync --frozen`
- Run server: `uv run fastapi run server.py`
- Run tests: `uv run pytest`
- Run single test: `uv run pytest test_server.py::test_function_name -v`
- Docker GPU: `docker compose up -d siren`
- Docker CPU: `docker compose up -d siren-cpu`

## Architecture
- **Global State**: `current_model` and `current_model_name` track the loaded WhisperModel instance
- **Model Management**: `get_whisper_model()` handles model loading/switching with automatic GPU detection
- **Authentication**: `verify_token()` dependency validates Bearer tokens against `SIREN_API_KEY` env var
- **Configuration**: User config persisted in `~/config.json` with model preference
- **Lifespan**: App startup loads saved model, shutdown cleans GPU memory

## Code Style
- **Format**: No specific formatter, follow existing 4-space indent style
- **Imports**: Group standard library, third-party, and local imports with blank lines between
- **Type Hints**: Use Python 3.11 syntax with Union types as `X | Y` instead of `Union[X, Y]`
- **Error Handling**: Use try/except blocks with specific exceptions and informative error messages
- **Logging**: Use the `logger` from the root module
- **Authentication**: Use Bearer token authentication with `verify_token` dependency
- **API Schema**: Follow OpenAI-compatible endpoint structure
- **Tests**: Use pytest fixtures and mocks, clearly name tests with pattern `test_function_what_condition`
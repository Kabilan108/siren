# Changelog

## 1.1.0 - 2026-07-28

- Added an async job API (`POST /v1/jobs/transcripts`, plus status and
  `/result` endpoints) that runs chunking, transcription, diarization, and
  alignment server-side from a single upload.
- Added speaker diarization at `POST /v1/audio/diarize`, backed by streaming
  Sortformer v2.1 running in a supervised worker subprocess.
- Added word-to-speaker alignment at `POST /v1/audio/align`.
- Added OpenAI-compatible word-level timestamps to both transcription
  backends, including streamed responses.
- Added word-driven pause segmentation for Parakeet transcripts.
- Added a NixOS module for running siren as a native service.
- Split the server into a package and added a golden regression harness
  pinning the pre-existing response bytes.

## 0.1.0 - 2026-07-08

- Initial release of the OpenAI-compatible audio transcription server.
- Added Faster Whisper and NVIDIA Parakeet transcription backends.
- Added Docker Compose support for CPU and GPU deployments.
- Added comparison and recording-duration helper scripts.
- Added tests for server behavior, audio helpers, and model loading.
- Added local release tooling for building and pushing GPU Docker images.

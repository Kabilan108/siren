# Siren Streaming Implementation Specification

This document specifies the implementation of real-time streaming transcription support for Siren using NVIDIA's Nemotron Speech Streaming model. This spec is self-contained and should be followed step-by-step.

## Overview

### Current State
- FastAPI server with OpenAI-compatible `/v1/audio/transcriptions` endpoint
- Batch transcription only (upload full audio, get full transcript)
- Supports Whisper (faster-whisper) and Parakeet (NeMo) models

### Target State
- New WebSocket endpoint `/ws/transcribe` for real-time streaming
- Integration with `nvidia/nemotron-speech-streaming-en-0.6b` model
- Partial transcriptions with stable prefix tracking
- Existing batch API remains unchanged

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protocol | WebSocket | Bidirectional streaming for audio in, text out |
| Model | Nemotron 0.6B | Native streaming, cache-aware, 600M params |
| Chunk size | 7 frames (560ms) | Balanced latency/accuracy |
| Message format | JSON | Simplicity, debugging ease |
| Stable tracking | `stable_len` field | Client knows what won't change |

## Dependencies

Add to `pyproject.toml`:

```toml
# NeMo toolkit should already be present for Parakeet
# Nemotron uses the same nemo_toolkit[asr] dependency
# No additional dependencies required for the model itself
```

Ensure NeMo version is >= 2.6.0 (already in project).

## WebSocket Protocol Specification

### Endpoint
```
ws://<host>:8000/ws/transcribe
```

### Authentication
Bearer token in first message or as query parameter:
```
ws://<host>:8000/ws/transcribe?token=<api_key>
```

### Client → Server Messages

#### Config Message (optional, must be first if sent)
```json
{
  "type": "config",
  "chunk_frames": 7,
  "language": "en"
}
```
- `chunk_frames`: Number of 80ms frames per chunk (default: 7 = 560ms)
- `language`: Language hint (currently only "en" supported by Nemotron)

#### Audio Message
```json
{
  "type": "audio",
  "data": "<base64-encoded PCM>",
  "seq": 1
}
```
- `data`: Base64-encoded raw PCM audio (16kHz, mono, 16-bit signed LE)
- `seq`: Sequence number for ordering (monotonically increasing)

#### End Message
```json
{
  "type": "end"
}
```
Signals end of audio stream. Server will send final transcription and close.

### Server → Client Messages

#### Partial Transcription
```json
{
  "type": "partial",
  "text": "hello world this is",
  "stable_len": 12,
  "seq": 5
}
```
- `text`: Current full transcription hypothesis
- `stable_len`: Number of characters from start that are finalized (won't change)
- `seq`: Corresponds to latest processed audio sequence

#### Final Transcription
```json
{
  "type": "final",
  "text": "Hello world, this is a test.",
  "seq": 10
}
```
- Sent after receiving "end" message
- Full transcription with punctuation/capitalization
- `stable_len` omitted (entire text is stable)

#### Error Message
```json
{
  "type": "error",
  "message": "Invalid audio format",
  "code": "INVALID_AUDIO"
}
```

Error codes:
- `AUTH_FAILED`: Invalid or missing token
- `INVALID_MESSAGE`: Malformed JSON or unknown message type
- `INVALID_AUDIO`: Audio decode failed
- `MODEL_ERROR`: Transcription model error
- `INTERNAL_ERROR`: Unexpected server error

---

## Implementation Steps

### Step 1: Add Nemotron to Supported Models

**File**: `siren/server.py`

1. Add Nemotron to the models list:

```python
NEMOTRON_MODELS = [
    "nvidia/nemotron-speech-streaming-en-0.6b",
]
```

2. Add helper function:

```python
def is_nemotron_model(model_name: str) -> bool:
    return model_name in NEMOTRON_MODELS or "nemotron-speech-streaming" in model_name
```

3. Update `get_compute_params()` if needed (Nemotron uses same GPU detection).

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/
```

---

### Step 2: Create Streaming Pipeline Module

**File**: `siren/streaming.py` (new file)

This module handles the Nemotron streaming inference pipeline.

```python
"""Streaming transcription pipeline using Nemotron."""

from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription."""

    chunk_frames: int = 7  # Number of 80ms frames (7 = 560ms latency)
    sample_rate: int = 16000

    @property
    def samples_per_frame(self) -> int:
        """Samples in one 80ms frame."""
        return int(self.sample_rate * 0.08)

    @property
    def samples_per_chunk(self) -> int:
        """Samples needed for one chunk."""
        return self.samples_per_frame * self.chunk_frames


@dataclass
class PartialResult:
    """A partial transcription result."""

    text: str
    stable_len: int
    seq: int
    is_final: bool = False


@dataclass
class StreamingSession:
    """Manages state for one streaming transcription session."""

    config: StreamingConfig
    model: object  # nemo_asr.models.ASRModel
    audio_buffer: bytes = field(default_factory=bytes)
    seq: int = 0
    _cache: dict = field(default_factory=dict)

    def add_audio(self, pcm_base64: str) -> None:
        """Add base64-encoded PCM audio to buffer."""
        self.audio_buffer += base64.b64decode(pcm_base64)

    def _bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 array."""
        samples = len(pcm_bytes) // 2
        int16_array = struct.unpack(f"<{samples}h", pcm_bytes)
        return np.array(int16_array, dtype=np.float32) / 32768.0

    async def process_chunks(self) -> AsyncIterator[PartialResult]:
        """Process buffered audio and yield partial results."""
        bytes_per_chunk = self.config.samples_per_chunk * 2  # 2 bytes per sample

        while len(self.audio_buffer) >= bytes_per_chunk:
            chunk_bytes = self.audio_buffer[:bytes_per_chunk]
            self.audio_buffer = self.audio_buffer[bytes_per_chunk:]
            self.seq += 1

            audio_array = self._bytes_to_float32(chunk_bytes)

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_chunk,
                audio_array,
            )

            if result:
                yield PartialResult(
                    text=result["text"],
                    stable_len=result["stable_len"],
                    seq=self.seq,
                )

    def _transcribe_chunk(self, audio: np.ndarray) -> dict | None:
        """Transcribe a single chunk using cache-aware streaming.

        This method runs in a thread pool executor.
        """
        # The Nemotron model uses cache-aware streaming
        # We need to maintain encoder caches across chunks
        #
        # For cache-aware RNNT models in NeMo, we use:
        # model.transcribe_step() or the streaming pipeline
        #
        # Implementation depends on NeMo's streaming API
        # See: nemo/collections/asr/inference/factory/pipeline_builder.py

        # TODO: Implement actual Nemotron streaming inference
        # This is a placeholder showing the expected interface

        # The model maintains internal cache state
        # We pass audio chunks and get partial hypotheses
        # stable_len indicates how much of the text is finalized

        return {
            "text": "",
            "stable_len": 0,
        }

    async def finalize(self) -> PartialResult:
        """Process remaining audio and return final result."""
        # Process any remaining buffered audio
        async for partial in self.process_chunks():
            pass  # Consume remaining chunks

        # Signal end of stream to model and get final result
        loop = asyncio.get_running_loop()
        final_text = await loop.run_in_executor(
            None,
            self._finalize_transcription,
        )

        self.seq += 1
        return PartialResult(
            text=final_text,
            stable_len=len(final_text),
            seq=self.seq,
            is_final=True,
        )

    def _finalize_transcription(self) -> str:
        """Finalize the transcription and return complete text.

        This method runs in a thread pool executor.
        """
        # Flush any remaining audio through the model
        # Apply final punctuation/capitalization
        # Clear caches

        # TODO: Implement actual finalization
        return ""
```

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/
```

---

### Step 3: Implement Nemotron Model Loading

**File**: `siren/streaming.py` (add to existing)

Add the actual Nemotron model integration:

```python
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf


def load_nemotron_model(model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"):
    """Load the Nemotron streaming ASR model."""
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Configure for streaming with cache-aware inference
    # att_context_size controls the chunk size:
    # [70, 6] = 7 frames = 560ms latency
    # The model's decoding strategy needs to be set for streaming

    return model


def create_streaming_session(
    model: nemo_asr.models.ASRModel,
    config: StreamingConfig | None = None,
) -> StreamingSession:
    """Create a new streaming session with the given model."""
    if config is None:
        config = StreamingConfig()

    return StreamingSession(config=config, model=model)
```

Update `StreamingSession._transcribe_chunk()` to use the actual NeMo streaming API:

```python
def _transcribe_chunk(self, audio: np.ndarray) -> dict | None:
    """Transcribe a single chunk using cache-aware streaming."""
    # NeMo cache-aware streaming uses the model's streaming methods
    # The exact API depends on the model type (RNNT vs CTC)
    #
    # For RNNT models like Nemotron:
    # 1. Prepare audio tensor
    # 2. Call model with cache state
    # 3. Get partial hypothesis and updated cache

    import torch

    # Convert to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # [1, samples]
    audio_len = torch.tensor([len(audio)])

    # For cache-aware streaming, we use transcribe() with streaming config
    # or the lower-level encode/decode methods with cache management
    #
    # The Nemotron model supports:
    # - model.transcribe() for batch
    # - Streaming via PipelineBuilder for real-time

    # Simplified approach: accumulate and re-transcribe
    # (Not optimal but functional for initial implementation)
    # TODO: Implement proper cache-aware streaming

    if not hasattr(self, '_accumulated_audio'):
        self._accumulated_audio = audio
    else:
        self._accumulated_audio = np.concatenate([self._accumulated_audio, audio])

    # Transcribe accumulated audio
    with torch.no_grad():
        result = self.model.transcribe([self._accumulated_audio])

    if result and len(result) > 0:
        text = result[0].text if hasattr(result[0], 'text') else str(result[0])
        # For now, mark most of the text as stable
        # A proper implementation would track hypothesis stability
        stable_len = max(0, len(text) - 20)  # Last 20 chars may change
        return {"text": text, "stable_len": stable_len}

    return None
```

**Note**: The above is a simplified implementation. Proper cache-aware streaming requires deeper integration with NeMo's streaming pipeline. Document this as a follow-up optimization.

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/
```

---

### Step 4: Add WebSocket Endpoint

**File**: `siren/server.py`

Add WebSocket imports and endpoint:

```python
from fastapi import WebSocket, WebSocketDisconnect
from siren.streaming import (
    StreamingConfig,
    StreamingSession,
    create_streaming_session,
    load_nemotron_model,
)

# Global streaming model (separate from batch model)
streaming_model: nemo_asr.models.ASRModel | None = None
streaming_model_lock = asyncio.Lock()


async def get_streaming_model():
    """Get or load the Nemotron streaming model."""
    global streaming_model

    async with streaming_model_lock:
        if streaming_model is None:
            loop = asyncio.get_running_loop()
            streaming_model = await loop.run_in_executor(
                None,
                load_nemotron_model,
            )
        return streaming_model


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, token: str | None = None):
    """WebSocket endpoint for streaming transcription."""
    await websocket.accept()

    # Authenticate
    auth_token = token
    if not auth_token:
        # Try to get token from first message
        try:
            first_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            if first_msg.get("type") == "auth":
                auth_token = first_msg.get("token")
        except asyncio.TimeoutError:
            await websocket.send_json({
                "type": "error",
                "message": "Authentication timeout",
                "code": "AUTH_FAILED",
            })
            await websocket.close()
            return

    if not verify_token_value(auth_token):
        await websocket.send_json({
            "type": "error",
            "message": "Invalid authentication token",
            "code": "AUTH_FAILED",
        })
        await websocket.close()
        return

    # Get model
    try:
        model = await get_streaming_model()
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to load model: {e}",
            "code": "MODEL_ERROR",
        })
        await websocket.close()
        return

    # Create session with default config
    config = StreamingConfig()
    session = create_streaming_session(model, config)

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "config":
                # Update config (must be before audio)
                if "chunk_frames" in msg:
                    config = StreamingConfig(chunk_frames=msg["chunk_frames"])
                    session = create_streaming_session(model, config)

            elif msg_type == "audio":
                # Add audio and process
                session.add_audio(msg["data"])
                async for partial in session.process_chunks():
                    await websocket.send_json({
                        "type": "partial",
                        "text": partial.text,
                        "stable_len": partial.stable_len,
                        "seq": partial.seq,
                    })

            elif msg_type == "end":
                # Finalize and send final result
                final = await session.finalize()
                await websocket.send_json({
                    "type": "final",
                    "text": final.text,
                    "seq": final.seq,
                })
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "code": "INVALID_MESSAGE",
                })

    except WebSocketDisconnect:
        pass  # Client disconnected, clean up silently
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "code": "INTERNAL_ERROR",
            })
        except:
            pass  # Connection may already be closed
    finally:
        await websocket.close()


def verify_token_value(token: str | None) -> bool:
    """Verify a token value directly (not as a dependency)."""
    if not token:
        return False
    expected = os.getenv("SIREN_API_KEY", "dev_token")
    return token == expected
```

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/
uv run fastapi dev siren/server.py &  # Start server
# Test with wscat or Python websocket client
```

---

### Step 5: Add WebSocket Tests

**File**: `tests/test_websocket.py` (new file)

```python
"""Tests for WebSocket streaming transcription."""

import asyncio
import base64
import struct

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from siren.server import app


def generate_silent_audio(duration_ms: int = 100, sample_rate: int = 16000) -> str:
    """Generate base64-encoded silent PCM audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Silent audio = all zeros
    pcm_bytes = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return base64.b64encode(pcm_bytes).decode("ascii")


class TestWebSocketAuth:
    """Test WebSocket authentication."""

    def test_auth_via_query_param(self):
        """Test authentication via query parameter."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"

    def test_auth_failure(self):
        """Test authentication failure."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=wrong") as ws:
                response = ws.receive_json()
                assert response["type"] == "error"
                assert response["code"] == "AUTH_FAILED"


class TestWebSocketStreaming:
    """Test WebSocket streaming transcription."""

    def test_config_message(self):
        """Test config message updates chunk size."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "config", "chunk_frames": 14})
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"

    def test_audio_message(self):
        """Test sending audio and receiving partials."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                # Send enough audio for one chunk (560ms at 16kHz)
                audio = generate_silent_audio(duration_ms=600)
                ws.send_json({"type": "audio", "data": audio, "seq": 1})

                # Should get a partial result (may be empty for silent audio)
                # Then send end
                ws.send_json({"type": "end"})

                # Collect all responses until final
                responses = []
                while True:
                    response = ws.receive_json()
                    responses.append(response)
                    if response["type"] == "final":
                        break

                assert any(r["type"] == "final" for r in responses)

    def test_invalid_message_type(self):
        """Test unknown message type returns error."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "unknown"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert response["code"] == "INVALID_MESSAGE"


class TestWebSocketProtocol:
    """Test WebSocket protocol edge cases."""

    def test_empty_audio(self):
        """Test empty audio stream."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"
                assert response["text"] == ""

    def test_multiple_audio_chunks(self):
        """Test sending multiple audio chunks."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                for seq in range(1, 4):
                    audio = generate_silent_audio(duration_ms=200)
                    ws.send_json({"type": "audio", "data": audio, "seq": seq})

                ws.send_json({"type": "end"})

                # Drain responses
                while True:
                    response = ws.receive_json()
                    if response["type"] == "final":
                        break
```

**Verification**:
```bash
uv run ty check tests/
uv run ruff check tests/
uv run ruff format tests/
uv run pytest tests/test_websocket.py -v
```

---

### Step 6: Update Model Listing Endpoint

**File**: `siren/server.py`

Update the `/v1/models` endpoint to include streaming models:

```python
@app.get("/v1/models")
async def list_models(_: str = Depends(verify_token)):
    """List available transcription models."""
    all_models = (
        list(WHISPER_MODELS)
        + list(PARAKEET_MODELS)
        + list(NEMOTRON_MODELS)
    )
    return {
        "data": [{"id": model} for model in all_models],
    }
```

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/
curl -H "Authorization: Bearer dev_token" http://localhost:8000/v1/models | jq
# Should include nvidia/nemotron-speech-streaming-en-0.6b
```

---

### Step 7: Documentation

**File**: `README.md` (update)

Add a section on streaming:

```markdown
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
```

**Verification**:
```bash
# Ensure markdown is valid
cat README.md
```

---

### Step 8: Upgrade to Cache-Aware Streaming

**IMPORTANT**: This step replaces the simplified "accumulate and retranscribe" approach with proper O(1) per-chunk cache-aware streaming.

**File**: `siren/streaming.py`

The current implementation re-transcribes all accumulated audio on each chunk, which is O(n²). The Nemotron model supports cache-aware streaming where each chunk is processed in O(1) time by maintaining encoder state between chunks.

#### 8.1: Update StreamingSession to hold cache state

Replace the `_cache` field with proper cache tensors:

```python
@dataclass
class StreamingSession:
    """Manages state for one streaming transcription session."""

    config: StreamingConfig
    model: object  # nemo_asr.models.ASRModel
    audio_buffer: bytes = field(default_factory=bytes)
    seq: int = 0

    # Cache state for streaming (initialized on first chunk)
    _cache_last_channel: torch.Tensor | None = field(default=None, repr=False)
    _cache_last_time: torch.Tensor | None = field(default=None, repr=False)
    _cache_last_channel_len: torch.Tensor | None = field(default=None, repr=False)
    _previous_hypotheses: list | None = field(default=None, repr=False)
    _full_transcript: str = field(default="", repr=False)
```

#### 8.2: Add cache initialization method

```python
def _init_cache(self) -> None:
    """Initialize encoder cache state for streaming."""
    if self._cache_last_channel is not None:
        return  # Already initialized

    (
        self._cache_last_channel,
        self._cache_last_time,
        self._cache_last_channel_len,
    ) = self.model.encoder.get_initial_cache_state(batch_size=1)
    self._previous_hypotheses = None
    self._full_transcript = ""
```

#### 8.3: Replace _transcribe_chunk with cache-aware implementation

```python
def _transcribe_chunk(self, audio: np.ndarray) -> dict | None:
    """Transcribe a single chunk using cache-aware streaming.

    Uses NeMo's conformer_stream_step() for O(1) per-chunk processing.
    """
    self._init_cache()

    # Prepare audio tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
    audio_len = torch.tensor([len(audio)], dtype=torch.long)

    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()
        audio_len = audio_len.cuda()

    # Preprocess audio to get mel spectrogram
    processed_signal, processed_signal_length = self.model.preprocessor(
        input_signal=audio_tensor,
        length=audio_len,
    )

    with torch.no_grad():
        # Process chunk with cache state
        (
            pred_out_stream,
            transcribed_texts,
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
            self._previous_hypotheses,
        ) = self.model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=self._cache_last_channel,
            cache_last_time=self._cache_last_time,
            cache_last_channel_len=self._cache_last_channel_len,
            previous_hypotheses=self._previous_hypotheses,
            return_transcription=True,
        )

    if transcribed_texts and len(transcribed_texts) > 0:
        new_text = transcribed_texts[0]

        # Track stability: the previous full transcript is stable
        # Only the new tokens may change
        stable_len = len(self._full_transcript)
        self._full_transcript = new_text

        return {
            "text": new_text,
            "stable_len": stable_len,
        }

    return None
```

#### 8.4: Update _finalize_transcription for cache-aware mode

```python
def _finalize_transcription(self) -> str:
    """Finalize the transcription and return complete text."""
    # If we have remaining audio that didn't form a full chunk, process it
    if len(self.audio_buffer) > 0:
        remaining_audio = self._bytes_to_float32(self.audio_buffer)
        self.audio_buffer = b""
        result = self._transcribe_chunk(remaining_audio)
        if result:
            self._full_transcript = result["text"]

    # Reset cache state for next session
    final_text = self._full_transcript
    self._cache_last_channel = None
    self._cache_last_time = None
    self._cache_last_channel_len = None
    self._previous_hypotheses = None
    self._full_transcript = ""

    return final_text
```

#### 8.5: Add torch import at module level

Ensure torch is imported at the top of the file:

```python
import torch
```

**Verification**:
```bash
uv run ty check siren/
uv run ruff check siren/
uv run ruff format siren/

# Test with actual audio (not just silent)
# Record some speech and send via WebSocket client
```

#### Troubleshooting

If `conformer_stream_step` is not available on the model:

1. Check if the model has the method:
   ```python
   hasattr(model, 'conformer_stream_step')
   ```

2. Some models use a different API. Check NeMo's `nemo/collections/asr/models/rnnt_models.py` for the exact method signature.

3. Alternative: Use NeMo's `TranscriptionPipeline` with streaming config:
   ```python
   from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
   from omegaconf import OmegaConf

   cfg = OmegaConf.create({
       "model_path": "nvidia/nemotron-speech-streaming-en-0.6b",
       "streaming": True,
       "chunk_size": 560,  # ms
   })
   pipeline = PipelineBuilder.build_pipeline(cfg)
   ```

---

## Final Verification Checklist

Run all checks:

```bash
# Type checking
uv run ty check siren/ tests/

# Linting
uv run ruff check siren/ tests/

# Formatting
uv run ruff format siren/ tests/

# Tests
uv run pytest tests/ -v

# Manual test
uv run fastapi dev siren/server.py
# In another terminal, test WebSocket with a client
```

## Notes for Implementer

1. **Nemotron Model Download**: The model is ~600MB and will be downloaded on first use. Ensure sufficient disk space and network access to Hugging Face.

2. **GPU Memory**: Nemotron + Whisper/Parakeet models may compete for GPU memory. Consider unloading batch models when streaming is active, or document memory requirements.

3. **Cache-Aware Streaming**: The initial implementation uses a simplified approach (re-transcribing accumulated audio). For production, implement proper cache-aware streaming using NeMo's `PipelineBuilder`:

```python
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "model_path": "nvidia/nemotron-speech-streaming-en-0.6b",
    "att_context_size": [70, 6],  # 7 frames = 560ms
    # ... other streaming config
})
pipeline = PipelineBuilder.build_pipeline(cfg)
```

4. **Stable Length Calculation**: The current implementation uses a simple heuristic (last 20 chars may change). A proper implementation would track hypothesis stability from the RNNT decoder.

5. **Error Recovery**: Consider adding reconnection support and session resumption for robustness.

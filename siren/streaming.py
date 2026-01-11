"""Streaming transcription pipeline using Nemotron."""

from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np
import torch
import nemo.collections.asr as nemo_asr


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

    # Cache state for streaming (initialized on first chunk)
    _cache_last_channel: torch.Tensor | None = field(default=None, repr=False)
    _cache_last_time: torch.Tensor | None = field(default=None, repr=False)
    _cache_last_channel_len: torch.Tensor | None = field(default=None, repr=False)
    _previous_hypotheses: list | None = field(default=None, repr=False)
    _full_transcript: str = field(default="", repr=False)

    def add_audio(self, pcm_base64: str) -> None:
        """Add base64-encoded PCM audio to buffer."""
        try:
            self.audio_buffer += base64.b64decode(pcm_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}") from e

    def _bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 array."""
        samples = len(pcm_bytes) // 2
        int16_array = struct.unpack(f"<{samples}h", pcm_bytes)
        return np.array(int16_array, dtype=np.float32) / 32768.0

    def _init_cache(self) -> None:
        """Initialize encoder cache state for streaming."""
        if self._cache_last_channel is not None:
            return  # Already initialized

        # Check if model supports cache-aware streaming
        try:
            if hasattr(self.model, "encoder") and hasattr(
                self.model.encoder, "get_initial_cache_state"
            ):
                cache_state = self.model.encoder.get_initial_cache_state(batch_size=1)
                if cache_state and len(cache_state) == 3:
                    (
                        self._cache_last_channel,
                        self._cache_last_time,
                        self._cache_last_channel_len,
                    ) = cache_state
                else:
                    # Model returned unexpected cache state format
                    self._cache_last_channel = None
                    self._cache_last_time = None
                    self._cache_last_channel_len = None
            else:
                # Fallback: model doesn't support cache-aware streaming
                self._cache_last_channel = None
                self._cache_last_time = None
                self._cache_last_channel_len = None
        except (AttributeError, TypeError, ValueError):
            # Any error accessing encoder methods means fallback mode
            self._cache_last_channel = None
            self._cache_last_time = None
            self._cache_last_channel_len = None

        # Always initialize these
        self._previous_hypotheses = None
        if not hasattr(self, "_full_transcript"):
            self._full_transcript = ""

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

        Uses NeMo's conformer_stream_step() for O(1) per-chunk processing when available,
        falls back to accumulate-and-retranscribe for models without streaming support.
        """
        self._init_cache()

        # Check if model supports cache-aware streaming
        if (
            hasattr(self.model, "conformer_stream_step")
            and hasattr(self.model, "preprocessor")
            and self._cache_last_channel is not None
        ):
            # Prepare audio tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(
                0
            )  # [1, samples]
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
                item = transcribed_texts[0]
                # Handle both string and Hypothesis objects
                if hasattr(item, "text"):
                    new_text = str(item.text)
                else:
                    new_text = str(item)

                # Track stability: the previous full transcript is stable
                # Only the new tokens may change
                stable_len = len(self._full_transcript)
                self._full_transcript = new_text

                return {
                    "text": new_text,
                    "stable_len": stable_len,
                }
        else:
            # Fallback: accumulate and re-transcribe for models without streaming
            if not hasattr(self, "_accumulated_audio"):
                self._accumulated_audio = audio
            else:
                self._accumulated_audio = np.concatenate(
                    [self._accumulated_audio, audio]
                )

            # Transcribe accumulated audio
            with torch.no_grad():
                result = self.model.transcribe([self._accumulated_audio])

            if result and len(result) > 0:
                # Handle different result formats from NeMo models
                item = result[0]
                if hasattr(item, "text"):
                    text = str(item.text)  # Ensure it's a string
                elif hasattr(item, "y_sequence"):
                    # Some models return hypothesis with y_sequence
                    text = str(item)
                else:
                    text = str(item)
                # Track what's stable (previous text length)
                stable_len = len(self._full_transcript)
                self._full_transcript = text
                return {"text": text, "stable_len": stable_len}

        return None

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
        """Finalize the transcription and return complete text."""
        # Minimum samples needed for processing (at least 100ms of audio)
        min_samples = int(self.config.sample_rate * 0.1)  # 1600 samples
        min_bytes = min_samples * 2  # 3200 bytes

        # If we have remaining audio that didn't form a full chunk, process it
        # Only if it's large enough to avoid tensor errors
        if len(self.audio_buffer) >= min_bytes:
            remaining_audio = self._bytes_to_float32(self.audio_buffer)
            self.audio_buffer = b""
            try:
                result = self._transcribe_chunk(remaining_audio)
                if result:
                    self._full_transcript = result["text"]
            except Exception:
                pass  # Ignore errors on small remaining buffers
        else:
            self.audio_buffer = b""  # Clear the buffer anyway

        # For fallback mode, ensure we have the final transcription
        if hasattr(self, "_accumulated_audio") and len(self._accumulated_audio) > 0:
            with torch.no_grad():
                result = self.model.transcribe([self._accumulated_audio])
                if result and len(result) > 0:
                    item = result[0]
                    if hasattr(item, "text"):
                        self._full_transcript = str(item.text)
                    else:
                        self._full_transcript = str(item)

        # Reset cache state for next session
        final_text = self._full_transcript
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._full_transcript = ""

        # Clean up fallback state if present
        if hasattr(self, "_accumulated_audio"):
            delattr(self, "_accumulated_audio")

        return final_text


def load_nemotron_model(model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"):
    """Load the Nemotron streaming ASR model."""
    # Disable CUDA graphs which can cause issues in streaming mode
    import os
    os.environ.setdefault("NEMO_CUDA_GRAPHS", "0")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Configure for streaming with cache-aware inference
    # att_context_size controls the chunk size:
    # [70, 6] = 7 frames = 560ms latency
    # The model's decoding strategy needs to be set for streaming

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Disable any torch.compile optimizations that might interfere
    if hasattr(model, '_compiled'):
        model._compiled = False

    return model


def create_streaming_session(
    model: nemo_asr.models.ASRModel,
    config: StreamingConfig | None = None,
) -> StreamingSession:
    """Create a new streaming session with the given model."""
    if config is None:
        config = StreamingConfig()

    return StreamingSession(config=config, model=model)

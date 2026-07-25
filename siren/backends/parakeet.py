import logging
import time
import warnings
from typing import Any

import torch

from siren.audio import get_wav_info
from siren.concurrency import run_in_worker_thread
from siren.logging_utils import log_event
from siren.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


def get_cuda_stats() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
    }


def parakeet_words(hypothesis: Any) -> list[TranscriptionWord]:
    timestamps = getattr(hypothesis, "timestamp", {}) or {}
    raw_words = timestamps.get("word") or []
    words = [
        TranscriptionWord(
            start=float(raw_word.get("start", 0.0)),
            end=float(raw_word.get("end", 0.0)),
            word=str(raw_word.get("word") or raw_word.get("text") or ""),
        )
        for raw_word in raw_words
        if str(raw_word.get("word") or raw_word.get("text") or "").strip()
    ]
    return sorted(words, key=lambda word: word.start)


def parakeet_segments(
    hypothesis: Any,
    *,
    word_timestamps: bool = False,
) -> list[TranscriptionSegment]:
    timestamps = getattr(hypothesis, "timestamp", {}) or {}
    raw_segments = timestamps.get("segment") or timestamps.get("word") or []
    segments: list[TranscriptionSegment] = []
    for segment in raw_segments:
        text = str(
            segment.get("segment")
            or segment.get("word")
            or segment.get("text")
            or ""
        ).strip()
        if not text:
            continue
        segments.append(
            TranscriptionSegment(
                id=len(segments),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=text,
            )
        )

    if not word_timestamps:
        return segments

    words = parakeet_words(hypothesis)
    if timestamps.get("segment"):
        word_index = 0
        for index, segment in enumerate(segments):
            next_start = (
                segments[index + 1].start
                if index + 1 < len(segments)
                else float("inf")
            )
            segment.words = []
            while word_index < len(words) and words[word_index].start < next_start:
                segment.words.append(words[word_index])
                word_index += 1
    else:
        for segment, word in zip(segments, words, strict=False):
            segment.words = [word]
    return segments


async def process_parakeet_transcription(
    audio_path: str,
    model: Any,
    *,
    word_timestamps: bool = False,
    request_id: str | None = None,
) -> TranscriptionResult:
    audio_info = get_wav_info(audio_path, request_id=request_id)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = await run_in_worker_thread(
        lambda: model.transcribe([audio_path], timestamps=True, verbose=False),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    log_event(
        logging.INFO,
        "parakeet_transcribe",
        request_id=request_id,
        latency_ms=elapsed_ms,
        **audio_info,
        **get_cuda_stats(),
    )
    if not output:
        return TranscriptionResult(
            text="",
            language="en",
            duration=float(audio_info.get("audio_duration_sec", 0.0)),
            segments=[],
        )
    hypothesis = output[0]
    return TranscriptionResult(
        text=str(hypothesis.text),
        language="en",
        duration=float(audio_info.get("audio_duration_sec", 0.0)),
        segments=parakeet_segments(
            hypothesis,
            word_timestamps=word_timestamps,
        ),
    )


class ParakeetBackend:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        return await process_parakeet_transcription(
            audio_path,
            self.model,
            word_timestamps=word_timestamps,
            request_id=request_id,
        )


def load_parakeet_backend(model_name: str) -> ParakeetBackend:
    warnings.filterwarnings("ignore", module="nemo")
    warnings.filterwarnings("ignore", message=".*torchaudio.*")

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return ParakeetBackend(model)

from typing import Any

from faster_whisper import WhisperModel

from siren.concurrency import run_in_worker_thread
from siren.schemas import TranscriptionResult, TranscriptionSegment


async def process_whisper_transcription(
    audio_path: str,
    model: WhisperModel,
    language: str | None = None,
) -> TranscriptionResult:
    def transcribe() -> tuple[list[Any], Any]:
        raw_segments, info = model.transcribe(audio_path, language=language)
        return list(raw_segments), info

    raw_segments, info = await run_in_worker_thread(transcribe)
    segments: list[TranscriptionSegment] = []
    for segment in raw_segments:
        text = segment.text.strip()
        if text:
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
    return TranscriptionResult(
        text=" ".join(segment.text for segment in segments),
        language=str(getattr(info, "language", language or "unknown")),
        duration=float(
            getattr(info, "duration", segments[-1].end if segments else 0.0)
        ),
        segments=segments,
    )


class WhisperBackend:
    def __init__(self, model: WhisperModel) -> None:
        self.model = model

    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None,
        request_id: str | None,
    ) -> TranscriptionResult:
        return await process_whisper_transcription(audio_path, self.model, language)

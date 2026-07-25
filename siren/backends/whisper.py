from typing import Any

from faster_whisper import WhisperModel

from siren.concurrency import run_in_worker_thread
from siren.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


async def process_whisper_transcription(
    audio_path: str,
    model: WhisperModel,
    language: str | None = None,
    *,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    def transcribe() -> tuple[list[Any], Any]:
        kwargs: dict[str, Any] = {"language": language}
        if word_timestamps:
            kwargs["word_timestamps"] = True
        raw_segments, info = model.transcribe(audio_path, **kwargs)
        return list(raw_segments), info

    raw_segments, info = await run_in_worker_thread(transcribe)
    segments: list[TranscriptionSegment] = []
    for segment in raw_segments:
        text = segment.text.strip()
        if text:
            words = None
            if word_timestamps:
                words = sorted(
                    [
                        TranscriptionWord(
                            start=float(word.start),
                            end=float(word.end),
                            word=str(word.word).strip(),
                        )
                        for word in (segment.words or [])
                    ],
                    key=lambda word: word.start,
                )
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    words=words,
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
        language: str | None = None,
        word_timestamps: bool = False,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        return await process_whisper_transcription(
            audio_path,
            self.model,
            language,
            word_timestamps=word_timestamps,
        )

from typing import Protocol

from siren.schemas import TranscriptionResult


class TranscriptionBackend(Protocol):
    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        request_id: str | None = None,
    ) -> TranscriptionResult: ...

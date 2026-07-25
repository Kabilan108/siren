from typing import Protocol

from siren.schemas import TranscriptionResult


class TranscriptionBackend(Protocol):
    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None,
        request_id: str | None,
    ) -> TranscriptionResult: ...

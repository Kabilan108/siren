"""Job-worker adapter for the API process's resident ASR model."""

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from siren import config
from siren.schemas import TranscriptionResult


def validate_resident_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SIREN_JOB_ASR_URL must be an HTTP loopback origin")
    # Accessing port also validates malformed/non-numeric ports.
    _ = parsed.port
    return url.rstrip("/")


class ResidentBackend:
    def __init__(self, client: httpx.AsyncClient, url: str, model_name: str) -> None:
        self.client = client
        self.url = validate_resident_url(url)
        self.model_name = model_name

    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        data = {"model": self.model_name, "response_format": "verbose_json"}
        if word_timestamps:
            data["timestamp_granularities[]"] = "word"
        if language is not None:
            data["language"] = language
        # The worker supplies bounded WAV chunks; read off its event loop.
        audio = await asyncio.to_thread(Path(audio_path).read_bytes)
        response = await self.client.post(
            self.url + "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.TOKEN}"},
            data=data,
            files={"file": (Path(audio_path).name, audio, "audio/wav")},
        )
        if response.status_code != 200:
            # Do not echo request headers or arbitrary server error bodies.
            raise RuntimeError(f"Resident ASR returned HTTP {response.status_code}")
        return TranscriptionResult.model_validate(response.json())

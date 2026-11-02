import logging
import wave
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from siren import config, models
from siren.schemas import TranscriptionResult, TranscriptionSegment
from siren.server import app

TOKEN = "dev_token"
PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER = "distil-small.en"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


def write_wav(path: Path, seconds: float = 2.0) -> str:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(16000 * seconds))
    return str(path)


class Harness:
    def __init__(self, audio_path: str) -> None:
        self.audio_path = audio_path
        self.backend = MagicMock()
        self.backend.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello",
                language="en",
                duration=2.0,
                segments=[TranscriptionSegment(id=0, start=0.0, end=2.0, text="hello")],
            )
        )
        self.get_backend = AsyncMock(return_value=self.backend)
        self.save_upload = AsyncMock(return_value=audio_path)

    async def post(self, client: httpx.AsyncClient, **data: str) -> httpx.Response:
        return await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("clip.wav", b"data", "audio/wav")},
            data=data,
        )


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    h = Harness(write_wav(tmp_path / "clip.wav"))
    with patch("siren.models.get_transcription_backend", h.get_backend), patch(
        "siren.api.transcriptions.save_upload_file", h.save_upload
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav", AsyncMock(return_value=h.audio_path)
    ):
        yield h


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_format,expected",
    [
        ("json", {"text": ""}),
        (
            "verbose_json",
            {"task": "transcribe", "language": "en", "duration": 2.0, "text": "", "segments": []},
        ),
    ],
)
async def test_gate_skips_parakeet_inference_without_speech(
    client: httpx.AsyncClient,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    response_format: str,
    expected: dict[str, object],
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "true")
    gate = MagicMock(return_value=False)
    with patch("siren.api.transcriptions.file_has_speech", gate), caplog.at_level(logging.INFO):
        response = await harness.post(client, model=PARAKEET, response_format=response_format)
    assert response.status_code == 200
    assert response.json() == expected
    gate.assert_called_once_with(harness.audio_path)
    harness.get_backend.assert_not_awaited()
    assert '"event": "transcribe_gated"' in caplog.text


@pytest.mark.asyncio
async def test_gate_passes_speech_through_outside_the_semaphore(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "true")
    held: list[bool] = []

    def gate(path: str) -> bool:
        held.append(models.inference_semaphore.locked())
        return True

    with patch("siren.api.transcriptions.file_has_speech", gate):
        response = await harness.post(client, model=PARAKEET)
    assert response.json() == {"text": "hello"}
    assert held == [False]
    harness.backend.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_failure_falls_through_to_inference(
    client: httpx.AsyncClient,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "true")
    gate = MagicMock(side_effect=ValueError("Speech gate requires finite audio samples"))
    with patch("siren.api.transcriptions.file_has_speech", gate), caplog.at_level(logging.WARNING):
        response = await harness.post(client, model=PARAKEET)
    assert response.json() == {"text": "hello"}
    assert '"event": "speech_gate_failed"' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,setting",
    [(WHISPER, "true"), (PARAKEET, "false")],
)
async def test_gate_is_skipped_for_whisper_and_when_disabled(
    client: httpx.AsyncClient,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    setting: str,
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", setting)
    gate = MagicMock(return_value=False)
    with patch("siren.api.transcriptions.file_has_speech", gate):
        response = await harness.post(client, model=model)
    assert response.json() == {"text": "hello"}
    gate.assert_not_called()


@pytest.mark.asyncio
async def test_gate_defaults_on_for_parakeet(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIREN_PARAKEET_SPEECH_GATE")
    gate = MagicMock(return_value=False)
    with patch("siren.api.transcriptions.file_has_speech", gate):
        response = await harness.post(client, model=PARAKEET)
    assert response.json() == {"text": ""}
    gate.assert_called_once()


@pytest.mark.asyncio
async def test_real_gate_rejects_digital_silence(
    client: httpx.AsyncClient, harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "true")
    response = await harness.post(client, model=PARAKEET)
    assert response.json() == {"text": ""}
    harness.get_backend.assert_not_awaited()


def test_parakeet_speech_gate_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIREN_PARAKEET_SPEECH_GATE")
    assert config.parakeet_speech_gate_enabled() is True
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "0")
    assert config.parakeet_speech_gate_enabled() is False
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "typo")
    with pytest.raises(ValueError, match="SIREN_PARAKEET_SPEECH_GATE"):
        config.parakeet_speech_gate_enabled()

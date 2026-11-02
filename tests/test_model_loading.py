import logging
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from siren import models
from siren.schemas import TranscriptionResult

PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"


@pytest.mark.asyncio
async def test_ensure_model_loaded_reuses_model():
    sentinel = object()
    with patch(
        "siren.models.load_backend", MagicMock(return_value=sentinel)
    ) as load_backend:
        backend = await models.ensure_model_loaded("distil-small.en")
        assert backend is sentinel
        assert models.current_model_name == "distil-small.en"

        backend_again = await models.ensure_model_loaded("distil-small.en")
        assert backend_again is sentinel
        assert load_backend.call_count == 1


@pytest.mark.asyncio
async def test_ensure_model_loaded_error():
    with patch("siren.models.load_backend", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            await models.ensure_model_loaded("distil-small.en")


@pytest.mark.asyncio
async def test_model_load_warms_up_before_publishing_the_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[tuple[int, int, float, bool]] = []
    paths: list[str] = []
    backend = MagicMock()

    async def transcribe(path: str, **kwargs: object) -> TranscriptionResult:
        paths.append(path)
        with wave.open(path, "rb") as wav_file:
            seen.append(
                (
                    wav_file.getframerate(),
                    wav_file.getnchannels(),
                    wav_file.getnframes() / wav_file.getframerate(),
                    models.current_backend is backend,
                )
            )
        return TranscriptionResult(text="", language="en", duration=1.5, segments=[])

    backend.transcribe = transcribe
    with patch("siren.models.load_backend", MagicMock(return_value=backend)), caplog.at_level(
        logging.INFO
    ):
        assert await models.ensure_model_loaded(PARAKEET) is backend
    assert seen == [(16000, 1, 1.5, False)]
    assert '"event": "model_warmup_complete"' in caplog.text
    assert not Path(paths[0]).exists()


@pytest.mark.asyncio
async def test_warmup_failure_does_not_block_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = MagicMock()
    backend.transcribe = AsyncMock(side_effect=RuntimeError("cuDNN exploded"))
    with patch("siren.models.load_backend", MagicMock(return_value=backend)), caplog.at_level(
        logging.WARNING
    ):
        assert await models.ensure_model_loaded(PARAKEET) is backend
    assert models.current_model_name == PARAKEET
    assert '"event": "model_warmup_failed"' in caplog.text
    assert "cuDNN exploded" in caplog.text

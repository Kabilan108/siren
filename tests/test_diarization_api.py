import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException

from siren.diarization import DEFAULT_DIARIZATION_MODEL
from siren.server import app

TOKEN = "dev_token"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_diarize_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/diarize",
        files={"file": ("test.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_diarize_rejects_wrong_model(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/diarize",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"model": "wrong/model"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Invalid model: 'wrong/model'. This endpoint supports only "
        f"'{DEFAULT_DIARIZATION_MODEL}'."
    )


@pytest.mark.asyncio
async def test_diarize_success_shape_and_cleanup(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    result = {
        "duration": 4.25,
        "model": DEFAULT_DIARIZATION_MODEL,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.25, "speaker": "SPEAKER_01"},
        ],
    }

    with patch(
        "siren.api.diarize.save_upload_file",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.ensure_16k_wav",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.get_wav_info",
        return_value={
            "audio_frames": 68000,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "audio_duration_sec": 4.25,
        },
    ), patch(
        "siren.api.diarize.run_diarization",
        AsyncMock(return_value=result),
    ) as run_diarization:
        response = await client.post(
            "/v1/audio/diarize",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == result
    run_diarization.assert_awaited_once()
    await_args = run_diarization.await_args
    assert await_args is not None
    assert await_args.args == (str(audio_path),)
    assert "request_id" in await_args.kwargs
    assert not audio_path.exists()


@pytest.mark.asyncio
async def test_diarize_cleans_original_and_converted_files_on_failure(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "audio.mp3"
    converted_path = tmp_path / "audio.wav"
    original_path.write_bytes(b"original")
    converted_path.write_bytes(b"converted")

    with patch(
        "siren.api.diarize.save_upload_file",
        AsyncMock(return_value=str(original_path)),
    ), patch(
        "siren.api.diarize.ensure_16k_wav",
        AsyncMock(return_value=str(converted_path)),
    ), patch(
        "siren.api.diarize.get_wav_info",
        return_value={"audio_duration_sec": 1.0},
    ), patch(
        "siren.api.diarize.run_diarization",
        AsyncMock(
            side_effect=HTTPException(
                status_code=500,
                detail="Diarization worker failed",
            )
        ),
    ):
        response = await client.post(
            "/v1/audio/diarize",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.mp3", b"audio", "audio/mpeg")},
        )

    assert response.status_code == 500
    assert not original_path.exists()
    assert not converted_path.exists()


@pytest.mark.asyncio
async def test_disconnect_before_dispatch_does_not_spawn_worker(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    run_diarization = AsyncMock()

    with patch(
        "siren.api.diarize.save_upload_file",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.ensure_16k_wav",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.get_wav_info",
        return_value={"audio_duration_sec": 1.0},
    ), patch(
        "siren.api.diarize.Request.is_disconnected",
        AsyncMock(return_value=True),
    ), patch(
        "siren.api.diarize.run_diarization",
        run_diarization,
    ):
        response = await client.post(
            "/v1/audio/diarize",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 499
    assert response.content == b""
    run_diarization.assert_not_awaited()
    assert not audio_path.exists()


@pytest.mark.asyncio
async def test_disconnect_while_running_cancels_supervisor_task(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    cancelled = asyncio.Event()

    async def run_until_cancelled(
        _audio_path: str,
        *,
        request_id: str,
    ) -> dict[str, object]:
        del request_id
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return {}

    with patch(
        "siren.api.diarize.save_upload_file",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.ensure_16k_wav",
        AsyncMock(return_value=str(audio_path)),
    ), patch(
        "siren.api.diarize.get_wav_info",
        return_value={"audio_duration_sec": 1.0},
    ), patch(
        "siren.api.diarize.Request.is_disconnected",
        AsyncMock(side_effect=[False, True]),
    ), patch(
        "siren.api.diarize._DISCONNECT_POLL_SECONDS",
        0.01,
    ), patch(
        "siren.api.diarize.run_diarization",
        run_until_cancelled,
    ):
        response = await client.post(
            "/v1/audio/diarize",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 499
    assert response.content == b""
    assert cancelled.is_set()
    assert not audio_path.exists()

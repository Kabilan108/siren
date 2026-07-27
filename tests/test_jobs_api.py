from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from siren.jobs.runner import JobQueueFullError, JobUploadTooLargeError
from siren.server import app

TOKEN = "dev_token"
JOB_ID = "job_0123456789abcdef0123456789abcdef"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_job_endpoints_require_authentication(
    client: httpx.AsyncClient,
) -> None:
    responses = [
        await client.post(
            "/v1/jobs/transcripts",
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
        ),
        await client.get(f"/v1/jobs/transcripts/{JOB_ID}"),
        await client.get(f"/v1/jobs/transcripts/{JOB_ID}/result"),
    ]

    assert [response.status_code for response in responses] == [401, 401, 401]


@pytest.mark.asyncio
async def test_create_job_returns_accepted_shape(
    client: httpx.AsyncClient,
) -> None:
    with patch(
        "siren.api.jobs.models.resolve_transcription_model_name",
        return_value="nvidia/parakeet-tdt-0.6b-v2",
    ), patch(
        "siren.api.jobs.job_runner.enqueue",
        AsyncMock(return_value=(JOB_ID, 3)),
    ) as enqueue:
        response = await client.post(
            "/v1/jobs/transcripts",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
            data={"model": "nvidia/parakeet-tdt-0.6b-v2", "language": "en"},
        )

    assert response.status_code == 202
    assert response.json() == {
        "id": JOB_ID,
        "status": "queued",
        "position": 3,
    }
    await_args = enqueue.await_args
    assert await_args is not None
    assert await_args.kwargs == {
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "language": "en",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (
            JobUploadTooLargeError(123),
            413,
            "Upload exceeds maximum size of 123 bytes",
        ),
        (
            JobQueueFullError(8),
            429,
            "Job queue is full (maximum 8 queued jobs)",
        ),
    ],
)
async def test_create_job_maps_admission_limits_to_http_errors(
    client: httpx.AsyncClient,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    with patch(
        "siren.api.jobs.models.resolve_transcription_model_name",
        return_value="nvidia/parakeet-tdt-0.6b-v2",
    ), patch(
        "siren.api.jobs.job_runner.enqueue",
        AsyncMock(side_effect=error),
    ):
        response = await client.post(
            "/v1/jobs/transcripts",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


@pytest.mark.asyncio
async def test_unknown_job_returns_404_for_status_and_result(
    client: httpx.AsyncClient,
) -> None:
    with patch(
        "siren.api.jobs.job_runner.status",
        AsyncMock(return_value=None),
    ):
        status_response = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        result_response = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}/result",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert status_response.status_code == 404
    assert status_response.json() == {"detail": "Job not found"}
    assert result_response.status_code == 404
    assert result_response.json() == {"detail": "Job not found"}


@pytest.mark.asyncio
async def test_noncompleted_result_returns_409(
    client: httpx.AsyncClient,
) -> None:
    with patch(
        "siren.api.jobs.job_runner.status",
        AsyncMock(
            return_value={
                "id": JOB_ID,
                "status": "running",
                "phase": "transcribing",
                "progress": 0.5,
            }
        ),
    ):
        response = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}/result",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "Job is not completed"}


@pytest.mark.asyncio
async def test_job_status_fields_are_conditional(
    client: httpx.AsyncClient,
) -> None:
    status_mock = AsyncMock(
        side_effect=[
            {
                "id": JOB_ID,
                "status": "queued",
                "phase": None,
                "progress": 0.0,
                "position": 2,
            },
            {
                "id": JOB_ID,
                "status": "failed",
                "phase": None,
                "progress": 0.75,
                "error": "model failed",
            },
        ]
    )
    with patch("siren.api.jobs.job_runner.status", status_mock):
        queued = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        failed = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert queued.json() == {
        "id": JOB_ID,
        "status": "queued",
        "phase": None,
        "progress": 0.0,
        "position": 2,
    }
    assert failed.json() == {
        "id": JOB_ID,
        "status": "failed",
        "phase": None,
        "progress": 0.75,
        "error": "model failed",
    }


@pytest.mark.asyncio
async def test_completed_result_schema_omits_word_arrays(
    client: httpx.AsyncClient,
) -> None:
    result = {
        "text": "hello",
        "language": "en",
        "duration": 1.0,
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "diarization_model": "test/sortformer",
        "speakers": ["SPEAKER_00"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": "hello",
                "words": [{"word": "hello"}],
            }
        ],
    }
    with patch(
        "siren.api.jobs.job_runner.status",
        AsyncMock(
            return_value={
                "id": JOB_ID,
                "status": "completed",
                "phase": None,
                "progress": 1.0,
            }
        ),
    ), patch(
        "siren.api.jobs.job_runner.result",
        MagicMock(return_value=result),
    ):
        response = await client.get(
            f"/v1/jobs/transcripts/{JOB_ID}/result",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "hello",
        "language": "en",
        "duration": 1.0,
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "diarization_model": "test/sortformer",
        "speakers": ["SPEAKER_00"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": "hello",
            }
        ],
    }

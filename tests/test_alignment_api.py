from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from siren.alignment import align_words
from siren.schemas import AlignmentResponse
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
async def test_align_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/align",
        json={"words": [], "turns": []},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_align_returns_speaker_aware_segments(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/audio/align",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hello"},
                {"start": 0.5, "end": 0.9, "word": "there"},
            ],
            "turns": [
                {"start": 0.0, "end": 0.45, "speaker": "SPEAKER_01"},
                {"start": 0.45, "end": 1.0, "speaker": "SPEAKER_00"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 0.4,
                "speaker": "SPEAKER_01",
                "text": "hello",
                "words": [
                    {
                        "start": 0.0,
                        "end": 0.4,
                        "word": "hello",
                        "speaker": "SPEAKER_01",
                    }
                ],
            },
            {
                "id": 1,
                "start": 0.5,
                "end": 0.9,
                "speaker": "SPEAKER_00",
                "text": "there",
                "words": [
                    {
                        "start": 0.5,
                        "end": 0.9,
                        "word": "there",
                        "speaker": "SPEAKER_00",
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_align_empty_words_returns_empty_response(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/audio/align",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"words": [], "turns": []},
    )

    assert response.status_code == 200
    assert response.json() == {"speakers": [], "segments": []}


@pytest.mark.asyncio
async def test_align_runs_computation_in_worker_thread(
    client: httpx.AsyncClient,
) -> None:
    expected = AlignmentResponse(speakers=[], segments=[])
    with patch(
        "siren.api.align.run_in_worker_thread",
        AsyncMock(return_value=expected),
    ) as run_in_worker_thread:
        response = await client.post(
            "/v1/audio/align",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"words": [], "turns": []},
        )

    assert response.status_code == 200
    run_in_worker_thread.assert_awaited_once_with(align_words, [], [])


@pytest.mark.asyncio
async def test_align_rejects_payload_over_size_bound(
    client: httpx.AsyncClient,
) -> None:
    with patch(
        "siren.api.align.run_in_worker_thread",
        AsyncMock(),
    ) as run_in_worker_thread:
        response = await client.post(
            "/v1/audio/align",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "words": [],
                "turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}] * 50_001,
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Too many turns: maximum is 50000."}
    run_in_worker_thread.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (
            {
                "words": [],
                "turns": [{"start": 2.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            },
            "turns[0] has end before start.",
        ),
    ],
)
async def test_align_rejects_invalid_intervals(
    client: httpx.AsyncClient,
    body: dict[str, object],
    detail: str,
) -> None:
    response = await client.post(
        "/v1/audio/align",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}


@pytest.mark.asyncio
async def test_align_rejects_non_finite_intervals(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/audio/align",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
        content=(b'{"words":[{"start":Infinity,"end":1.0,"word":"bad"}],"turns":[]}'),
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "words[0] has non-finite start or end values."}

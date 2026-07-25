import io
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from faster_whisper import WhisperModel

from siren.audio import save_upload_file
from siren.backends.parakeet import parakeet_segments
from siren.backends.whisper import process_whisper_transcription
from siren.models import get_whisper_params
from siren.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from siren.server import app

TOKEN = "dev_token"
VALID_MODEL = "distil-small.en"
INVALID_MODEL = "invalid-model"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Create a test client with overridden dependencies"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as test_client:
        yield test_client


def transcription_result(text: str = "ok") -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language="en",
        duration=1.5,
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.5, text=text)],
    )


@pytest.mark.asyncio
async def test_list_models_success(client):
    """Test listing available models with valid Bearer token"""
    response = await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert all("id" in model for model in data["data"])
    assert VALID_MODEL in [model["id"] for model in data["data"]]
    assert any(model["id"].startswith("nvidia/parakeet") for model in data["data"])


@pytest.mark.asyncio
async def test_health_reports_siren_version(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.1.0"}


@pytest.mark.asyncio
async def test_list_models_unauthorized(client):
    """Test listing models without Bearer token"""
    response = await client.get("/v1/models")

    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_models_invalid_scheme(client):
    """Test listing models with invalid authentication scheme"""
    response = await client.get(
        "/v1/models",
        headers={"Authorization": f"Basic {TOKEN}"},
    )

    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_models_invalid_token(client):
    """Test listing models with invalid token"""
    response = await client.get(
        "/v1/models", headers={"Authorization": "Bearer invalid_token"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token."


@pytest.mark.asyncio
async def test_transcribe_audio_success(client, tmp_path):
    """Test audio transcription with valid input (Whisper path)."""
    expected_text = (
        "And so, my fellow Americans, ask not what your country can do for you. "
        "Ask what you can do for your country."
    )
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"RIFF$\x00\x00\x00WAVEfmt ")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result(expected_text))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ) as get_backend, patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={"model": "distil-large-v3", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == expected_text
        get_backend.assert_awaited_once()
        backend.transcribe.assert_awaited_once()
        assert backend.transcribe.call_args.args == (str(temp_audio),)
        assert backend.transcribe.call_args.kwargs["language"] == "en"
        assert "request_id" in backend.transcribe.call_args.kwargs

    assert not temp_audio.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_parakeet_route(client, tmp_path):
    """Test audio transcription with Parakeet model routes correctly."""
    temp_audio = tmp_path / "audio.mp3"
    temp_audio.write_bytes(b"fake")
    expected_text = "hello parakeet"

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result(expected_text))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ) as get_backend, patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.mp3", b"data", "audio/mpeg")},
            data={"model": "nvidia/parakeet-tdt-0.6b-v2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == expected_text
        get_backend.assert_awaited_once()
        backend.transcribe.assert_awaited_once()
        assert backend.transcribe.call_args.args == (str(temp_audio),)
        assert backend.transcribe.call_args.kwargs["language"] is None
        assert "request_id" in backend.transcribe.call_args.kwargs

    assert not temp_audio.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_cleans_converted_files(client, tmp_path):
    """Test cleanup for original and converted files."""
    original = tmp_path / "original.mp3"
    converted = tmp_path / "converted.wav"
    original.write_bytes(b"fake")
    converted.write_bytes(b"converted")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result())

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(original)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(converted)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.mp3", b"data", "audio/mpeg")},
            data={"model": "distil-large-v3", "language": "en"},
        )
        assert response.status_code == 200

    assert not original.exists()
    assert not converted.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_default_verbose_json_has_no_words(client, tmp_path):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result("timestamped"))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "nvidia/parakeet-tdt-0.6b-v2",
                "response_format": "verbose_json",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "task": "transcribe",
        "language": "en",
        "duration": 1.5,
        "text": "timestamped",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "timestamped"}
        ],
    }
    assert backend.transcribe.call_args.kwargs["word_timestamps"] is False


@pytest.mark.asyncio
async def test_transcribe_audio_pause_segmentation_uses_backend_words(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")
    result = transcription_result("one two three")
    result.segments[0].end = 32.0
    result.segments[0].words = [
        TranscriptionWord(start=0.0, end=0.4, word="one"),
        TranscriptionWord(start=0.5, end=0.9, word="two"),
        TranscriptionWord(start=31.0, end=31.5, word="three"),
    ]

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=result)

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "nvidia/parakeet-tdt-0.6b-v2",
                "response_format": "verbose_json",
                "segmentation": "pause",
            },
        )

    assert response.status_code == 200
    assert response.json()["segments"] == [
        {"id": 0, "start": 0.0, "end": 0.9, "text": "one two"},
        {"id": 1, "start": 31.0, "end": 31.5, "text": "three"},
    ]
    assert backend.transcribe.call_args.kwargs["word_timestamps"] is True


@pytest.mark.asyncio
async def test_transcribe_audio_pause_segmentation_requires_verbose_json(client):
    response = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", b"data", "audio/wav")},
        data={
            "model": "nvidia/parakeet-tdt-0.6b-v2",
            "segmentation": "pause",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "segmentation=pause requires response_format=verbose_json."
    )


@pytest.mark.asyncio
async def test_transcribe_audio_unknown_segmentation_is_rejected(client):
    response = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", b"data", "audio/wav")},
        data={"segmentation": "sentence"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid segmentation: 'sentence'. Supported values: 'native', 'pause'."
    )


@pytest.mark.asyncio
async def test_transcribe_audio_pause_segmentation_rejects_whisper(client):
    response = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", b"data", "audio/wav")},
        data={
            "model": VALID_MODEL,
            "response_format": "verbose_json",
            "segmentation": "pause",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "segmentation=pause is currently supported only for Parakeet models"
    )


@pytest.mark.asyncio
async def test_transcribe_audio_word_granularity_returns_ordered_words(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")
    result = transcription_result("two words")
    result.segments[0].words = [
        TranscriptionWord(start=0.1, end=0.4, word="two"),
        TranscriptionWord(start=0.5, end=0.9, word=" words"),
    ]

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=result)

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "distil-large-v3",
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["segment", "word"],
            },
        )

    assert response.status_code == 200
    words = response.json()["segments"][0]["words"]
    assert words == [
        {"start": 0.1, "end": 0.4, "word": "two"},
        {"start": 0.5, "end": 0.9, "word": " words"},
    ]
    assert [word["start"] for word in words] == sorted(
        word["start"] for word in words
    )
    assert backend.transcribe.call_args.kwargs["word_timestamps"] is True


@pytest.mark.asyncio
async def test_transcribe_audio_pause_falls_back_to_native_without_words(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    result = transcription_result("no-words")
    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=result)

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "nvidia/parakeet-tdt-0.6b-v2",
                "response_format": "verbose_json",
                "segmentation": "pause",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert [segment["text"] for segment in payload["segments"]] == ["no-words"]


@pytest.mark.asyncio
async def test_transcribe_audio_strips_unrequested_backend_words(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    result = transcription_result("with-words")
    result.segments[0].words = [TranscriptionWord(start=0.0, end=0.5, word="with")]
    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=result)

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={"model": "distil-large-v3", "response_format": "verbose_json"},
        )

    assert response.status_code == 200
    assert all("words" not in segment for segment in response.json()["segments"])


@pytest.mark.asyncio
async def test_transcribe_audio_word_granularity_with_json_format_is_rejected(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result("plain"))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "distil-large-v3",
                "response_format": "json",
                "timestamp_granularities[]": "word",
            },
        )

    assert response.status_code == 400
    assert "verbose_json" in response.json()["detail"]
    backend.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_audio_unknown_timestamp_granularity_is_rejected(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result("timestamped"))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={
                "model": "distil-large-v3",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "sentence",
            },
        )

    assert response.status_code == 400
    assert "sentence" in response.json()["detail"]
    backend.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_audio_preserves_legacy_response_format_behavior(
    client,
    tmp_path,
):
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"fake")

    backend = MagicMock()
    backend.transcribe = AsyncMock(return_value=transcription_result("legacy"))

    with patch(
        "siren.models.get_transcription_backend",
        AsyncMock(return_value=backend),
    ), patch(
        "siren.api.transcriptions.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.api.transcriptions.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ):
        response = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={"model": "distil-large-v3", "response_format": "text"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "legacy"}


@pytest.mark.asyncio
async def test_save_upload_file_streams_chunks(tmp_path):
    upload = MagicMock()
    upload.filename = "meeting.flac"
    upload.read = AsyncMock(side_effect=[b"first", b"second", b""])

    saved_path = await save_upload_file(upload)
    try:
        assert Path(saved_path).read_bytes() == b"firstsecond"
        assert upload.read.await_count == 3
        assert all(call.args == (1024 * 1024,) for call in upload.read.await_args_list)
    finally:
        Path(saved_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_upload_file_removes_partial_file(tmp_path):
    upload = MagicMock()
    upload.filename = "meeting.flac"
    upload.read = AsyncMock(side_effect=[b"partial", OSError("disk full")])
    saved_path = tmp_path / "partial.flac"

    with patch(
        "siren.audio.tempfile.NamedTemporaryFile",
        return_value=saved_path.open("w+b"),
    ), pytest.raises(HTTPException):
        await save_upload_file(upload)

    assert not saved_path.exists()


@pytest.mark.asyncio
async def test_whisper_generator_is_consumed_off_event_loop():
    loop_thread = threading.get_ident()
    decode_threads = []
    model = MagicMock(spec=WhisperModel)

    def segments():
        decode_threads.append(threading.get_ident())
        yield type(
            "Segment", (), {"text": "hello", "start": 0.0, "end": 1.0}
        )()

    model.transcribe.return_value = (
        segments(),
        type("Info", (), {"language": "en", "duration": 1.0})(),
    )

    result = await process_whisper_transcription("audio.wav", model)

    assert result.text == "hello"
    assert decode_threads and decode_threads[0] != loop_thread
    assert "word_timestamps" not in model.transcribe.call_args.kwargs


@pytest.mark.asyncio
async def test_whisper_word_timestamps_are_requested_and_ordered():
    model = MagicMock(spec=WhisperModel)
    segment = type(
        "Segment",
        (),
        {
            "text": "first second",
            "start": 0.0,
            "end": 1.0,
            "words": [
                type(
                    "Word",
                    (),
                    {"word": " second", "start": 0.5, "end": 0.8},
                )(),
                type(
                    "Word",
                    (),
                    {"word": "first", "start": 0.1, "end": 0.4},
                )(),
            ],
        },
    )()
    model.transcribe.return_value = (
        iter([segment]),
        type("Info", (), {"language": "en", "duration": 1.0})(),
    )

    result = await process_whisper_transcription(
        "audio.wav",
        model,
        word_timestamps=True,
    )

    assert model.transcribe.call_args.kwargs["word_timestamps"] is True
    assert result.segments[0].words is not None
    assert [word.start for word in result.segments[0].words] == [0.1, 0.5]
    assert [word.word for word in result.segments[0].words] == [
        "first",
        "second",
    ]


def test_parakeet_segments_prefers_sentence_segments():
    hypothesis = type(
        "Hypothesis",
        (),
        {
            "timestamp": {
                "segment": [
                    {"segment": "", "start": 0.0, "end": 0.1},
                    {"segment": "First sentence.", "start": 0.2, "end": 1.4},
                    {"segment": "Second sentence.", "start": 1.6, "end": 3.0},
                ],
                "word": [{"word": "First", "start": 0.2, "end": 0.5}],
            }
        },
    )()

    assert [
        segment.model_dump(exclude_none=True)
        for segment in parakeet_segments(hypothesis)
    ] == [
        {"id": 0, "start": 0.2, "end": 1.4, "text": "First sentence."},
        {"id": 1, "start": 1.6, "end": 3.0, "text": "Second sentence."},
    ]


def test_parakeet_segments_include_ordered_words_when_requested():
    hypothesis = type(
        "Hypothesis",
        (),
        {
            "timestamp": {
                "segment": [
                    {"segment": "First sentence.", "start": 0.0, "end": 1.0},
                    {"segment": "Second sentence.", "start": 1.0, "end": 2.0},
                ],
                "word": [
                    {"word": "sentence", "start": 0.4, "end": 0.8},
                    {"word": "First", "start": 0.1, "end": 0.3},
                    {"word": "Second", "start": 1.1, "end": 1.4},
                ],
            }
        },
    )()

    segments = parakeet_segments(hypothesis, word_timestamps=True)

    assert segments[0].words is not None
    assert [word.start for word in segments[0].words] == [0.1, 0.4]
    assert segments[1].words is not None
    assert [word.word for word in segments[1].words] == ["Second"]


@pytest.mark.asyncio
async def test_transcribe_audio_invalid_model(client):
    """Test transcription with invalid model name"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", audio_data, "audio/wav")},
        data={"model": INVALID_MODEL},
    )

    assert response.status_code == 404
    assert "Invalid model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_audio_unauthorized(client):
    """Test transcription without Bearer token"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_data, "audio/wav")},
    )

    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_whisper_params():
    """Test whisper parameters based on CUDA availability"""
    with patch("torch.cuda.is_available") as mock_cuda:
        # Test with CUDA available
        mock_cuda.return_value = True
        params = get_whisper_params()
        assert params["device"] == "cuda"
        assert params["compute_type"] == "float16"

        # Test without CUDA
        mock_cuda.return_value = False
        params = get_whisper_params()
        assert params["device"] == "cpu"
        assert params["compute_type"] == "int8"


if __name__ == "__main__":
    pytest.main([__file__])

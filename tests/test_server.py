import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from siren.server import (
    WhisperModel,
    app,
    get_whisper_params,
)

TOKEN = "dev_token"
VALID_MODEL = "distil-small.en"
INVALID_MODEL = "invalid-model"


@pytest.fixture
def client():
    """Create a test client with overridden dependencies"""
    return TestClient(app)


@pytest.fixture
def mock_whisper_model():
    """Mock WhisperModel instance"""
    mock = MagicMock(spec=WhisperModel)
    # Create a mock segment with the expected text
    expected_text = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."
    mock.transcribe.return_value = (
        [type("Segment", (), {"text": expected_text})()],  # Mock segments
        {"language": "en"},  # Mock info
    )
    return mock


def test_list_models_success(client):
    """Test listing available models with valid Bearer token"""
    response = client.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert all("id" in model for model in data["data"])
    assert VALID_MODEL in [model["id"] for model in data["data"]]
    assert any(model["id"].startswith("nvidia/parakeet") for model in data["data"])


def test_list_models_unauthorized(client):
    """Test listing models without Bearer token"""
    response = client.get("/v1/models")

    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


def test_list_models_invalid_scheme(client):
    """Test listing models with invalid authentication scheme"""
    response = client.get("/v1/models", headers={"Authorization": f"Basic {TOKEN}"})

    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


def test_list_models_invalid_token(client):
    """Test listing models with invalid token"""
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer invalid_token"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token."


@pytest.mark.asyncio
async def test_transcribe_audio_success(client, mock_whisper_model, tmp_path):
    """Test audio transcription with valid input (Whisper path)."""
    expected_text = (
        "And so, my fellow Americans, ask not what your country can do for you. "
        "Ask what you can do for your country."
    )
    temp_audio = tmp_path / "audio.wav"
    temp_audio.write_bytes(b"RIFF$\x00\x00\x00WAVEfmt ")

    with patch(
        "siren.server.get_transcription_model",
        AsyncMock(return_value=mock_whisper_model),
    ) as get_model, patch(
        "siren.server.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.server.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.server.process_whisper_transcription",
        AsyncMock(return_value=expected_text),
    ) as process_whisper:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.wav", b"data", "audio/wav")},
            data={"model": "distil-large-v3", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == expected_text
        get_model.assert_awaited_once()
        process_whisper.assert_awaited_once()
        called_path, called_model, called_language = process_whisper.call_args.args
        assert called_path == str(temp_audio)
        assert called_model is mock_whisper_model
        assert called_language == "en"

    assert not temp_audio.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_parakeet_route(client, tmp_path):
    """Test audio transcription with Parakeet model routes correctly."""
    temp_audio = tmp_path / "audio.mp3"
    temp_audio.write_bytes(b"fake")
    expected_text = "hello parakeet"

    with patch(
        "siren.server.get_transcription_model",
        AsyncMock(return_value=MagicMock()),
    ) as get_model, patch(
        "siren.server.save_upload_file",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.server.ensure_16k_wav",
        AsyncMock(return_value=str(temp_audio)),
    ), patch(
        "siren.server.process_parakeet_transcription",
        AsyncMock(return_value=expected_text),
    ) as process_parakeet:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.mp3", b"data", "audio/mpeg")},
            data={"model": "nvidia/parakeet-tdt-0.6b-v2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == expected_text
        get_model.assert_awaited_once()
        process_parakeet.assert_awaited_once()
        called_path, called_model = process_parakeet.call_args.args
        assert called_path == str(temp_audio)
        assert called_model is get_model.return_value
        assert "request_id" in process_parakeet.call_args.kwargs

    assert not temp_audio.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_cleans_converted_files(client, tmp_path):
    """Test cleanup for original and converted files."""
    original = tmp_path / "original.mp3"
    converted = tmp_path / "converted.wav"
    original.write_bytes(b"fake")
    converted.write_bytes(b"converted")

    with patch(
        "siren.server.get_transcription_model",
        AsyncMock(return_value=MagicMock()),
    ), patch(
        "siren.server.save_upload_file",
        AsyncMock(return_value=str(original)),
    ), patch(
        "siren.server.ensure_16k_wav",
        AsyncMock(return_value=str(converted)),
    ), patch(
        "siren.server.process_whisper_transcription",
        AsyncMock(return_value="ok"),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("test.mp3", b"data", "audio/mpeg")},
            data={"model": "distil-large-v3", "language": "en"},
        )
        assert response.status_code == 200

    assert not original.exists()
    assert not converted.exists()


@pytest.mark.asyncio
async def test_transcribe_audio_invalid_model(client):
    """Test transcription with invalid model name"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": ("test.wav", audio_data, "audio/wav")},
        params={"model": INVALID_MODEL},
    )

    assert response.status_code == 404
    assert "Invalid model" in response.json()["detail"]


def test_transcribe_audio_unauthorized(client):
    """Test transcription without Bearer token"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = client.post(
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

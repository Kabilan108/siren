import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from server import (
    WhisperModel,
    app,
    get_whisper_params,
)

# Constants
API_KEY = "dev_key"
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
    mock.transcribe.return_value = (
        [type("Segment", (), {"text": "Hello world"})()],  # Mock segments
        {"language": "en"},  # Mock info
    )
    return mock


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch, tmp_path):
    """Set up environment variables and config file"""
    monkeypatch.setenv("API_KEY", API_KEY)
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("server.CONFIG_FILE", config_file)
    return config_file


def test_list_models_success(client):
    """Test listing available models with valid API key"""
    response = client.get("/v1/models", headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert all("id" in model for model in data["data"])
    assert VALID_MODEL in [model["id"] for model in data["data"]]


def test_list_models_unauthorized(client):
    """Test listing models without API key"""
    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


@pytest.mark.asyncio
async def test_transcribe_audio_success(client, mock_whisper_model):
    """Test audio transcription with valid input"""
    with patch("server.get_whisper_model", AsyncMock(return_value=mock_whisper_model)):
        with open("test.wav", "rb") as audio_file:
            audio_data = audio_file.read()
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.wav", audio_data, "audio/wav")},
            params={"model": "distil-large-v3"},  # Explicitly set a valid model
        )
        print(response.text)  # Debug output
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."


@pytest.mark.asyncio
async def test_transcribe_audio_invalid_model(client):
    """Test transcription with invalid model name"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = client.post(
        "/v1/audio/transcriptions",
        headers={"X-API-Key": API_KEY},
        files={"file": ("test.wav", audio_data, "audio/wav")},
        params={"model": INVALID_MODEL},
    )

    assert response.status_code == 404
    assert "Invalid model" in response.json()["detail"]


def test_transcribe_audio_unauthorized(client):
    """Test transcription without API key"""
    audio_data = io.BytesIO(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_data, "audio/wav")},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


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

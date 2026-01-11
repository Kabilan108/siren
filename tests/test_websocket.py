"""Tests for WebSocket streaming transcription."""

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from siren.server import app


def generate_silent_audio(duration_ms: int = 100, sample_rate: int = 16000) -> str:
    """Generate base64-encoded silent PCM audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Silent audio = all zeros
    pcm_bytes = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return base64.b64encode(pcm_bytes).decode("ascii")


# Create a mock model that returns empty transcriptions
def create_mock_model():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = [MagicMock(text="")]
    return mock_model


class TestWebSocketAuth:
    """Test WebSocket authentication."""

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_auth_via_query_param(self, mock_get_model):
        """Test authentication via query parameter."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"

    def test_auth_failure(self):
        """Test authentication failure."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=wrong") as ws:
                response = ws.receive_json()
                assert response["type"] == "error"
                assert response["code"] == "AUTH_FAILED"


class TestWebSocketStreaming:
    """Test WebSocket streaming transcription."""

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_config_message(self, mock_get_model):
        """Test config message updates chunk size."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "config", "chunk_frames": 14})
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_audio_message(self, mock_get_model):
        """Test sending audio and receiving partials."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                # Send enough audio for one chunk (560ms at 16kHz)
                audio = generate_silent_audio(duration_ms=600)
                ws.send_json({"type": "audio", "data": audio, "seq": 1})

                # Should get a partial result (may be empty for silent audio)
                # Then send end
                ws.send_json({"type": "end"})

                # Collect all responses until final
                responses = []
                while True:
                    response = ws.receive_json()
                    responses.append(response)
                    if response["type"] == "final":
                        break

                assert any(r["type"] == "final" for r in responses)

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_invalid_message_type(self, mock_get_model):
        """Test unknown message type returns error."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "unknown"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert response["code"] == "INVALID_MESSAGE"


class TestWebSocketProtocol:
    """Test WebSocket protocol edge cases."""

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_empty_audio(self, mock_get_model):
        """Test empty audio stream."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                ws.send_json({"type": "end"})
                response = ws.receive_json()
                assert response["type"] == "final"
                assert response["text"] == ""

    @patch("siren.server.get_streaming_model", new_callable=AsyncMock)
    def test_multiple_audio_chunks(self, mock_get_model):
        """Test sending multiple audio chunks."""
        mock_get_model.return_value = create_mock_model()

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?token=dev_token") as ws:
                for seq in range(1, 4):
                    audio = generate_silent_audio(duration_ms=200)
                    ws.send_json({"type": "audio", "data": audio, "seq": seq})

                ws.send_json({"type": "end"})

                # Drain responses
                while True:
                    response = ws.receive_json()
                    if response["type"] == "final":
                        break

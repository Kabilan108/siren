import wave
from unittest.mock import AsyncMock, patch

import pytest

from siren.server import ensure_16k_wav, is_16khz_wav, is_parakeet_model


def write_wav(path, sample_rate=16000, channels=1):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate)


def test_is_parakeet_model():
    assert is_parakeet_model("nvidia/parakeet-tdt-0.6b-v2")
    assert is_parakeet_model("nvidia/parakeet-custom")
    assert not is_parakeet_model("distil-large-v3")


def test_is_16khz_wav_true(tmp_path):
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, sample_rate=16000, channels=1)
    assert is_16khz_wav(str(audio_path)) is True


def test_is_16khz_wav_false(tmp_path):
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, sample_rate=8000, channels=1)
    assert is_16khz_wav(str(audio_path)) is False


@pytest.mark.asyncio
async def test_ensure_16k_wav_skips_conversion(tmp_path):
    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path, sample_rate=16000, channels=1)
    with patch("siren.server.convert_to_16k_wav", AsyncMock()) as convert:
        result = await ensure_16k_wav(str(audio_path))
        assert result == str(audio_path)
        convert.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_16k_wav_converts(tmp_path):
    audio_path = tmp_path / "audio.wav"
    converted_path = tmp_path / "converted.wav"
    write_wav(audio_path, sample_rate=8000, channels=1)
    with patch(
        "siren.server.convert_to_16k_wav", AsyncMock(return_value=str(converted_path))
    ) as convert:
        result = await ensure_16k_wav(str(audio_path))
        assert result == str(converted_path)
        convert.assert_awaited_once_with(str(audio_path))

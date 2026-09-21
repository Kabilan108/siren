from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from siren import config
from siren.backends import speech_gate, whisper


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "en", "fr"])
async def test_shared_recipe_preserves_explicit_language(
    monkeypatch: pytest.MonkeyPatch, language: str | None,
) -> None:
    monkeypatch.delenv("SIREN_WHISPER_SPEECH_GATE", raising=False)
    model = MagicMock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language=language or "en", duration=2.0))
    result = await whisper.process_whisper_transcription("original.wav", model, language)
    options = model.transcribe.call_args.kwargs
    assert model.transcribe.call_args.args == ("original.wav",)
    assert options["language"] == (language or "en")
    assert options["beam_size"] == 1
    assert options["temperature"] == 0.0
    assert options["condition_on_previous_text"] is False
    assert options["vad_filter"] is False
    assert options["initial_prompt"] == (None if language == "fr" else config.WHISPER_INITIAL_PROMPT)
    assert result.duration == 2.0


@pytest.mark.asyncio
async def test_rejected_audio_skips_asr_and_preserves_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_WHISPER_SPEECH_GATE", "true")
    monkeypatch.setattr(whisper, "decode_audio", lambda *args, **kwargs: np.zeros(32000, dtype=np.float32))
    model = MagicMock()
    result = await whisper.process_whisper_transcription("quiet.wav", model, word_timestamps=True)
    model.transcribe.assert_not_called()
    assert result.text == "" and result.segments == []
    assert result.duration == 2.0 and result.language == "en"


@pytest.mark.asyncio
async def test_rescue_never_trims_or_amplifies_asr_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_WHISPER_SPEECH_GATE", "1")
    audio = np.array([0.0, 0.0001, -0.0001, 0.0], dtype=np.float32)
    original = audio.copy()
    calls = []

    def detector(samples: np.ndarray, options: object) -> list[dict[str, int]]:
        calls.append(samples.copy())
        return [] if len(calls) == 1 else [{"start": 1, "end": 3}]

    monkeypatch.setattr(speech_gate, "get_speech_timestamps", detector)
    monkeypatch.setattr(whisper, "decode_audio", lambda *args, **kwargs: audio)
    model = MagicMock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=0.00025))
    await whisper.process_whisper_transcription("quiet-speech.wav", model, word_timestamps=True)
    assert len(calls) == 2 and np.max(np.abs(calls[1])) == 0.5
    assert model.transcribe.call_args.args[0] is audio
    np.testing.assert_array_equal(audio, original)
    assert model.transcribe.call_args.kwargs["word_timestamps"] is True


def test_raw_speech_does_not_need_rescue(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MagicMock(return_value=[{"start": 2, "end": 4}])
    monkeypatch.setattr(speech_gate, "get_speech_timestamps", detector)
    assert speech_gate.has_speech(np.ones(10, dtype=np.float32))
    assert detector.call_count == 1


def test_gate_requires_both_passes_to_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MagicMock(return_value=[])
    monkeypatch.setattr(speech_gate, "get_speech_timestamps", detector)
    assert not speech_gate.has_speech(np.ones(10, dtype=np.float32))
    assert detector.call_count == 2


def test_invalid_gate_setting_is_not_silently_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_WHISPER_SPEECH_GATE", "typo")
    with pytest.raises(ValueError, match="SIREN_WHISPER_SPEECH_GATE"):
        config.whisper_speech_gate_enabled()

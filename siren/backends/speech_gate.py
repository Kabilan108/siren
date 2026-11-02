"""Reject only whole inputs with no detected speech; never trim ASR input."""

import wave

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from numpy.typing import NDArray


def has_speech(audio: NDArray[np.float32]) -> bool:
    if audio.size == 0:
        return False
    if not np.isfinite(audio).all():
        raise ValueError("Speech gate requires finite audio samples")
    options = VadOptions(
        threshold=0.3,
        min_speech_duration_ms=0,
        min_silence_duration_ms=500,
        speech_pad_ms=400,
    )
    if get_speech_timestamps(audio, options):
        return True
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return False
    # Detection-only rescue for quiet speech. Do not modify the ASR samples.
    normalized = (audio / peak) * 0.5
    return bool(get_speech_timestamps(normalized, options))


def read_16k_mono(audio_path: str) -> NDArray[np.float32]:
    """Read 16 kHz mono 16-bit PCM directly; PyAV decoding costs ~60 ms per clip."""
    try:
        with wave.open(audio_path, "rb") as wav_file:
            if (
                wav_file.getframerate() == 16000
                and wav_file.getnchannels() == 1
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
            ):
                pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
                return pcm.astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    return decode_audio(audio_path, sampling_rate=16000)


def file_has_speech(audio_path: str) -> bool:
    return has_speech(read_16k_mono(audio_path))

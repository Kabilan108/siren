"""Reject only whole inputs with no detected speech; never trim ASR input."""

import numpy as np
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

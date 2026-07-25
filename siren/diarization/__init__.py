import os

DEFAULT_DIARIZATION_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
DEFAULT_MEMORY_FRACTION = 0.35
DEFAULT_TIMEOUT_SECONDS = 900.0


def get_diarization_model() -> str:
    return os.environ.get("SIREN_DIARIZE_MODEL", DEFAULT_DIARIZATION_MODEL)


def get_memory_fraction() -> float:
    return float(
        os.environ.get(
            "SIREN_DIARIZE_MEMORY_FRACTION",
            str(DEFAULT_MEMORY_FRACTION),
        )
    )


def get_timeout_seconds() -> float:
    return float(
        os.environ.get(
            "SIREN_DIARIZE_TIMEOUT",
            str(DEFAULT_TIMEOUT_SECONDS),
        )
    )

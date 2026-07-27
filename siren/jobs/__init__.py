import os
from pathlib import Path

DEFAULT_CHUNK_SECONDS = 300.0
DEFAULT_MAX_CONCURRENT_JOBS = 1
DEFAULT_MAX_QUEUED_JOBS = 8
DEFAULT_MAX_UPLOAD_BYTES = 2_000_000_000
DEFAULT_MEMORY_FRACTION = 0.6
DEFAULT_RETENTION_DAYS = 7.0
DEFAULT_TIMEOUT_SECONDS = 7200.0


def get_chunk_seconds() -> float:
    return float(
        os.environ.get(
            "SIREN_JOB_CHUNK_SECONDS",
            str(DEFAULT_CHUNK_SECONDS),
        )
    )


def get_max_concurrent_jobs() -> int:
    value = int(
        os.environ.get(
            "SIREN_MAX_CONCURRENT_JOBS",
            str(DEFAULT_MAX_CONCURRENT_JOBS),
        )
    )
    if value < 1:
        raise ValueError("SIREN_MAX_CONCURRENT_JOBS must be at least 1")
    return value


def get_max_queued_jobs() -> int:
    value = int(
        os.environ.get(
            "SIREN_MAX_QUEUED_JOBS",
            str(DEFAULT_MAX_QUEUED_JOBS),
        )
    )
    if value < 1:
        raise ValueError("SIREN_MAX_QUEUED_JOBS must be at least 1")
    return value


def get_max_upload_bytes() -> int:
    value = int(
        os.environ.get(
            "SIREN_JOB_MAX_UPLOAD_BYTES",
            str(DEFAULT_MAX_UPLOAD_BYTES),
        )
    )
    if value < 1:
        raise ValueError("SIREN_JOB_MAX_UPLOAD_BYTES must be at least 1")
    return value


def get_memory_fraction() -> float:
    value = float(
        os.environ.get(
            "SIREN_JOB_MEMORY_FRACTION",
            str(DEFAULT_MEMORY_FRACTION),
        )
    )
    if not 0.0 < value <= 1.0:
        raise ValueError("SIREN_JOB_MEMORY_FRACTION must be greater than 0 and at most 1")
    return value


def get_retention_days() -> float:
    value = float(
        os.environ.get(
            "SIREN_SPOOL_RETENTION_DAYS",
            str(DEFAULT_RETENTION_DAYS),
        )
    )
    if value < 0.0:
        raise ValueError("SIREN_SPOOL_RETENTION_DAYS must not be negative")
    return value


def get_spool_dir() -> Path:
    return Path(
        os.environ.get(
            "SIREN_SPOOL_DIR",
            "~/.local/state/siren/jobs",
        )
    ).expanduser()


def get_timeout_seconds() -> float:
    value = float(
        os.environ.get(
            "SIREN_JOB_TIMEOUT",
            str(DEFAULT_TIMEOUT_SECONDS),
        )
    )
    if value <= 0.0:
        raise ValueError("SIREN_JOB_TIMEOUT must be greater than 0")
    return value

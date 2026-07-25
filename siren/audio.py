import asyncio
import logging
import tempfile
import wave
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from siren.config import UPLOAD_CHUNK_BYTES
from siren.logging_utils import log_event


def get_wav_info(
    audio_path: str,
    *,
    request_id: str | None = None,
) -> dict[str, float | int]:
    try:
        with wave.open(audio_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = frames / rate if rate else 0.0
            return {
                "audio_frames": frames,
                "audio_sample_rate": rate,
                "audio_channels": channels,
                "audio_duration_sec": duration,
            }
    except (wave.Error, EOFError) as exc:
        fields: dict[str, object] = {
            "audio_path": audio_path,
            "error": str(exc),
        }
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(logging.WARNING, "audio_info_failed", **fields)
        return {}


async def save_upload_file(
    upload_file: UploadFile,
    *,
    request_id: str | None = None,
) -> str:
    temp_path: str | None = None
    try:
        suffix = Path(upload_file.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while content := await upload_file.read(UPLOAD_CHUNK_BYTES):
                temp_file.write(content)
            return temp_path
    except Exception as exc:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        fields: dict[str, object] = {"error": str(exc)}
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(logging.ERROR, "upload_save_failed", **fields)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio file",
        )


def is_16khz_wav(audio_path: str) -> bool:
    try:
        with wave.open(audio_path, "rb") as wav_file:
            return wav_file.getframerate() == 16000 and wav_file.getnchannels() == 1
    except (wave.Error, EOFError):
        return False


async def convert_to_16k_wav(
    audio_path: str,
    *,
    request_id: str | None = None,
) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        output_path = temp_file.name

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()
    if process.returncode != 0:
        fields: dict[str, object] = {
            "audio_path": audio_path,
            "error": stderr.decode(errors="ignore"),
        }
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(logging.ERROR, "audio_conversion_failed", **fields)
        Path(output_path).unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to convert audio file",
        )
    return output_path


async def ensure_16k_wav(
    audio_path: str,
    *,
    request_id: str | None = None,
) -> str:
    if is_16khz_wav(audio_path):
        return audio_path
    if request_id is None:
        return await convert_to_16k_wav(audio_path)
    return await convert_to_16k_wav(audio_path, request_id=request_id)

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from siren.api.auth import verify_token
from siren.audio import ensure_16k_wav, get_wav_info, save_upload_file
from siren.diarization import get_diarization_model
from siren.diarization.supervisor import run_diarization
from siren.logging_utils import log_event
from siren.schemas import DiarizationResponse

router = APIRouter()
_DISCONNECT_POLL_SECONDS = 2.0


async def _run_until_disconnected(
    request: Request,
    audio_path: str,
    *,
    request_id: str,
) -> dict[str, object] | None:
    diarization_task = asyncio.create_task(
        run_diarization(audio_path, request_id=request_id)
    )
    try:
        while True:
            done, _pending = await asyncio.wait(
                {diarization_task},
                timeout=_DISCONNECT_POLL_SECONDS,
            )
            if diarization_task in done:
                return diarization_task.result()
            if await request.is_disconnected():
                diarization_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await diarization_task
                return None
    finally:
        if not diarization_task.done():
            diarization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await diarization_task


@router.post(
    "/v1/audio/diarize",
    response_model=DiarizationResponse,
    dependencies=[Depends(verify_token)],
)
async def diarize_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(
        None,
        description="ID of the configured diarization model.",
    ),
) -> DiarizationResponse | Response:
    original_path: str | None = None
    converted_path: str | None = None
    request_id = uuid.uuid4().hex
    request_start = time.perf_counter()
    try:
        configured_model = get_diarization_model()
        if model is not None and model != configured_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Invalid model: '{model}'. This endpoint supports only "
                    f"'{configured_model}'."
                ),
            )

        original_path = await save_upload_file(file, request_id=request_id)
        audio_path = await ensure_16k_wav(original_path, request_id=request_id)
        if audio_path != original_path:
            converted_path = audio_path
        audio_info = get_wav_info(audio_path, request_id=request_id)
        log_event(
            logging.INFO,
            "diarize_request",
            request_id=request_id,
            filename=file.filename,
            model=configured_model,
            audio_bytes=Path(original_path).stat().st_size,
            **audio_info,
        )

        if await request.is_disconnected():
            log_event(
                logging.INFO,
                "diarize_disconnected",
                request_id=request_id,
                phase="before_dispatch",
            )
            return Response(status_code=499)

        result = await _run_until_disconnected(
            request,
            audio_path,
            request_id=request_id,
        )
        if result is None:
            log_event(
                logging.INFO,
                "diarize_disconnected",
                request_id=request_id,
                phase="worker_running",
            )
            return Response(status_code=499)
        return DiarizationResponse.model_validate(result)
    except HTTPException as exc:
        log_event(
            logging.ERROR,
            "diarize_error",
            request_id=request_id,
            status_code=exc.status_code,
            latency_ms=int((time.perf_counter() - request_start) * 1000),
            error=str(exc.detail),
        )
        raise
    except Exception as exc:
        log_event(
            logging.ERROR,
            "diarize_error",
            request_id=request_id,
            latency_ms=int((time.perf_counter() - request_start) * 1000),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diarization failed",
        )
    finally:
        await file.close()
        for path in {converted_path, original_path}:
            if path:
                Path(path).unlink(missing_ok=True)

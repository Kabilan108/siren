from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from siren import models
from siren.api.auth import verify_token
from siren.jobs.errors import sanitize_job_error
from siren.jobs.runner import (
    JobQueueFullError,
    JobUploadTooLargeError,
    job_runner,
)
from siren.schemas import (
    TranscriptJobAccepted,
    TranscriptJobResult,
    TranscriptJobStatus,
)

router = APIRouter()


@router.post(
    "/v1/jobs/transcripts",
    response_model=TranscriptJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_token)],
)
async def create_transcript_job(
    file: UploadFile = File(...),
    model: str | None = Form(
        None,
        description="ID of the ASR model to use.",
    ),
    language: str | None = Form(
        None,
        description="Optional ISO-639-1 input language.",
    ),
) -> TranscriptJobAccepted:
    try:
        target_model = models.resolve_transcription_model_name(model)
        job_id, position = await job_runner.enqueue(
            file,
            model=target_model,
            language=language,
        )
        return TranscriptJobAccepted(
            id=job_id,
            status="queued",
            position=position,
        )
    except JobUploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()


@router.get(
    "/v1/jobs/transcripts/{job_id}",
    response_model=TranscriptJobStatus,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_token)],
)
async def get_transcript_job(
    job_id: str,
) -> TranscriptJobStatus | JSONResponse:
    state = await job_runner.status(job_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    response = TranscriptJobStatus.model_validate(state)
    payload = response.model_dump(exclude_none=True)
    payload["phase"] = response.phase
    if "error" in payload:
        payload["error"] = sanitize_job_error(str(payload["error"]))
    return JSONResponse(content=payload)


@router.get(
    "/v1/jobs/transcripts/{job_id}/result",
    response_model=TranscriptJobResult,
    dependencies=[Depends(verify_token)],
)
async def get_transcript_job_result(job_id: str) -> TranscriptJobResult:
    state = await job_runner.status(job_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    if state.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job is not completed",
        )
    result = job_runner.result(job_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job result is unavailable",
        )
    return TranscriptJobResult.model_validate(result)

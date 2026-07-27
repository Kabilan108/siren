import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.types import Message

from siren import config, models
from siren.api import (
    align_router,
    diarize_router,
    health_router,
    jobs_router,
    models_router,
    transcriptions_router,
)
from siren.diarization.supervisor import terminate_active_worker
from siren.jobs import get_max_upload_bytes
from siren.jobs.runner import job_runner
from siren.logging_utils import log_event


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await job_runner.start()
    except Exception as exc:
        log_event(logging.ERROR, "job_runner_start_failed", error=str(exc))
    try:
        models.current_model_name = config.load_model_name()
        await models.ensure_model_loaded(models.current_model_name)
        yield
    finally:
        log_event(logging.INFO, "shutdown_started")
        await job_runner.stop()
        await terminate_active_worker()
        models.unload_model()
        log_event(logging.INFO, "shutdown_complete")


_BODY_LIMITS = (
    # buffer_when_unsized is False where the handler already caps the body as
    # it streams; buffering a multi-gigabyte upload here would defeat that.
    ("/v1/jobs/transcripts", lambda: get_max_upload_bytes() + 1024 * 1024, False),
    ("/v1/audio/align", lambda: 64 * 1024 * 1024, True),
)


app = FastAPI(
    title="siren",
    description="API for transcribing audio using Whisper and Parakeet models, compatible with OpenAI schema",
    version=config.SIREN_VERSION,
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(models_router)
app.include_router(transcriptions_router)
app.include_router(jobs_router)
app.include_router(diarize_router)
app.include_router(align_router)


@app.middleware("http")
async def enforce_body_limits(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.method == "POST":
        for prefix, limit, buffer_when_unsized in _BODY_LIMITS:
            if not request.url.path.startswith(prefix):
                continue
            max_bytes = limit()
            too_large = JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body exceeds the maximum of {max_bytes} bytes."
                    )
                },
            )
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit():
                if int(content_length) > max_bytes:
                    return too_large
            elif buffer_when_unsized:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        return too_large
                request._body = bytes(body)

                async def receive() -> Message:
                    return {
                        "type": "http.request",
                        "body": request._body,
                        "more_body": False,
                    }

                request._receive = receive
            break
    return await call_next(request)

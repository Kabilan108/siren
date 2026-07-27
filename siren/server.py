import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

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
    ("/v1/jobs/transcripts", lambda: get_max_upload_bytes() + 1024 * 1024),
    ("/v1/audio/align", lambda: 64 * 1024 * 1024),
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
        for prefix, limit in _BODY_LIMITS:
            if request.url.path.startswith(prefix):
                content_length = request.headers.get("content-length")
                if content_length and content_length.isdigit():
                    if int(content_length) > limit():
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": (
                                    "Request body exceeds the maximum of "
                                    f"{limit()} bytes."
                                )
                            },
                        )
                break
    return await call_next(request)

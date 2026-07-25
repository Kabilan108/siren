import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from siren import config, models
from siren.api import (
    diarize_router,
    health_router,
    models_router,
    transcriptions_router,
)
from siren.diarization.supervisor import terminate_active_worker
from siren.logging_utils import log_event


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    models.current_model_name = config.load_model_name()
    await models.ensure_model_loaded(models.current_model_name)
    try:
        yield
    finally:
        log_event(logging.INFO, "shutdown_started")
        await terminate_active_worker()
        models.unload_model()
        log_event(logging.INFO, "shutdown_complete")


app = FastAPI(
    title="siren",
    description="API for transcribing audio using Whisper and Parakeet models, compatible with OpenAI schema",
    version=config.SIREN_VERSION,
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(models_router)
app.include_router(transcriptions_router)
app.include_router(diarize_router)

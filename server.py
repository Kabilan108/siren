import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from faster_whisper import WhisperModel
from pydantic import BaseModel

logger = logging.getLogger("uvicorn")


class TranscriptionResponse(BaseModel):
    text: str


class ModelInfo(BaseModel):
    id: str
    name: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


# Authentication
API_KEY = os.environ.get("API_KEY", "dev_key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return api_key


# Application state and lifespan
whisper_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    logger.info("Loading Whisper model...")
    model_size = "distil-large-v3"
    whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    logger.info("Whisper model loaded successfully")
    yield
    logger.info("Shutting down and cleaning up resources...")
    del whisper_model
    logger.info("Cleanup complete")


# Helper functions
@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    if whisper_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription model not loaded",
        )
    return whisper_model


def get_available_whisper_models() -> list[ModelInfo]:
    """Get available Whisper models."""
    # Faster Whisper supports these models
    models = [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-medium.en",
        "distil-small.en",
        "distil-large-v3",
        "large-v3-turbo",
        "turbo",
    ]
    return [
        ModelInfo(id=f"faster-whisper/{model}", name=f"Faster Whisper: {model}")
        for model in models
    ]


async def save_upload_file(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await upload_file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio file",
        )


async def process_transcription(
    file_path: str,
    language: str | None = None,
) -> tuple[Generator, dict]:
    model = get_model()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: model.transcribe(
            file_path, beam_size=5, language=language, condition_on_previous_text=False
        ),
    )


app = FastAPI(
    title="Speech-to-Text API Server",
    description="API for transcribing audio using Whisper, compatible with OpenAI schema",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_models():
    """List available models in OpenAI-compatible format"""
    try:
        models = get_available_whisper_models()
        return ModelsResponse(data=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@app.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = None,
):
    """OpenAI compatible transcription endpoint"""
    file_path = await save_upload_file(file)
    try:
        logger.info(f"Starting transcription for file: {file.filename}")
        segments, info = await process_transcription(file_path, language)
        # Combine all segment texts into a single string, like OpenAI does
        full_text = " ".join(segment.text.strip() for segment in segments)
        logger.info(
            f"Transcription completed. Text length: {len(full_text)} characters"
        )
        return TranscriptionResponse(text=full_text)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

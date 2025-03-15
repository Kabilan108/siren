import asyncio
import io
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
from pydantic import BaseModel

logger = logging.getLogger("uvicorn")

TOKEN = os.environ.get("SIREN_API_KEY", "dev_token")
CONFIG_FILE = Path("~/config.json").expanduser()
DEFAULT_MODEL = "distil-small.en"

token_header = HTTPBearer(auto_error=True)

current_model: WhisperModel | None = None
current_model_name: str | None = None


class TranscriptionResponse(BaseModel):
    text: str


class ModelInfo(BaseModel):
    id: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


def get_whisper_params():
    """Get parameters for WhisperModel"""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"
    return {
        "device": device,
        "compute_type": compute_type,
    }


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(token_header),
) -> str:
    """Verify the Bearer token"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Bearer token required.",
            headers={"WW-Authenticate": "Bearer"},
        )

    if credentials.credentials != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WW-Authenticate": "Bearer"},
        )

    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_model, current_model_name
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                current_model_name = config.get("model", DEFAULT_MODEL)
        else:
            current_model_name = DEFAULT_MODEL
    except Exception as e:
        logger.warn(f"Failed to load config, using default model: {e}")
        current_model_name = DEFAULT_MODEL

    logger.info(f"Loading Whisper model: {current_model_name}")
    current_model = WhisperModel(current_model_name, **get_whisper_params())
    logger.info("Whisper model loaded successfully")
    yield
    logger.info("Shutting down and cleaning up resources...")
    del current_model
    torch.cuda.empty_cache()
    logger.info("Cleanup complete")


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
    return [ModelInfo(id=m) for m in models]


def save_config(model_name: str):
    """Save the current model name to config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"model": model_name}, f)
    except Exception as e:
        logger.warn(f"Failed to save config: {e}")


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
    audio_data: bytes,
    model: WhisperModel,
    language: str | None = None,
) -> tuple[Generator, dict]:
    """Process transcription directly from audio bytes"""
    audio_buffer = io.BytesIO(audio_data)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: model.transcribe(audio_buffer, language=language)
    )


async def get_whisper_model(model: str | None = None) -> WhisperModel:
    global current_model, current_model_name
    available_models = {m.id for m in get_available_whisper_models()}

    target_model = model if model is not None else current_model_name

    if target_model not in available_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model: '{target_model}'. Use /v1/models to see available models.",
        )

    if target_model == current_model_name and current_model is not None:
        return current_model

    logger.info(f"Switching model from '{current_model_name}' to '{target_model}'")
    try:
        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()

        current_model = WhisperModel(target_model, **get_whisper_params())
        current_model_name = target_model
        save_config(target_model)
        logger.info(f"Whisper model '{current_model_name}' loaded successfully")
        return current_model

    except Exception as e:
        logger.error(f"Failed to load model '{target_model}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{target_model}': {str(e)}",
        )


app = FastAPI(
    title="siren",
    description="API for transcribing audio using Whisper, compatible with OpenAI schema",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_token)],
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
    dependencies=[Depends(verify_token)],
)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = None,
    model: str | None = None,
    whisper_model: WhisperModel = Depends(get_whisper_model),
):
    """OpenAI compatible transcription endpoint"""
    try:
        # Read audio data directly into memory
        audio_data = await file.read()
        logger.info(
            f"Starting transcription for file: {file.filename}, size: {len(audio_data)} bytes"
        )

        # Process transcription directly with audio bytes
        segments, info = await process_transcription(
            audio_data, whisper_model, language
        )
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
        # No file cleanup needed anymore since we're not saving to disk
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

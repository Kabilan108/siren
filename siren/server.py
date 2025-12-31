import asyncio
import json
import logging
import os
import tempfile
import warnings
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
from pydantic import BaseModel

warnings.filterwarnings("ignore", module="nemo")
warnings.filterwarnings("ignore", message=".*torchaudio.*")

import nemo.collections.asr as nemo_asr

# Suppress NeMo warnings for cleaner output

logger = logging.getLogger("uvicorn")

TOKEN = os.environ.get("SIREN_API_KEY", "dev_token")
CONFIG_FILE = Path("~/config.json").expanduser()
DEFAULT_MODEL = "distil-small.en"

PARAKEET_MODELS = [
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-1.1b",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-ctc-0.6b",
]

token_header = HTTPBearer(auto_error=True)

current_model: WhisperModel | nemo_asr.models.ASRModel | None = None
current_model_name: str | None = None
model_lock = asyncio.Lock()
model_ready = asyncio.Event()
model_loading_task: asyncio.Task | None = None
model_loading_target: str | None = None
model_load_error: Exception | None = None


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
        logger.warning(f"Failed to load config, using default model: {e}")
        current_model_name = DEFAULT_MODEL

    logger.info(f"Loading model: {current_model_name}")
    await ensure_model_loaded(current_model_name)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down and cleaning up resources...")
    del current_model
    torch.cuda.empty_cache()
    logger.info("Cleanup complete")


def get_available_models() -> list[ModelInfo]:
    """Get available transcription models (Whisper + Parakeet)."""
    # Faster Whisper models
    whisper_models = [
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
    # Combine with Parakeet models
    all_models = whisper_models + PARAKEET_MODELS
    return [ModelInfo(id=m) for m in all_models]


def is_parakeet_model(model_name: str) -> bool:
    """Check if a model name is a Parakeet model."""
    return model_name in PARAKEET_MODELS or model_name.startswith("nvidia/parakeet")


def save_config(model_name: str):
    """Save the current model name to config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"model": model_name}, f)
    except Exception as e:
        logger.warning(f"Failed to save config: {e}")


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


def is_16khz_wav(audio_path: str) -> bool:
    """Check if file is a 16 kHz mono WAV file."""
    try:
        with wave.open(audio_path, "rb") as wav_file:
            return wav_file.getframerate() == 16000 and wav_file.getnchannels() == 1
    except (wave.Error, EOFError):
        return False


async def convert_to_16k_wav(audio_path: str) -> str:
    """Convert audio file to 16 kHz mono WAV using ffmpeg."""
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
        logger.error(f"ffmpeg conversion failed: {stderr.decode(errors='ignore')}")
        os.unlink(output_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to convert audio file",
        )
    return output_path


async def ensure_16k_wav(audio_path: str) -> str:
    """Return a 16 kHz mono WAV file path, converting only when needed."""
    if is_16khz_wav(audio_path):
        return audio_path
    return await convert_to_16k_wav(audio_path)


async def process_whisper_transcription(
    audio_path: str,
    model: WhisperModel,
    language: str | None = None,
) -> str:
    """Process transcription with Whisper model."""
    loop = asyncio.get_running_loop()
    segments, _info = await loop.run_in_executor(
        None, lambda: model.transcribe(audio_path, language=language)
    )
    return " ".join(segment.text.strip() for segment in segments)


async def process_parakeet_transcription(
    audio_path: str,
    model: nemo_asr.models.ASRModel,
) -> str:
    """Process transcription with Parakeet model."""
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, lambda: model.transcribe([audio_path]))
    if not output:
        return ""
    return output[0].text


def load_model(model_name: str) -> WhisperModel | nemo_asr.models.ASRModel:
    """Load a transcription model (Whisper or Parakeet)."""
    if is_parakeet_model(model_name):
        logger.info(f"Loading Parakeet model: {model_name}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    else:
        logger.info(f"Loading Whisper model: {model_name}")
        return WhisperModel(model_name, **get_whisper_params())


async def _load_model_in_background(target_model: str) -> None:
    """Load model in a background thread and update globals when ready."""
    global current_model, current_model_name, model_load_error
    try:
        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()

        model = await asyncio.to_thread(load_model, target_model)
        current_model = model
        current_model_name = target_model
        save_config(target_model)
        logger.info(f"Model '{current_model_name}' loaded successfully")
    except Exception as e:
        model_load_error = e
        logger.error(f"Failed to load model '{target_model}': {e}")
    finally:
        model_ready.set()


async def ensure_model_loaded(
    target_model: str,
) -> WhisperModel | nemo_asr.models.ASRModel:
    """Ensure target model is loaded without blocking the event loop."""
    global model_loading_task, model_loading_target, model_load_error

    async with model_lock:
        if (
            target_model == current_model_name
            and current_model is not None
            and model_load_error is None
        ):
            return current_model

        if (
            model_loading_task is not None
            and not model_loading_task.done()
            and model_loading_target != target_model
        ):
            await model_loading_task

        if (
            model_loading_task is None
            or model_loading_task.done()
            or model_loading_target != target_model
        ):
            model_ready.clear()
            model_load_error = None
            model_loading_target = target_model
            model_loading_task = asyncio.create_task(
                _load_model_in_background(target_model)
            )

        task = model_loading_task

    if task is not None:
        await task

    if model_load_error is not None:
        raise model_load_error

    if current_model is None:
        raise RuntimeError(f"Model '{target_model}' failed to load")

    return current_model


async def get_transcription_model(
    model: str | None = None,
) -> WhisperModel | nemo_asr.models.ASRModel:
    global current_model, current_model_name
    available_models = {m.id for m in get_available_models()}

    target_model = model if model is not None else current_model_name

    if target_model not in available_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model: '{target_model}'. Use /v1/models to see available models.",
        )

    if (
        target_model == current_model_name
        and current_model is not None
        and model_load_error is None
    ):
        return current_model

    logger.info(f"Switching model from '{current_model_name}' to '{target_model}'")
    try:
        model_instance = await ensure_model_loaded(target_model)
        return model_instance

    except Exception as e:
        logger.error(f"Failed to load model '{target_model}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{target_model}': {str(e)}",
        )


app = FastAPI(
    title="siren",
    description="API for transcribing audio using Whisper and Parakeet models, compatible with OpenAI schema",
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
        models = get_available_models()
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
    model: str | None = Form(
        None,
        description="ID of the model to use. Supports Whisper models and Parakeet models (e.g., nvidia/parakeet-tdt-0.6b-v2).",
    ),
    language: str | None = Form(
        None,
        description="The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency. Note: Parakeet models only support English.",
    ),
):
    """OpenAI compatible transcription endpoint"""
    original_path = None
    converted_path = None
    try:
        transcription_model = await get_transcription_model(model)
        target_model = model if model is not None else current_model_name

        original_path = await save_upload_file(file)
        audio_path = await ensure_16k_wav(original_path)
        if audio_path != original_path:
            converted_path = audio_path
        audio_size = os.path.getsize(original_path)
        logger.info(
            f"Starting transcription for file: {file.filename}, size: {audio_size} bytes, model: {target_model}, language: {language}"
        )

        # Process transcription based on model type
        if is_parakeet_model(target_model):
            full_text = await process_parakeet_transcription(
                audio_path, transcription_model
            )
        else:
            full_text = await process_whisper_transcription(
                audio_path, transcription_model, language
            )

        logger.info(
            f"Transcription completed. Text length: {len(full_text)} characters"
        )
        return TranscriptionResponse(text=full_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )
    finally:
        await file.close()
        for path in {converted_path, original_path}:
            if path and os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

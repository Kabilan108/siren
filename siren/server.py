import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
import warnings
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
from pydantic import BaseModel
from pydub import AudioSegment, silence

warnings.filterwarnings("ignore", module="nemo")
warnings.filterwarnings("ignore", message=".*torchaudio.*")

import nemo.collections.asr as nemo_asr

# Suppress NeMo warnings for cleaner output

logger = logging.getLogger("uvicorn")

TOKEN = os.environ.get("SIREN_API_KEY", "dev_token")
CONFIG_FILE = Path("~/config.json").expanduser()
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"

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


def log_event(level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str))


def get_wav_info(audio_path: str) -> dict[str, float | int]:
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
        log_event(logging.WARNING, "audio_info_failed", audio_path=audio_path, error=str(exc))
        return {}


def get_cuda_stats() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
    }


@dataclass
class StreamingConfig:
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    min_silence_ms: int = 900
    trailing_silence_ms: int = 200
    silence_thresh_db: int = -40
    max_segment_sec: int = 90
    cap_grace_ms: int = 1500
    overlap_ms: int = 400
    use_timestamps_on_cap: bool = True


@dataclass
class StreamingState:
    buffer: bytearray
    cap_deadline: float | None


def bytes_per_ms(config: StreamingConfig) -> int:
    return int(config.sample_rate * config.sample_width * config.channels / 1000)


def buffer_duration_ms(buffer: bytearray, config: StreamingConfig) -> float:
    b_per_ms = bytes_per_ms(config)
    if b_per_ms == 0:
        return 0.0
    return len(buffer) / b_per_ms


def detect_trailing_silence_cut(
    buffer: bytearray, config: StreamingConfig
) -> int | None:
    b_per_ms = bytes_per_ms(config)
    if b_per_ms == 0:
        return None
    if len(buffer) < config.min_silence_ms * b_per_ms:
        return None
    total_ms = int(len(buffer) / b_per_ms)
    window_ms = config.min_silence_ms + config.trailing_silence_ms
    if total_ms > window_ms:
        window_bytes = int(window_ms * b_per_ms)
        window_bytes = min(window_bytes, len(buffer))
        segment_bytes = bytes(buffer[-window_bytes:])
        window_offset_ms = total_ms - int(window_bytes / b_per_ms)
    else:
        segment_bytes = bytes(buffer)
        window_offset_ms = 0
    segment = AudioSegment(
        data=segment_bytes,
        sample_width=config.sample_width,
        frame_rate=config.sample_rate,
        channels=config.channels,
    )
    silence_ranges = silence.detect_silence(
        segment,
        min_silence_len=config.min_silence_ms,
        silence_thresh=config.silence_thresh_db,
    )
    if not silence_ranges:
        return None
    end_ms = len(segment)
    last_start, last_end = silence_ranges[-1]
    if end_ms - last_end <= config.trailing_silence_ms:
        return window_offset_ms + last_end
    return None


def buffer_has_nonsilent_audio(
    buffer: bytearray, config: StreamingConfig
) -> bool:
    if not buffer:
        return False
    segment = AudioSegment(
        data=bytes(buffer),
        sample_width=config.sample_width,
        frame_rate=config.sample_rate,
        channels=config.channels,
    )
    nonsilent = silence.detect_nonsilent(
        segment,
        min_silence_len=config.trailing_silence_ms,
        silence_thresh=config.silence_thresh_db,
    )
    return bool(nonsilent)


def segment_has_nonsilent_audio(
    segment_bytes: bytes, config: StreamingConfig
) -> bool:
    if not segment_bytes:
        return False
    segment = AudioSegment(
        data=segment_bytes,
        sample_width=config.sample_width,
        frame_rate=config.sample_rate,
        channels=config.channels,
    )
    nonsilent = silence.detect_nonsilent(
        segment,
        min_silence_len=config.trailing_silence_ms,
        silence_thresh=config.silence_thresh_db,
    )
    return bool(nonsilent)


def extract_word_timestamps(entry: object) -> list[dict] | None:
    candidates = [
        "timestamps",
        "word_timestamps",
        "words",
        "word_ts",
        "word_timestamp",
    ]
    for name in candidates:
        value = getattr(entry, name, None)
        if not value:
            continue
        if isinstance(value, dict):
            for key in ("words", "word", "word_timestamps"):
                inner = value.get(key)
                if isinstance(inner, list):
                    return inner
        if isinstance(value, list):
            return value
    return None


def normalize_word_timestamps(
    words: list[dict], duration_ms: float
) -> list[dict]:
    if not words:
        return []
    max_end = 0.0
    for w in words:
        end = w.get("end") if isinstance(w, dict) else None
        if isinstance(end, (int, float)) and end > max_end:
            max_end = float(end)
    if max_end <= 0:
        return []
    if max_end <= duration_ms / 1000.0 + 1.0:
        scale = 1000.0
    else:
        scale = 1.0
    normalized: list[dict] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        start = w.get("start")
        end = w.get("end")
        word = w.get("word") or w.get("text") or ""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            normalized.append(
                {
                    "start_ms": float(start) * scale,
                    "end_ms": float(end) * scale,
                    "word": str(word),
                }
            )
    return normalized


def trim_text_by_timestamp(
    words: list[dict], cut_ms: int, fallback_text: str
) -> str:
    if not words:
        return fallback_text
    kept = [w["word"] for w in words if w["end_ms"] <= cut_ms and w["word"]]
    if not kept:
        return fallback_text
    return " ".join(kept).strip()


def write_pcm_to_wav(
    pcm_bytes: bytes, config: StreamingConfig
) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        output_path = temp_file.name
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(config.sample_width)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(pcm_bytes)
    return output_path


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


def verify_websocket_token(websocket: WebSocket) -> None:
    auth_header = websocket.headers.get("Authorization", "")
    token = None
    source = "none"
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        source = "header"
    if token is None:
        token = websocket.query_params.get("token")
        source = "query"
    if token != TOKEN:
        def mask_token(value: str | None) -> str:
            if not value:
                return "<empty>"
            if len(value) <= 8:
                return f"{value}(len={len(value)})"
            return f"{value[:4]}...{value[-4:]}(len={len(value)})"

        logger.warning(
            "websocket auth failed",
            extra={
                "source": source,
                "token": mask_token(token),
                "expected": mask_token(TOKEN),
            },
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


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
    request_id: str | None = None,
) -> str:
    """Process transcription with Parakeet model."""
    loop = asyncio.get_running_loop()
    audio_info = get_wav_info(audio_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = await loop.run_in_executor(None, lambda: model.transcribe([audio_path]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    log_event(
        logging.INFO,
        "parakeet_transcribe",
        request_id=request_id,
        latency_ms=elapsed_ms,
        **audio_info,
        **get_cuda_stats(),
    )
    if not output:
        return ""
    return output[0].text


async def transcribe_parakeet_segment(
    audio_path: str,
    model: nemo_asr.models.ASRModel,
    request_id: str | None,
    use_timestamps: bool,
) -> tuple[str, list[dict] | None]:
    loop = asyncio.get_running_loop()
    audio_info = get_wav_info(audio_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    def _run_transcribe():
        if use_timestamps:
            try:
                return model.transcribe([audio_path], timestamps=True)
            except TypeError:
                return model.transcribe([audio_path])
        return model.transcribe([audio_path])

    output = await loop.run_in_executor(None, _run_transcribe)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    log_event(
        logging.INFO,
        "parakeet_stream_transcribe",
        request_id=request_id,
        latency_ms=elapsed_ms,
        **audio_info,
        **get_cuda_stats(),
        timestamps=use_timestamps,
    )
    if not output:
        return "", None
    entry = output[0]
    text = getattr(entry, "text", str(entry))
    words = extract_word_timestamps(entry) if use_timestamps else None
    if words:
        duration_ms = audio_info.get("audio_duration_sec", 0) * 1000
        words = normalize_word_timestamps(words, duration_ms)
    return text, words


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
    request_id = uuid.uuid4().hex
    request_start = time.perf_counter()
    try:
        transcription_model = await get_transcription_model(model)
        target_model = model if model is not None else current_model_name

        original_path = await save_upload_file(file)
        audio_path = await ensure_16k_wav(original_path)
        if audio_path != original_path:
            converted_path = audio_path
        audio_size = os.path.getsize(original_path)
        audio_info = get_wav_info(audio_path)
        log_event(
            logging.INFO,
            "transcribe_request",
            request_id=request_id,
            model=target_model,
            language=language,
            filename=file.filename,
            audio_bytes=audio_size,
            **audio_info,
        )
        logger.info(
            f"Starting transcription for file: {file.filename}, size: {audio_size} bytes, model: {target_model}, language: {language}"
        )

        # Process transcription based on model type
        if is_parakeet_model(target_model):
            full_text = await process_parakeet_transcription(
                audio_path, transcription_model, request_id=request_id
            )
        else:
            full_text = await process_whisper_transcription(
                audio_path, transcription_model, language
            )

        total_ms = int((time.perf_counter() - request_start) * 1000)
        log_event(
            logging.INFO,
            "transcribe_complete",
            request_id=request_id,
            model=target_model,
            latency_ms=total_ms,
            text_length=len(full_text),
        )
        logger.info(
            f"Transcription completed. Text length: {len(full_text)} characters"
        )
        return TranscriptionResponse(text=full_text)
    except HTTPException:
        raise
    except Exception as e:
        log_event(
            logging.ERROR,
            "transcribe_error",
            request_id=request_id,
            error=str(e),
        )
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


def apply_streaming_overrides(config: StreamingConfig, payload: dict) -> None:
    for key in (
        "sample_rate",
        "channels",
        "sample_width",
        "min_silence_ms",
        "trailing_silence_ms",
        "silence_thresh_db",
        "max_segment_sec",
        "cap_grace_ms",
        "overlap_ms",
        "use_timestamps_on_cap",
    ):
        if key in payload:
            setattr(config, key, payload[key])


@app.websocket("/v1/audio/stream")
async def stream_transcriptions(websocket: WebSocket):
    try:
        verify_websocket_token(websocket)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    config = StreamingConfig()
    state = StreamingState(buffer=bytearray(), cap_deadline=None)
    request_id = uuid.uuid4().hex

    for key in (
        "sample_rate",
        "channels",
        "sample_width",
        "min_silence_ms",
        "trailing_silence_ms",
        "silence_thresh_db",
        "max_segment_sec",
        "cap_grace_ms",
        "overlap_ms",
    ):
        if key in websocket.query_params:
            try:
                value = int(websocket.query_params[key])
                setattr(config, key, value)
            except ValueError:
                continue

    model_param = websocket.query_params.get("model")
    model = await get_transcription_model(model_param)
    target_model = model_param if model_param is not None else current_model_name
    if not is_parakeet_model(target_model or ""):
        await websocket.send_text(
            json.dumps(
                {"type": "error", "message": "streaming only supports Parakeet models"}
            )
        )
        await websocket.close(code=1003)
        return

    async def emit_segment(
        segment_bytes: bytes, cut_ms: int, forced: bool
    ) -> bool:
        audio_path = write_pcm_to_wav(segment_bytes, config)
        try:
            use_timestamps = forced and config.use_timestamps_on_cap
            text, words = await transcribe_parakeet_segment(
                audio_path, model, request_id=request_id, use_timestamps=use_timestamps
            )
            if forced and words:
                text = trim_text_by_timestamp(words, cut_ms, text)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "segment",
                        "text": text,
                        "forced": forced,
                        "cut_ms": cut_ms,
                    }
                )
            )
            return bool(words)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    async def process_buffer(flush: bool = False) -> None:
        if flush and state.buffer and not buffer_has_nonsilent_audio(
            state.buffer, config
        ):
            state.buffer.clear()
            state.cap_deadline = None
            return
        min_flush_ms = 80
        while True:
            duration_ms = buffer_duration_ms(state.buffer, config)
            if duration_ms <= 0:
                state.cap_deadline = None
                return

            cut_ms = None
            forced = False

            if not flush:
                cut_ms = detect_trailing_silence_cut(state.buffer, config)

            if cut_ms is None:
                max_ms = config.max_segment_sec * 1000
                if duration_ms >= max_ms:
                    now = time.monotonic()
                    if state.cap_deadline is None:
                        state.cap_deadline = now + config.cap_grace_ms / 1000.0
                    if flush or now >= state.cap_deadline:
                        cut_ms = int(max_ms)
                        forced = True
                else:
                    state.cap_deadline = None

            if cut_ms is None:
                if flush:
                    cut_ms = int(duration_ms)
                    forced = True
                else:
                    return

            b_per_ms = bytes_per_ms(config)
            cut_bytes = int(cut_ms * b_per_ms)
            if cut_bytes <= 0 or cut_bytes > len(state.buffer):
                return

            segment_bytes = bytes(state.buffer[:cut_bytes])
            tail = bytes(state.buffer[cut_bytes:])

            state.cap_deadline = None

            if forced and cut_ms < 200:
                log_event(
                    logging.INFO,
                    "parakeet_stream_cut",
                    request_id=request_id,
                    cut_ms=cut_ms,
                    duration_ms=int(duration_ms),
                    buffer_bytes=len(state.buffer),
                    flush=flush,
                    max_segment_sec=config.max_segment_sec,
                    min_silence_ms=config.min_silence_ms,
                    overlap_ms=config.overlap_ms,
                )

            if flush and forced and cut_ms < min_flush_ms:
                state.buffer = bytearray(tail)
                return

            if flush and forced and not segment_has_nonsilent_audio(
                segment_bytes, config
            ):
                state.buffer = bytearray(tail)
                return

            timestamps_used = await emit_segment(segment_bytes, cut_ms, forced)

            if forced and config.overlap_ms > 0 and not timestamps_used and not flush:
                overlap_bytes = int(config.overlap_ms * b_per_ms)
                if overlap_bytes > 0 and overlap_bytes < len(segment_bytes) and tail:
                    prefix = segment_bytes[-overlap_bytes:]
                    state.buffer = bytearray(prefix + tail)
                else:
                    state.buffer = bytearray(tail)
            else:
                state.buffer = bytearray(tail)

            if not state.buffer:
                return

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                if chunk:
                    state.buffer.extend(chunk)
                    await process_buffer(flush=False)
            elif "text" in message and message["text"] is not None:
                text = message["text"].strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {}
                if payload.get("type") == "config":
                    apply_streaming_overrides(config, payload)
                    await websocket.send_text(json.dumps({"type": "config", "ok": True}))
                elif payload.get("type") == "end" or text == "end":
                    await process_buffer(flush=True)
                    await websocket.send_text(json.dumps({"type": "end", "ok": True}))
                elif payload.get("type") == "close":
                    await websocket.close()
                    return
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

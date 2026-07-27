import asyncio
import logging
from typing import TypedDict

import torch
from fastapi import HTTPException, status
from faster_whisper import WhisperModel

from siren import config
from siren.backends.base import TranscriptionBackend
from siren.backends.parakeet import load_parakeet_backend
from siren.backends.whisper import WhisperBackend
from siren.concurrency import run_in_worker_thread
from siren.logging_utils import log_event
from siren.schemas import ModelInfo

current_backend: TranscriptionBackend | None = None
current_model_name: str | None = None
model_lock = asyncio.Lock()
inference_semaphore = asyncio.Semaphore(1)
model_ready = asyncio.Event()
model_loading_task: asyncio.Task[None] | None = None
model_loading_target: str | None = None
model_load_error: Exception | None = None


class WhisperParams(TypedDict):
    device: str
    compute_type: str


def get_whisper_params() -> WhisperParams:
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


def get_available_models() -> list[ModelInfo]:
    return [
        ModelInfo(id=model_name)
        for model_name in config.WHISPER_MODELS + config.PARAKEET_MODELS
    ]


def is_parakeet_model(model_name: str) -> bool:
    return model_name in config.PARAKEET_MODELS or model_name.startswith(
        "nvidia/parakeet"
    )


def load_backend(
    model_name: str,
    *,
    request_id: str | None = None,
) -> TranscriptionBackend:
    fields: dict[str, object] = {"model": model_name}
    if request_id is not None:
        fields["request_id"] = request_id
    log_event(logging.INFO, "model_load_started", **fields)
    if is_parakeet_model(model_name):
        return load_parakeet_backend(model_name)
    return WhisperBackend(WhisperModel(model_name, **get_whisper_params()))


async def _load_model_in_background(
    target_model: str,
    *,
    request_id: str | None = None,
) -> None:
    global current_backend, current_model_name, model_load_error
    try:
        if current_backend is not None:
            current_backend = None
            torch.cuda.empty_cache()

        backend = await run_in_worker_thread(
            load_backend,
            target_model,
            request_id=request_id,
        )
        current_backend = backend
        current_model_name = target_model
        config.save_model_name(target_model, request_id=request_id)
        fields: dict[str, object] = {"model": target_model}
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(logging.INFO, "model_loaded", **fields)
    except Exception as exc:
        model_load_error = exc
        fields = {
            "model": target_model,
            "error": str(exc),
        }
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(
            logging.ERROR,
            "model_load_failed",
            **fields,
        )
    finally:
        model_ready.set()


async def ensure_model_loaded(
    target_model: str,
    *,
    request_id: str | None = None,
) -> TranscriptionBackend:
    global model_loading_task, model_loading_target, model_load_error

    async with model_lock:
        if (
            target_model == current_model_name
            and current_backend is not None
            and model_load_error is None
        ):
            return current_backend

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
                _load_model_in_background(
                    target_model,
                    request_id=request_id,
                )
            )

        task = model_loading_task

    if task is not None:
        await task

    if model_load_error is not None:
        raise model_load_error

    if current_backend is None:
        raise RuntimeError(f"Model '{target_model}' failed to load")

    return current_backend


def resolve_transcription_model_name(model: str | None) -> str:
    available_models = {model_info.id for model_info in get_available_models()}
    target_model = model if model is not None else current_model_name
    if target_model not in available_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model: '{target_model}'. Use /v1/models to see available models.",
        )
    return target_model


async def get_transcription_backend(
    model: str | None = None,
    *,
    request_id: str | None = None,
) -> TranscriptionBackend:
    target_model = resolve_transcription_model_name(model)

    if (
        target_model == current_model_name
        and current_backend is not None
        and model_load_error is None
    ):
        return current_backend

    fields: dict[str, object] = {
        "from_model": current_model_name,
        "to_model": target_model,
    }
    if request_id is not None:
        fields["request_id"] = request_id
    log_event(logging.INFO, "model_switch_requested", **fields)
    try:
        return await ensure_model_loaded(target_model, request_id=request_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{target_model}': {str(exc)}",
        )


def reset_model_state() -> None:
    global current_backend, current_model_name
    global model_loading_task, model_loading_target, model_load_error
    global model_lock, inference_semaphore, model_ready

    current_backend = None
    current_model_name = None
    model_lock = asyncio.Lock()
    inference_semaphore = asyncio.Semaphore(1)
    model_ready = asyncio.Event()
    model_loading_task = None
    model_loading_target = None
    model_load_error = None


def unload_model() -> None:
    global current_backend

    current_backend = None
    torch.cuda.empty_cache()

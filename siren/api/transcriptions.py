import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from siren import config, models
from siren.api.auth import verify_token
from siren.audio import ensure_16k_wav, get_wav_info, save_upload_file
from siren.backends.speech_gate import file_has_speech
from siren.concurrency import run_in_worker_thread
from siren.logging_utils import log_event
from siren.schemas import (
    TranscriptionResponse,
    TranscriptionResult,
    VerboseTranscriptionResponse,
)
from siren.segmentation import segment_words
from siren.vocabulary import MAX_TERM_CHARS, MAX_TERMS, parse_vocabulary

router = APIRouter()


async def _speech_gate_rejects(audio_path: str, *, request_id: str) -> tuple[bool, int]:
    """Return (no speech found, gate ms). A failing check lets the request through."""
    start = time.perf_counter()
    try:
        speech = await run_in_worker_thread(file_has_speech, audio_path)
    except Exception as exc:
        log_event(
            logging.WARNING,
            "speech_gate_failed",
            request_id=request_id,
            error=str(exc),
        )
        speech = True
    gate_ms = int((time.perf_counter() - start) * 1000)
    if not speech:
        log_event(
            logging.INFO,
            "transcribe_gated",
            request_id=request_id,
            gate_ms=gate_ms,
        )
    return not speech, gate_ms


@router.post(
    "/v1/audio/transcriptions",
    response_model=VerboseTranscriptionResponse | TranscriptionResponse,
    response_model_exclude_none=True,
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
    response_format: str = Form(
        "json",
        description="Response shape: json for text only, or verbose_json for timestamped segments.",
    ),
    timestamp_granularities: list[str] | None = Form(
        None,
        alias="timestamp_granularities[]",
        description="Timestamp granularities to populate for verbose_json responses. Supported timestamp_granularities[] values: word and segment.",
    ),
    segmentation: str = Form(
        "native",
        description="Segment construction mode. Supported segmentation values: native (backend-native segments) and pause (word-driven Parakeet segments).",
    ),
    vocabulary: str | None = Form(
        None,
        description=(
            "JSON array of terms to boost for nvidia/parakeet-tdt-0.6b-v2, "
            f"at most {MAX_TERMS} terms of 1-{MAX_TERM_CHARS} characters. "
            "Omit it to use the server default; [] disables boosting. "
            "Other models ignore it."
        ),
    ),
) -> VerboseTranscriptionResponse | TranscriptionResponse:
    original_path: str | None = None
    converted_path: str | None = None
    request_id = uuid.uuid4().hex
    request_start = time.perf_counter()
    try:
        if segmentation not in {"native", "pause"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid segmentation: '{segmentation}'. Supported values: 'native', 'pause'.",
            )
        if segmentation == "pause" and response_format != "verbose_json":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="segmentation=pause requires response_format=verbose_json.",
            )
        if timestamp_granularities:
            unknown = set(timestamp_granularities) - {"word", "segment"}
            if unknown:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid timestamp_granularities: {sorted(unknown)}. Supported values: 'word', 'segment'.",
                )
            if response_format != "verbose_json":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="timestamp_granularities requires response_format=verbose_json.",
                )
        boost_terms: list[str] | None = None
        if vocabulary is not None:
            try:
                boost_terms = parse_vocabulary(vocabulary)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid vocabulary: {exc}",
                )
        return_word_timestamps = (
            timestamp_granularities is not None
            and "word" in timestamp_granularities
        )
        word_timestamps = return_word_timestamps or segmentation == "pause"
        target_model = models.resolve_transcription_model_name(model)
        if segmentation == "pause" and not models.is_parakeet_model(target_model):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="segmentation=pause is currently supported only for Parakeet models",
            )

        original_path = await save_upload_file(file, request_id=request_id)
        audio_path = await ensure_16k_wav(original_path, request_id=request_id)
        if audio_path != original_path:
            converted_path = audio_path
        audio_size = Path(original_path).stat().st_size
        audio_info = get_wav_info(audio_path, request_id=request_id)
        log_event(
            logging.INFO,
            "transcribe_request",
            request_id=request_id,
            model=target_model,
            language=language,
            filename=file.filename,
            audio_bytes=audio_size,
            word_timestamps=word_timestamps,
            segmentation=segmentation,
            vocabulary_terms=None if boost_terms is None else len(boost_terms),
            **audio_info,
        )

        gated = False
        gate_ms: int | None = None
        if models.is_parakeet_model(target_model) and config.parakeet_speech_gate_enabled():
            # Silero runs on the CPU, so the gate stays outside the GPU semaphore.
            gated, gate_ms = await _speech_gate_rejects(audio_path, request_id=request_id)
        if gated:
            result = TranscriptionResult(
                text="",
                language="en",
                duration=float(audio_info.get("audio_duration_sec", 0.0)),
                segments=[],
            )
        else:
            async with models.inference_semaphore:
                backend = await models.get_transcription_backend(
                    target_model,
                    request_id=request_id,
                )
                result = await backend.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    request_id=request_id,
                    vocabulary=boost_terms,
                )

        if segmentation == "pause" and not gated:
            words = [
                word
                for segment in result.segments
                for word in (segment.words or [])
            ]
            if words:
                result.segments = segment_words(words)
            else:
                log_event(
                    logging.WARNING,
                    "segmentation_fallback_native",
                    request_id=request_id,
                    reason="backend returned no word timestamps",
                )

        total_ms = int((time.perf_counter() - request_start) * 1000)
        log_event(
            logging.INFO,
            "transcribe_complete",
            request_id=request_id,
            model=target_model,
            latency_ms=total_ms,
            text_length=len(result.text),
            segment_count=len(result.segments),
            word_timestamps=word_timestamps,
            segmentation=segmentation,
            gated=gated,
            gate_ms=gate_ms,
        )
        if response_format == "verbose_json":
            payload = result.model_dump()
            if not return_word_timestamps:
                for segment in payload["segments"]:
                    segment.pop("words", None)
            return VerboseTranscriptionResponse(**payload)
        return TranscriptionResponse(text=result.text)
    except HTTPException as exc:
        log_event(
            logging.ERROR,
            "transcribe_error",
            request_id=request_id,
            status_code=exc.status_code,
            error=str(exc.detail),
        )
        raise
    except Exception as exc:
        log_event(
            logging.ERROR,
            "transcribe_error",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(exc)}",
        )
    finally:
        await file.close()
        for path in {converted_path, original_path}:
            if path:
                Path(path).unlink(missing_ok=True)

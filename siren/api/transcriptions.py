import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from siren import models
from siren.api.auth import verify_token
from siren.audio import ensure_16k_wav, get_wav_info, save_upload_file
from siren.logging_utils import log_event
from siren.schemas import TranscriptionResponse, VerboseTranscriptionResponse
from siren.segmentation import segment_words

router = APIRouter()


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
            **audio_info,
        )

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
            )

        if segmentation == "pause":
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

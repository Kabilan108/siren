import asyncio
import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from siren.alignment import align_words
from siren.audio import get_wav_info
from siren.diarization import get_diarization_model
from siren.diarization.worker import normalize_speaker_label, parse_turns
from siren.jobs import get_chunk_seconds, get_memory_fraction
from siren.logging_utils import log_event
from siren.models import load_backend
from siren.schemas import DiarizationTurn, TranscriptionWord
from siren.io import atomic_write_json

_PARENT_POLL_SECONDS = 5.0
_SILENCE_DURATION_SECONDS = 0.4
_SILENCE_WINDOW_SECONDS = 20.0
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def configure_worker_logging() -> None:
    logger = logging.getLogger("uvicorn")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _start_parent_watchdog(
    parent_pid: int,
    *,
    poll_seconds: float = _PARENT_POLL_SECONDS,
) -> threading.Thread:
    def watch_parent() -> None:
        while True:
            time.sleep(poll_seconds)
            current_parent_pid = os.getppid()
            if current_parent_pid != parent_pid:
                log_event(
                    logging.ERROR,
                    "job_worker_parent_changed",
                    parent_pid=parent_pid,
                    current_parent_pid=current_parent_pid,
                )
                os._exit(1)

    watchdog = threading.Thread(
        target=watch_parent,
        name="job-parent-watchdog",
        daemon=True,
    )
    watchdog.start()
    return watchdog


def silence_aligned_cut_points(
    silence_intervals: Sequence[tuple[float, float]],
    duration: float,
    *,
    chunk_seconds: float = 300.0,
    window_seconds: float = 20.0,
) -> list[float]:
    """Choose silence midpoints nearest fixed chunk boundaries."""
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and non-negative")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be finite and greater than zero")
    if not math.isfinite(window_seconds) or window_seconds < 0.0:
        raise ValueError("window_seconds must be finite and non-negative")

    midpoints: list[float] = []
    for start, end in silence_intervals:
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start > end
            or end < 0.0
            or start > duration
        ):
            continue
        midpoints.append((max(0.0, start) + min(duration, end)) / 2.0)
    midpoints = sorted(set(midpoints))
    cuts: list[float] = []
    target = chunk_seconds
    while target < duration:
        candidates = [
            midpoint
            for midpoint in midpoints
            if abs(midpoint - target) <= window_seconds
            and (not cuts or midpoint > cuts[-1])
        ]
        cut = (
            min(
                candidates,
                key=lambda midpoint: (abs(midpoint - target), midpoint),
            )
            if candidates
            else target
        )
        if cut > 0.0 and cut < duration and (not cuts or cut > cuts[-1]):
            cuts.append(cut)
        target += chunk_seconds
    return cuts


def parse_silence_intervals(stderr: str, duration: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match is not None:
            current_start = max(0.0, float(start_match.group(1)))
        end_match = _SILENCE_END.search(line)
        if end_match is not None and current_start is not None:
            end = min(duration, float(end_match.group(1)))
            if end >= current_start:
                intervals.append((current_start, end))
            current_start = None
    if current_start is not None and current_start <= duration:
        intervals.append((current_start, duration))
    return intervals


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise RuntimeError(message)
    return result


def _convert_to_wav(input_path: Path, wav_path: Path) -> None:
    _run_command(
        (
            "ffmpeg",
            "-y",
            "-protocol_whitelist",
            "file,pipe,crypto",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        )
    )


def _detect_silences(wav_path: Path, duration: float) -> list[tuple[float, float]]:
    result = _run_command(
        (
            "ffmpeg",
            "-hide_banner",
            "-protocol_whitelist",
            "file,pipe,crypto",
            "-i",
            str(wav_path),
            "-af",
            f"silencedetect=noise=-35dB:d={_SILENCE_DURATION_SECONDS}",
            "-f",
            "null",
            "-",
        )
    )
    return parse_silence_intervals(result.stderr, duration)


def _split_wav(
    wav_path: Path,
    scratch_dir: Path,
    cut_points: Sequence[float],
    duration: float,
) -> list[tuple[Path, float]]:
    boundaries = [0.0, *cut_points, duration]
    chunks: list[tuple[Path, float]] = []
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        if end <= start:
            continue
        chunk_path = scratch_dir / f"chunk-{index:04d}.wav"
        _run_command(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-protocol_whitelist",
                "file,pipe,crypto",
                "-i",
                str(wav_path),
                "-t",
                f"{end - start:.6f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            )
        )
        chunks.append((chunk_path, start))
    return chunks


def _emit_phase(phase: str, progress: float) -> None:
    log_event(
        logging.INFO,
        "job_phase",
        phase=phase,
        progress=min(1.0, max(0.0, progress)),
    )


async def _transcribe_chunks(
    chunks: Sequence[tuple[Path, float]],
    *,
    model_name: str,
    language: str | None,
) -> tuple[str, str, list[TranscriptionWord]]:
    backend = load_backend(model_name)
    texts: list[str] = []
    words: list[TranscriptionWord] = []
    detected_language = language or "en"
    try:
        for index, (chunk_path, start_offset) in enumerate(chunks):
            result = await backend.transcribe(
                str(chunk_path),
                language=language,
                word_timestamps=True,
            )
            if result.text.strip():
                texts.append(result.text.strip())
            detected_language = result.language or detected_language
            for segment in result.segments:
                for word in segment.words or []:
                    words.append(
                        TranscriptionWord(
                            start=word.start + start_offset,
                            end=word.end + start_offset,
                            word=word.word,
                        )
                    )
            _emit_phase(
                "transcribing",
                0.1 + (0.6 * (index + 1) / max(1, len(chunks))),
            )
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    return " ".join(texts), detected_language, words


def _load_sortformer_model() -> type[Any]:
    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel


def _diarize(wav_path: Path) -> tuple[str, list[DiarizationTurn]]:
    model_name = get_diarization_model()
    model = _load_sortformer_model().from_pretrained(model_name).cuda().eval()
    try:
        with torch.inference_mode():
            outputs = model.diarize(audio=[str(wav_path)], batch_size=1)
        turns = [
            DiarizationTurn(
                start=float(turn["start"]),
                end=float(turn["end"]),
                speaker=normalize_speaker_label(str(turn["speaker"])),
            )
            for turn in parse_turns(outputs[0])
        ]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return model_name, turns


def _load_job(job_dir: Path) -> dict[str, object]:
    payload = json.loads((job_dir / "job.json").read_text())
    if not isinstance(payload, dict):
        raise ValueError("job.json does not contain a JSON object")
    return payload


def run_pipeline(job_dir: Path) -> dict[str, object]:
    job = _load_job(job_dir)
    input_filename = job.get("input_file")
    model_name = job.get("model")
    language = job.get("language")
    if not isinstance(input_filename, str):
        raise ValueError("job.json is missing input_file")
    if not isinstance(model_name, str):
        raise ValueError("job.json is missing model")
    if language is not None and not isinstance(language, str):
        raise ValueError("job.json language must be a string or null")

    result_path = job_dir / "result.json"
    result_path.unlink(missing_ok=True)
    scratch_dir = job_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    wav_path = scratch_dir / "audio.wav"

    torch.cuda.set_per_process_memory_fraction(get_memory_fraction())
    _emit_phase("chunking", 0.01)
    _convert_to_wav(job_dir / input_filename, wav_path)
    duration = float(get_wav_info(str(wav_path)).get("audio_duration_sec", 0.0))
    silences = _detect_silences(wav_path, duration)
    cuts = silence_aligned_cut_points(
        silences,
        duration,
        chunk_seconds=get_chunk_seconds(),
        window_seconds=_SILENCE_WINDOW_SECONDS,
    )
    chunks = _split_wav(wav_path, scratch_dir, cuts, duration)
    if not chunks:
        raise RuntimeError("Audio produced no transcription chunks")
    _emit_phase("chunking", 0.1)

    _emit_phase("transcribing", 0.1)
    text, detected_language, words = asyncio.run(
        _transcribe_chunks(
            chunks,
            model_name=model_name,
            language=language,
        )
    )

    _emit_phase("diarizing", 0.72)
    diarization_model, turns = _diarize(wav_path)
    _emit_phase("diarizing", 0.9)

    _emit_phase("aligning", 0.92)
    alignment = align_words(words, turns)
    payload: dict[str, object] = {
        "text": text,
        "language": detected_language,
        "duration": duration,
        "model": model_name,
        "diarization_model": diarization_model,
        "speakers": alignment.speakers,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
            }
            for segment in alignment.segments
        ],
    }
    atomic_write_json(result_path, payload)
    _emit_phase("aligning", 0.99)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    configure_worker_logging()
    _start_parent_watchdog(os.getppid())
    arguments = list(sys.argv[1:] if argv is None else argv)
    job_dir: Path | None = None
    try:
        if len(arguments) != 1:
            raise ValueError("Expected arguments: <job_dir>")
        job_dir = Path(arguments[0])
        run_pipeline(job_dir)
    except BaseException as exc:
        if job_dir is not None:
            (job_dir / "result.json").unlink(missing_ok=True)
        log_event(logging.ERROR, "job_error", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

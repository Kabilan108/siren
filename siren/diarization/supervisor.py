import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

from fastapi import HTTPException, status

from siren.diarization import get_timeout_seconds
from siren.logging_utils import log_event

diarize_lock = asyncio.Lock()
active_worker_process: asyncio.subprocess.Process | None = None
_TERMINATION_GRACE_SECONDS = 10.0
_STDERR_READ_BYTES = 65536
_MAX_STDERR_LINE_BYTES = 65536


def _worker_command(audio_path: str, output_path: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "siren.diarization.worker",
        audio_path,
        output_path,
    )


def _log_worker_stderr_line(raw_line: bytes, *, request_id: str) -> None:
    decoded_line = raw_line.decode(errors="replace")
    try:
        line: object = json.loads(decoded_line)
    except json.JSONDecodeError:
        line = decoded_line
    log_event(
        logging.INFO,
        "diarize_worker_log",
        request_id=request_id,
        line=line,
    )


async def _relay_stderr(
    stderr: asyncio.StreamReader,
    *,
    request_id: str,
) -> None:
    buffered_line = bytearray()
    discarding_truncated_line = False

    while chunk := await stderr.read(_STDERR_READ_BYTES):
        offset = 0
        while offset < len(chunk):
            newline_index = chunk.find(b"\n", offset)
            segment_end = len(chunk) if newline_index == -1 else newline_index
            segment = chunk[offset:segment_end]

            if not discarding_truncated_line:
                remaining_bytes = _MAX_STDERR_LINE_BYTES - len(buffered_line)
                if len(segment) > remaining_bytes:
                    buffered_line.extend(segment[:remaining_bytes])
                    _log_worker_stderr_line(
                        bytes(buffered_line) + b"...truncated",
                        request_id=request_id,
                    )
                    buffered_line.clear()
                    discarding_truncated_line = True
                else:
                    buffered_line.extend(segment)

            if newline_index == -1:
                break

            if not discarding_truncated_line:
                _log_worker_stderr_line(
                    bytes(buffered_line).rstrip(b"\r"),
                    request_id=request_id,
                )
            buffered_line.clear()
            discarding_truncated_line = False
            offset = newline_index + 1

    if buffered_line and not discarding_truncated_line:
        _log_worker_stderr_line(
            bytes(buffered_line).rstrip(b"\r"),
            request_id=request_id,
        )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
) -> None:
    process_group_id = process.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)

    try:
        await asyncio.wait_for(
            asyncio.shield(process.wait()),
            timeout=_TERMINATION_GRACE_SECONDS,
        )
    except TimeoutError:
        pass

    if _process_group_exists(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


async def terminate_active_worker() -> None:
    global active_worker_process

    process = active_worker_process
    if process is None:
        return
    try:
        if _process_group_exists(process.pid):
            log_event(
                logging.INFO,
                "diarize_worker_shutdown",
                pid=process.pid,
            )
            await _terminate_process_group(process)
    finally:
        if active_worker_process is process:
            active_worker_process = None


def _load_result(output_path: Path) -> dict[str, object]:
    payload = json.loads(output_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Diarization worker returned a non-object result")
    return payload


async def run_diarization(
    audio_path: str,
    *,
    timeout_seconds: float | None = None,
    request_id: str,
) -> dict[str, object]:
    global active_worker_process

    request_start = time.perf_counter()
    timeout = get_timeout_seconds() if timeout_seconds is None else timeout_seconds
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task[None] | None = None

    try:
        async with diarize_lock:
            with tempfile.TemporaryDirectory(prefix="siren-diarize-") as temp_dir:
                output_path = Path(temp_dir) / "result.json"
                process = await asyncio.create_subprocess_exec(
                    *_worker_command(audio_path, str(output_path)),
                    start_new_session=True,
                    stderr=asyncio.subprocess.PIPE,
                )
                active_worker_process = process
                log_event(
                    logging.INFO,
                    "diarize_worker_started",
                    request_id=request_id,
                    pid=process.pid,
                )
                if process.stderr is None:
                    raise RuntimeError("Diarization worker stderr pipe is unavailable")
                stderr_task = asyncio.create_task(
                    _relay_stderr(process.stderr, request_id=request_id)
                )

                try:
                    await asyncio.wait_for(
                        asyncio.shield(process.wait()),
                        timeout=timeout,
                    )
                except TimeoutError:
                    await _terminate_process_group(process)
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail="Diarization timed out",
                    )
                except BaseException:
                    await _terminate_process_group(process)
                    raise
                finally:
                    if stderr_task is not None:
                        try:
                            await stderr_task
                        except Exception as relay_exc:
                            log_event(
                                logging.ERROR,
                                "diarize_worker_relay_failed",
                                request_id=request_id,
                                error=str(relay_exc),
                            )

                if process.returncode != 0:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Diarization worker failed",
                    )

                result = _load_result(output_path)

        turns = result.get("turns", [])
        speakers = result.get("speakers", [])
        log_event(
            logging.INFO,
            "diarize_complete",
            request_id=request_id,
            latency_ms=int((time.perf_counter() - request_start) * 1000),
            n_turns=len(turns) if isinstance(turns, list) else 0,
            n_speakers=len(speakers) if isinstance(speakers, list) else 0,
        )
        return result
    except BaseException as exc:
        if process is not None and _process_group_exists(process.pid):
            await _terminate_process_group(process)
        fields: dict[str, object] = {
            "request_id": request_id,
            "error": str(exc),
        }
        if isinstance(exc, HTTPException):
            fields["status_code"] = exc.status_code
        log_event(logging.ERROR, "diarize_error", **fields)
        if isinstance(exc, HTTPException):
            raise
        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diarization failed",
        )
    finally:
        if active_worker_process is process:
            active_worker_process = None

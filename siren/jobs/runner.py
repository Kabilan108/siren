import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import UploadFile

from siren.config import UPLOAD_CHUNK_BYTES
from siren.diarization.supervisor import _terminate_process_group
from siren.jobs import (
    get_max_concurrent_jobs,
    get_max_queued_jobs,
    get_max_upload_bytes,
    get_retention_days,
    get_spool_dir,
    get_timeout_seconds,
)
from siren.gpu import batch_gpu_lock
from siren.logging_utils import log_event
from siren.schemas import TranscriptJobResult

_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_PHASES = {"chunking", "transcribing", "diarizing", "aligning"}
_STDERR_READ_BYTES = 65536
_MAX_STDERR_LINE_BYTES = 65536

WorkerCommand = Callable[[Path], tuple[str, ...]]


class JobQueueFullError(Exception):
    def __init__(self, max_queued_jobs: int) -> None:
        self.max_queued_jobs = max_queued_jobs
        super().__init__(
            f"Job queue is full (maximum {max_queued_jobs} queued jobs)"
        )


class JobUploadTooLargeError(Exception):
    def __init__(self, max_upload_bytes: int) -> None:
        self.max_upload_bytes = max_upload_bytes
        super().__init__(
            f"Upload exceeds maximum size of {max_upload_bytes} bytes"
        )


def _default_worker_command(job_dir: Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "siren.jobs.worker", str(job_dir))


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, separators=(",", ":"))
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a JSON object")
    return payload


def _input_suffix(filename: str | None) -> str:
    suffix = Path(filename or "audio.wav").suffix.lower()
    if not suffix or len(suffix) > 16 or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        return ".bin"
    return suffix


class JobRunner:
    def __init__(
        self,
        *,
        spool_dir: Path | None = None,
        max_concurrent_jobs: int | None = None,
        max_queued_jobs: int | None = None,
        max_upload_bytes: int | None = None,
        timeout_seconds: float | None = None,
        retention_days: float | None = None,
        worker_command: WorkerCommand | None = None,
    ) -> None:
        self.spool_dir = spool_dir if spool_dir is not None else get_spool_dir()
        self.max_concurrent_jobs = (
            max_concurrent_jobs
            if max_concurrent_jobs is not None
            else get_max_concurrent_jobs()
        )
        self.max_queued_jobs = (
            max_queued_jobs
            if max_queued_jobs is not None
            else get_max_queued_jobs()
        )
        self.max_upload_bytes = (
            max_upload_bytes
            if max_upload_bytes is not None
            else get_max_upload_bytes()
        )
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else get_timeout_seconds()
        )
        self.retention_days = (
            retention_days if retention_days is not None else get_retention_days()
        )
        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")
        if self.max_queued_jobs < 1:
            raise ValueError("max_queued_jobs must be at least 1")
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be at least 1")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be greater than 0")
        if self.retention_days < 0.0:
            raise ValueError("retention_days must not be negative")

        self._worker_command = worker_command or _default_worker_command
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued_order: list[str] = []
        self._queue_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._running_job_ids: set[str] = set()
        self.active_workers: dict[str, asyncio.subprocess.Process] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._queue = asyncio.Queue()
        self._queued_order.clear()
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.sweep()
        self.recover_interrupted_jobs()
        self._started = True
        self._dispatcher_task = asyncio.create_task(
            self._dispatch(),
            name="siren-job-dispatcher",
        )

    async def stop(self) -> None:
        if not self._started:
            return
        dispatcher = self._dispatcher_task
        self._dispatcher_task = None
        if dispatcher is not None:
            dispatcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await dispatcher

        job_tasks = tuple(self._job_tasks)
        for task in job_tasks:
            task.cancel()
        if job_tasks:
            await asyncio.gather(*job_tasks, return_exceptions=True)

        remaining_workers = list(self.active_workers.values())
        if remaining_workers:
            await asyncio.gather(
                *(
                    _terminate_process_group(process)
                    for process in remaining_workers
                ),
                return_exceptions=True,
            )

        async with self._queue_lock:
            queued_job_ids = tuple(self._queued_order)
            self._queued_order.clear()
        for job_id in queued_job_ids:
            state = self._load_state(job_id)
            if state is not None and state.get("status") == "queued":
                self._persist_failure(
                    job_id,
                    error="server_shutdown",
                    exit_code=None,
                )

        self.active_workers.clear()
        self._running_job_ids.clear()
        self.sweep()
        self._started = False

    async def enqueue(
        self,
        upload: UploadFile,
        *,
        model: str,
        language: str | None,
    ) -> tuple[str, int]:
        await self.start()
        async with self._queue_lock:
            if len(self._queued_order) >= self.max_queued_jobs:
                raise JobQueueFullError(self.max_queued_jobs)

        job_id = f"job_{uuid.uuid4().hex}"
        job_dir = self.spool_dir / job_id
        input_path = job_dir / f"input{_input_suffix(upload.filename)}"
        temporary_input_path = job_dir / f".{input_path.name}.upload"
        job_dir_created = False
        try:
            job_dir.mkdir(mode=0o700)
            job_dir_created = True
            job_dir.chmod(0o700)
            upload_bytes = 0
            with temporary_input_path.open("wb") as output_file:
                while content := await upload.read(UPLOAD_CHUNK_BYTES):
                    upload_bytes += len(content)
                    if upload_bytes > self.max_upload_bytes:
                        raise JobUploadTooLargeError(self.max_upload_bytes)
                    output_file.write(content)
                output_file.flush()
                os.fsync(output_file.fileno())
            temporary_input_path.replace(input_path)
            now = time.time()
            state: dict[str, object] = {
                "id": job_id,
                "status": "queued",
                "phase": None,
                "progress": 0.0,
                "model": model,
                "language": language,
                "input_file": input_path.name,
                "created_at": now,
                "updated_at": now,
            }
            async with self._queue_lock:
                if len(self._queued_order) >= self.max_queued_jobs:
                    raise JobQueueFullError(self.max_queued_jobs)
                _atomic_write_json(job_dir / "job.json", state)
                self._queued_order.append(job_id)
                position = len(self._queued_order)
                self._queue.put_nowait(job_id)
            log_event(
                logging.INFO,
                "job_queued",
                id=job_id,
                position=position,
            )
            return job_id, position
        except BaseException:
            temporary_input_path.unlink(missing_ok=True)
            if job_dir_created:
                shutil.rmtree(job_dir, ignore_errors=True)
            raise

    async def status(self, job_id: str) -> dict[str, object] | None:
        state = self._load_state(job_id)
        if state is None:
            return None
        if state.get("status") == "queued":
            async with self._queue_lock:
                try:
                    state["position"] = self._queued_order.index(job_id) + 1
                except ValueError:
                    state["position"] = 1
        return state

    def result(self, job_id: str) -> dict[str, object] | None:
        state = self._load_state(job_id)
        if state is None or state.get("status") != "completed":
            return None
        result = self._load_valid_result(self.spool_dir / job_id)
        if result is None:
            return None
        return result.model_dump()

    def recover_interrupted_jobs(self) -> int:
        recovered = 0
        for job_dir in self._job_directories():
            state_path = job_dir / "job.json"
            if not self._is_regular_file(state_path):
                continue
            try:
                state = _read_json_object(state_path)
            except Exception as exc:
                log_event(
                    logging.ERROR,
                    "job_recovery_failed",
                    path=str(state_path),
                    error=str(exc),
                )
                continue
            status = state.get("status")
            if status not in {"queued", "running"}:
                continue
            if status == "running":
                result = self._load_valid_result(job_dir)
                if result is not None:
                    state.update(
                        {
                            "status": "completed",
                            "phase": None,
                            "progress": 1.0,
                            "updated_at": time.time(),
                        }
                    )
                    state.pop("error", None)
                    _atomic_write_json(state_path, state)
                    recovered += 1
                    log_event(
                        logging.INFO,
                        "job_completed",
                        id=state.get("id", job_dir.name),
                        latency_ms=0,
                        n_speakers=len(result.speakers),
                        n_segments=len(result.segments),
                    )
                    continue
            error = (
                "orphaned by server restart"
                if status == "running"
                else "server restarted before start"
            )
            state.update(
                {
                    "status": "failed",
                    "phase": None,
                    "error": error,
                    "updated_at": time.time(),
                }
            )
            _atomic_write_json(state_path, state)
            recovered += 1
            log_event(
                logging.ERROR,
                "job_failed",
                id=state.get("id", job_dir.name),
                error=error,
                exit_code=None,
            )
        return recovered

    def sweep(self, *, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - (
            self.retention_days * 24.0 * 60.0 * 60.0
        )
        deleted = 0
        active_job_ids = (
            set(self.active_workers)
            | self._running_job_ids
            | set(self._queued_order)
        )
        for job_dir in self._job_directories():
            if job_dir.name in active_job_ids:
                continue
            state_path = job_dir / "job.json"
            try:
                try:
                    state_mode = state_path.lstat().st_mode
                except FileNotFoundError:
                    modified_at = job_dir.lstat().st_mtime
                else:
                    if stat.S_ISLNK(state_mode) or not stat.S_ISREG(state_mode):
                        continue
                    modified_at = state_path.lstat().st_mtime
            except OSError as exc:
                log_event(
                    logging.ERROR,
                    "spool_sweep_failed",
                    path=str(job_dir),
                    error=str(exc),
                )
                continue
            if modified_at >= cutoff:
                continue
            try:
                shutil.rmtree(job_dir)
            except Exception as exc:
                log_event(
                    logging.ERROR,
                    "spool_sweep_failed",
                    path=str(job_dir),
                    error=str(exc),
                )
                continue
            deleted += 1
        log_event(logging.INFO, "spool_swept", n_deleted=deleted)
        return deleted

    def _job_directories(self) -> list[Path]:
        directories: list[Path] = []
        try:
            entries = self.spool_dir.iterdir()
            for entry in entries:
                if _JOB_ID.fullmatch(entry.name) is None:
                    continue
                try:
                    mode = entry.lstat().st_mode
                except OSError as exc:
                    log_event(
                        logging.ERROR,
                        "spool_scan_failed",
                        path=str(entry),
                        error=str(exc),
                    )
                    continue
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    continue
                directories.append(entry)
        except OSError as exc:
            log_event(
                logging.ERROR,
                "spool_scan_failed",
                path=str(self.spool_dir),
                error=str(exc),
            )
        return directories

    @staticmethod
    def _is_regular_file(path: Path) -> bool:
        try:
            mode = path.lstat().st_mode
        except OSError:
            return False
        return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)

    def _load_valid_result(
        self,
        job_dir: Path,
    ) -> TranscriptJobResult | None:
        result_path = job_dir / "result.json"
        if not self._is_regular_file(result_path):
            return None
        try:
            return TranscriptJobResult.model_validate(
                _read_json_object(result_path)
            )
        except Exception:
            return None

    def _load_state(self, job_id: str) -> dict[str, object] | None:
        if _JOB_ID.fullmatch(job_id) is None:
            return None
        job_dir = self.spool_dir / job_id
        try:
            mode = job_dir.lstat().st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            return None
        state_path = job_dir / "job.json"
        if not self._is_regular_file(state_path):
            return None
        try:
            return _read_json_object(state_path)
        except FileNotFoundError:
            return None

    def _write_transition(
        self,
        job_id: str,
        *,
        status: str,
        phase: str | None,
        progress: float,
        error: str | None = None,
    ) -> dict[str, object]:
        state_path = self.spool_dir / job_id / "job.json"
        state = _read_json_object(state_path)
        state.update(
            {
                "status": status,
                "phase": phase,
                "progress": min(1.0, max(0.0, progress)),
                "updated_at": time.time(),
            }
        )
        if error is None:
            state.pop("error", None)
        else:
            state["error"] = error
        _atomic_write_json(state_path, state)
        return state

    def _persist_failure(
        self,
        job_id: str,
        *,
        error: str,
        exit_code: int | None,
    ) -> None:
        current_progress = 0.0
        with contextlib.suppress(Exception):
            state = self._load_state(job_id)
            if state is not None:
                persisted_progress = state.get("progress", 0.0)
                if isinstance(persisted_progress, int | float):
                    current_progress = float(persisted_progress)
        with contextlib.suppress(Exception):
            self._write_transition(
                job_id,
                status="failed",
                phase=None,
                progress=current_progress,
                error=error,
            )
        log_event(
            logging.ERROR,
            "job_failed",
            id=job_id,
            error=error,
            exit_code=exit_code,
        )

    async def _dispatch(self) -> None:
        while True:
            job_id = await self._queue.get()
            await self._semaphore.acquire()
            async with self._queue_lock:
                with contextlib.suppress(ValueError):
                    self._queued_order.remove(job_id)
            task = asyncio.create_task(
                self._run_and_release(job_id),
                name=f"siren-{job_id}",
            )
            self._job_tasks.add(task)
            task.add_done_callback(self._job_tasks.discard)

    async def _run_and_release(self, job_id: str) -> None:
        try:
            await self._run_job(job_id)
        finally:
            self._semaphore.release()
            self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        started_at = time.perf_counter()
        process: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[None] | None = None
        worker_error: dict[str, str] = {}
        exit_code: int | None = None
        gate_acquired = False
        self._running_job_ids.add(job_id)
        try:
            self._write_transition(
                job_id,
                status="running",
                phase=None,
                progress=0.0,
            )
            job_dir = self.spool_dir / job_id
            await batch_gpu_lock.acquire()
            gate_acquired = True
            process = await asyncio.create_subprocess_exec(
                *self._worker_command(job_dir),
                start_new_session=True,
                stderr=asyncio.subprocess.PIPE,
            )
            self.active_workers[job_id] = process
            log_event(logging.INFO, "job_started", id=job_id, pid=process.pid)
            if process.stderr is None:
                raise RuntimeError("Job worker stderr pipe is unavailable")
            stderr_task = asyncio.create_task(
                self._relay_stderr(
                    process.stderr,
                    job_id=job_id,
                    worker_error=worker_error,
                )
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                await _terminate_process_group(process)
                exit_code = process.returncode
                raise RuntimeError(
                    f"Job timed out after {self.timeout_seconds:g} seconds"
                )
            except BaseException:
                await _terminate_process_group(process)
                raise
            finally:
                if stderr_task is not None:
                    try:
                        await stderr_task
                    except Exception as exc:
                        log_event(
                            logging.ERROR,
                            "job_worker_relay_failed",
                            id=job_id,
                            error=str(exc),
                        )

            exit_code = process.returncode
            if exit_code != 0:
                raise RuntimeError(worker_error.get("error", "Job worker failed"))

            result_path = job_dir / "result.json"
            result = TranscriptJobResult.model_validate(
                _read_json_object(result_path)
            )
            self._write_transition(
                job_id,
                status="completed",
                phase=None,
                progress=1.0,
            )
            log_event(
                logging.INFO,
                "job_completed",
                id=job_id,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                n_speakers=len(result.speakers),
                n_segments=len(result.segments),
            )
        except asyncio.CancelledError:
            if process is not None:
                await _terminate_process_group(process)
                exit_code = process.returncode
            if self._load_valid_result(self.spool_dir / job_id) is not None:
                self._write_transition(
                    job_id,
                    status="completed",
                    phase=None,
                    progress=1.0,
                )
            else:
                self._persist_failure(
                    job_id,
                    error="server_shutdown",
                    exit_code=exit_code,
                )
            raise
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            self._persist_failure(
                job_id,
                error=error,
                exit_code=exit_code,
            )
        finally:
            if gate_acquired:
                batch_gpu_lock.release()
            if self.active_workers.get(job_id) is process:
                self.active_workers.pop(job_id, None)
            self._running_job_ids.discard(job_id)
            self.sweep()

    async def _relay_stderr(
        self,
        stderr: asyncio.StreamReader,
        *,
        job_id: str,
        worker_error: dict[str, str],
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
                        self._handle_stderr_line_safely(
                            job_id,
                            bytes(buffered_line) + b"...truncated",
                            worker_error,
                        )
                        buffered_line.clear()
                        discarding_truncated_line = True
                    else:
                        buffered_line.extend(segment)

                if newline_index == -1:
                    break

                if not discarding_truncated_line:
                    self._handle_stderr_line_safely(
                        job_id,
                        bytes(buffered_line).rstrip(b"\r"),
                        worker_error,
                    )
                buffered_line.clear()
                discarding_truncated_line = False
                offset = newline_index + 1

        if buffered_line and not discarding_truncated_line:
            self._handle_stderr_line_safely(
                job_id,
                bytes(buffered_line).rstrip(b"\r"),
                worker_error,
            )

    def _handle_stderr_line_safely(
        self,
        job_id: str,
        raw_line: bytes,
        worker_error: dict[str, str],
    ) -> None:
        try:
            self._handle_stderr_line(job_id, raw_line, worker_error)
        except Exception as exc:
            with contextlib.suppress(Exception):
                log_event(
                    logging.ERROR,
                    "job_worker_line_failed",
                    id=job_id,
                    error=str(exc),
                )

    def _handle_stderr_line(
        self,
        job_id: str,
        raw_line: bytes,
        worker_error: dict[str, str],
    ) -> None:
        decoded_line = raw_line.decode(errors="replace")
        try:
            line: object = json.loads(decoded_line)
        except Exception:
            line = decoded_line

        if isinstance(line, dict) and line.get("event") == "job_phase":
            phase = line.get("phase")
            progress = line.get("progress")
            if phase in _PHASES and isinstance(progress, int | float):
                normalized_progress = min(1.0, max(0.0, float(progress)))
                self._write_transition(
                    job_id,
                    status="running",
                    phase=str(phase),
                    progress=normalized_progress,
                )
                log_event(
                    logging.INFO,
                    "job_phase",
                    id=job_id,
                    phase=phase,
                    progress=normalized_progress,
                )
                return
        if isinstance(line, dict) and line.get("event") == "job_error":
            error = line.get("error")
            if isinstance(error, str) and error:
                worker_error["error"] = error
        log_event(
            logging.INFO,
            "job_worker_log",
            id=job_id,
            line=line,
        )


job_runner = JobRunner()

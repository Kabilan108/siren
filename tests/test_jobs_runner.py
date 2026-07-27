import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from siren.jobs.runner import (
    JobQueueFullError,
    JobRunner,
    JobUploadTooLargeError,
)


def upload(
    filename: str = "meeting.mp3",
    content: bytes = b"audio",
) -> UploadFile:
    upload_file = tempfile.SpooledTemporaryFile()
    upload_file.write(content)
    upload_file.seek(0)
    return UploadFile(
        file=cast(BinaryIO, upload_file),
        filename=filename,
    )


def worker_command(script: str) -> Callable[[Path], tuple[str, ...]]:
    def command(job_dir: Path) -> tuple[str, ...]:
        return sys.executable, "-c", script, str(job_dir)

    return command


def result_payload() -> dict[str, object]:
    return {
        "text": "hello there",
        "language": "en",
        "duration": 1.5,
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "diarization_model": "test/sortformer",
        "speakers": ["SPEAKER_00"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.5,
                "speaker": "SPEAKER_00",
                "text": "hello there",
            }
        ],
    }


async def wait_for_status(
    runner: JobRunner,
    job_id: str,
    expected: str,
    *,
    timeout: float = 5.0,
) -> dict[str, object]:
    async with asyncio.timeout(timeout):
        while True:
            state = await runner.status(job_id)
            if state is not None and state["status"] == expected:
                return state
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_runner_transitions_queued_running_completed(
    tmp_path: Path,
) -> None:
    payload = result_payload()
    script = "\n".join(
        [
            "import json, pathlib, sys, time",
            "job_dir = pathlib.Path(sys.argv[1])",
            "sys.stderr.write(json.dumps({'event':'job_phase','phase':'transcribing','progress':0.5}) + '\\n')",
            "sys.stderr.flush()",
            "time.sleep(0.1)",
            f"(job_dir / 'result.json').write_text({json.dumps(payload)!r})",
        ]
    )
    runner = JobRunner(
        spool_dir=tmp_path,
        worker_command=worker_command(script),
    )
    try:
        job_id, position = await runner.enqueue(
            upload(),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language="en",
        )
        assert position == 1
        running = await wait_for_status(runner, job_id, "running")
        assert running["phase"] in {None, "transcribing"}

        completed = await wait_for_status(runner, job_id, "completed")
        assert completed["progress"] == 1.0
        assert runner.result(job_id) == payload
        assert (tmp_path / job_id / "input.mp3").read_bytes() == b"audio"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_upload_limit_cleans_partial_job_directory(tmp_path: Path) -> None:
    runner = JobRunner(
        spool_dir=tmp_path,
        max_upload_bytes=4,
    )
    try:
        with pytest.raises(
            JobUploadTooLargeError,
            match="maximum size of 4 bytes",
        ):
            await runner.enqueue(
                upload(content=b"12345"),
                model="nvidia/parakeet-tdt-0.6b-v2",
                language=None,
            )

        assert list(tmp_path.iterdir()) == []
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_job_directory_mode_is_private(tmp_path: Path) -> None:
    runner = JobRunner(
        spool_dir=tmp_path,
        worker_command=worker_command("import time; time.sleep(10)"),
    )
    job_id, _position = await runner.enqueue(
        upload(),
        model="nvidia/parakeet-tdt-0.6b-v2",
        language=None,
    )
    try:
        mode = (tmp_path / job_id).stat().st_mode
        assert stat.S_IMODE(mode) == 0o700
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_runner_persists_worker_failure(tmp_path: Path) -> None:
    script = "\n".join(
        [
            "import json, sys",
            "sys.stderr.write(json.dumps({'event':'job_phase','phase':'diarizing','progress':0.75}) + '\\n')",
            "sys.stderr.write(json.dumps({'event':'job_error','error':'model exploded'}) + '\\n')",
            "raise SystemExit(7)",
        ]
    )
    runner = JobRunner(
        spool_dir=tmp_path,
        worker_command=worker_command(script),
    )
    try:
        job_id, _position = await runner.enqueue(
            upload(),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        failed = await wait_for_status(runner, job_id, "failed")

        assert failed["phase"] is None
        assert failed["progress"] == 0.75
        assert failed["error"] == "model exploded"
        assert not (tmp_path / job_id / "result.json").exists()
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_queue_limit_rejects_excess_queued_job(tmp_path: Path) -> None:
    runner = JobRunner(
        spool_dir=tmp_path,
        max_concurrent_jobs=1,
        max_queued_jobs=1,
        worker_command=worker_command("import time; time.sleep(10)"),
    )
    try:
        first_id, _position = await runner.enqueue(
            upload("first.wav"),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        await wait_for_status(runner, first_id, "running")
        await runner.enqueue(
            upload("second.wav"),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )

        with pytest.raises(
            JobQueueFullError,
            match="maximum 1 queued jobs",
        ):
            await runner.enqueue(
                upload("third.wav"),
                model="nvidia/parakeet-tdt-0.6b-v2",
                language=None,
            )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_runner_timeout_marks_job_failed(tmp_path: Path) -> None:
    runner = JobRunner(
        spool_dir=tmp_path,
        timeout_seconds=0.05,
        worker_command=worker_command("import time; time.sleep(10)"),
    )
    try:
        job_id, _position = await runner.enqueue(
            upload(),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        failed = await wait_for_status(runner, job_id, "failed")

        assert failed["error"] == "Job timed out after 0.05 seconds"
        assert runner.active_workers == {}
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_shutdown_during_spawn_persists_failed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_started = asyncio.Event()

    async def delayed_spawn(
        *args: object,
        **kwargs: object,
    ) -> asyncio.subprocess.Process:
        del args, kwargs
        spawn_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    runner = JobRunner(spool_dir=tmp_path)
    job_id, _position = await runner.enqueue(
        upload(),
        model="nvidia/parakeet-tdt-0.6b-v2",
        language=None,
    )
    await asyncio.wait_for(spawn_started.wait(), timeout=2.0)

    await runner.stop()

    state = await runner.status(job_id)
    assert state is not None
    assert state["status"] == "failed"
    assert state["phase"] is None
    assert state["error"] == "server_shutdown"
    assert runner.active_workers == {}


def test_startup_recovery_marks_running_and_queued_failed(tmp_path: Path) -> None:
    for index, status in enumerate(("running", "queued")):
        job_dir = tmp_path / f"job_{index:032x}"
        job_dir.mkdir()
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "id": job_dir.name,
                    "status": status,
                    "phase": "transcribing" if status == "running" else None,
                    "progress": 0.4 if status == "running" else 0.0,
                }
            )
        )

    runner = JobRunner(spool_dir=tmp_path)
    assert runner.recover_interrupted_jobs() == 2

    running = json.loads((tmp_path / f"job_{0:032x}" / "job.json").read_text())
    queued = json.loads((tmp_path / f"job_{1:032x}" / "job.json").read_text())
    assert running["status"] == "failed"
    assert running["error"] == "orphaned by server restart"
    assert running["phase"] is None
    assert queued["status"] == "failed"
    assert queued["error"] == "server restarted before start"


def test_startup_recovery_accepts_valid_completed_result(tmp_path: Path) -> None:
    job_dir = tmp_path / f"job_{3:032x}"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": job_dir.name,
                "status": "running",
                "phase": "aligning",
                "progress": 0.95,
            }
        )
    )
    payload = result_payload()
    (job_dir / "result.json").write_text(json.dumps(payload))

    runner = JobRunner(spool_dir=tmp_path)
    assert runner.recover_interrupted_jobs() == 1

    state = json.loads((job_dir / "job.json").read_text())
    assert state["status"] == "completed"
    assert state["phase"] is None
    assert state["progress"] == 1.0
    assert "error" not in state
    assert runner.result(job_dir.name) == payload


def test_sweep_deletes_only_expired_job_directories(tmp_path: Path) -> None:
    old_dir = tmp_path / f"job_{1:032x}"
    fresh_dir = tmp_path / f"job_{2:032x}"
    unrelated_dir = tmp_path / "other"
    for directory in (old_dir, fresh_dir, unrelated_dir):
        directory.mkdir()
    old_state = old_dir / "job.json"
    fresh_state = fresh_dir / "job.json"
    old_state.write_text("{}")
    fresh_state.write_text("{}")
    (unrelated_dir / "job.json").write_text("{}")
    now = time.time()
    os.utime(old_state, (now - 8 * 86400, now - 8 * 86400))

    runner = JobRunner(spool_dir=tmp_path, retention_days=7.0)
    assert runner.sweep(now=now) == 1
    assert not old_dir.exists()
    assert fresh_dir.exists()
    assert unrelated_dir.exists()


def test_sweep_skips_symlinks_and_foreign_directories(
    tmp_path: Path,
) -> None:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    marker = target_dir / "keep"
    marker.write_text("safe")
    symlink_dir = spool_dir / f"job_{4:032x}"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)
    foreign_dir = spool_dir / "job_not-a-valid-id"
    foreign_dir.mkdir()
    now = time.time()
    old = now - 8 * 86400
    os.utime(target_dir, (old, old))
    os.utime(foreign_dir, (old, old))

    runner = JobRunner(spool_dir=spool_dir, retention_days=7.0)
    assert runner.sweep(now=now) == 0

    assert symlink_dir.is_symlink()
    assert marker.read_text() == "safe"
    assert foreign_dir.exists()


def test_sweep_deletes_expired_job_without_state_file(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / f"job_{5:032x}"
    job_dir.mkdir()
    (job_dir / ".input.wav.upload").write_bytes(b"partial")
    now = time.time()
    old = now - 8 * 86400
    os.utime(job_dir, (old, old))

    runner = JobRunner(spool_dir=tmp_path, retention_days=7.0)
    assert runner.sweep(now=now) == 1
    assert not job_dir.exists()


def test_sweep_continues_after_one_directory_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_dir = tmp_path / f"job_{7:032x}"
    deleted_dir = tmp_path / f"job_{8:032x}"
    now = time.time()
    old = now - 8 * 86400
    for job_dir in (failed_dir, deleted_dir):
        job_dir.mkdir()
        state_path = job_dir / "job.json"
        state_path.write_text("{}")
        os.utime(state_path, (old, old))

    real_rmtree = shutil.rmtree

    def flaky_rmtree(path: str | Path) -> None:
        if Path(path) == failed_dir:
            raise OSError("busy")
        real_rmtree(path)

    monkeypatch.setattr("siren.jobs.runner.shutil.rmtree", flaky_rmtree)
    runner = JobRunner(spool_dir=tmp_path, retention_days=7.0)

    assert runner.sweep(now=now) == 1
    assert failed_dir.exists()
    assert not deleted_dir.exists()


@pytest.mark.asyncio
async def test_stderr_poison_lines_do_not_stop_progress_relay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = f"job_{6:032x}"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": job_id,
                "status": "running",
                "phase": None,
                "progress": 0.0,
            }
        )
    )
    relay_logs = MagicMock()
    monkeypatch.setattr("siren.jobs.runner.log_event", relay_logs)
    stderr = asyncio.StreamReader()
    huge_integer = b'{"value":' + (b"9" * 5000) + b"}\n"
    deeply_nested = (b"[" * 10_000) + b"0" + (b"]" * 10_000) + b"\n"
    progress = (
        b'{"event":"job_phase","phase":"transcribing","progress":0.5}\n'
    )
    stderr.feed_data(huge_integer + deeply_nested + progress)
    stderr.feed_eof()
    runner = JobRunner(spool_dir=tmp_path)

    await runner._relay_stderr(stderr, job_id=job_id, worker_error={})

    state = json.loads((job_dir / "job.json").read_text())
    assert state["phase"] == "transcribing"
    assert state["progress"] == 0.5
    raw_lines = [
        call.kwargs["line"]
        for call in relay_logs.call_args_list
        if call.args[1] == "job_worker_log"
    ]
    assert raw_lines[:2] == [
        huge_integer.rstrip().decode(),
        deeply_nested.rstrip().decode(),
    ]


@pytest.mark.asyncio
async def test_fifo_positions_and_concurrency_one_serialization(
    tmp_path: Path,
) -> None:
    release_path = tmp_path / "release"
    order_path = tmp_path / "order.log"
    payload = result_payload()
    script = "\n".join(
        [
            "import json, pathlib, sys, time",
            "job_dir = pathlib.Path(sys.argv[1])",
            f"release = pathlib.Path({str(release_path)!r})",
            f"order = pathlib.Path({str(order_path)!r})",
            "with order.open('a') as output: output.write(f'start:{job_dir.name}\\n')",
            "while not release.exists(): time.sleep(0.01)",
            "time.sleep(0.05)",
            f"(job_dir / 'result.json').write_text({json.dumps(payload)!r})",
            "with order.open('a') as output: output.write(f'end:{job_dir.name}\\n')",
        ]
    )
    runner = JobRunner(
        spool_dir=tmp_path / "spool",
        max_concurrent_jobs=1,
        worker_command=worker_command(script),
    )
    try:
        first_id, _ = await runner.enqueue(
            upload("one.wav"),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        await wait_for_status(runner, first_id, "running")
        second_id, second_position = await runner.enqueue(
            upload("two.wav"),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        third_id, third_position = await runner.enqueue(
            upload("three.wav"),
            model="nvidia/parakeet-tdt-0.6b-v2",
            language=None,
        )
        assert second_position == 1
        assert third_position == 2
        assert (await runner.status(second_id))["position"] == 1  # type: ignore[index]
        assert (await runner.status(third_id))["position"] == 2  # type: ignore[index]

        release_path.touch()
        for job_id in (first_id, second_id, third_id):
            await wait_for_status(runner, job_id, "completed")

        assert order_path.read_text().splitlines() == [
            f"start:{first_id}",
            f"end:{first_id}",
            f"start:{second_id}",
            f"end:{second_id}",
            f"start:{third_id}",
            f"end:{third_id}",
        ]
    finally:
        release_path.touch()
        await runner.stop()

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from siren.diarization import supervisor
from siren.server import app, lifespan


@pytest.fixture(autouse=True)
def fresh_diarize_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "diarize_lock", asyncio.Lock())
    monkeypatch.setattr(supervisor, "active_worker_process", None)


def fake_command(script: str) -> object:
    def command(audio_path: str, output_path: str) -> tuple[str, ...]:
        return sys.executable, "-c", script, audio_path, output_path

    return command


@pytest.mark.asyncio
async def test_run_diarization_success_and_stderr_relay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "duration": 2.5,
        "model": "test/model",
        "speakers": ["SPEAKER_00"],
        "turns": [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}],
    }
    script = (
        "import pathlib,sys;"
        f"pathlib.Path(sys.argv[2]).write_text({json.dumps(payload)!r});"
        "sys.stderr.write('{\"event\":\"fake_log\",\"value\":1}\\nraw line\\n')"
    )
    log_event = MagicMock()
    monkeypatch.setattr(supervisor, "_worker_command", fake_command(script))
    monkeypatch.setattr(supervisor, "log_event", log_event)

    result = await supervisor.run_diarization(
        str(tmp_path / "audio.wav"),
        request_id="request-1",
    )

    assert result == payload
    relayed_lines = [
        call.kwargs["line"]
        for call in log_event.call_args_list
        if call.args[1] == "diarize_worker_log"
    ]
    assert relayed_lines == [{"event": "fake_log", "value": 1}, "raw line"]
    complete_call = next(
        call
        for call in log_event.call_args_list
        if call.args[1] == "diarize_complete"
    )
    assert complete_call.kwargs["n_turns"] == 1
    assert complete_call.kwargs["n_speakers"] == 1


@pytest.mark.asyncio
async def test_stderr_relay_truncates_long_line_without_hanging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "duration": 1.0,
        "model": "test/model",
        "speakers": [],
        "turns": [],
    }
    script = "\n".join(
        [
            "import os, pathlib, sys",
            "os.write(2, b'x' * 200_000)",
            f"pathlib.Path(sys.argv[2]).write_text({json.dumps(payload)!r})",
        ]
    )
    log_event = MagicMock()
    monkeypatch.setattr(supervisor, "_worker_command", fake_command(script))
    monkeypatch.setattr(supervisor, "log_event", log_event)

    result = await supervisor.run_diarization(
        str(tmp_path / "audio.wav"),
        timeout_seconds=2.0,
        request_id="request-long-stderr",
    )

    assert result == payload
    relayed_lines = [
        call.kwargs["line"]
        for call in log_event.call_args_list
        if call.args[1] == "diarize_worker_log"
    ]
    assert len(relayed_lines) == 1
    assert isinstance(relayed_lines[0], str)
    assert relayed_lines[0].endswith("...truncated")
    assert len(relayed_lines[0]) == supervisor._MAX_STDERR_LINE_BYTES + len(
        "...truncated"
    )


@pytest.mark.asyncio
async def test_stderr_relay_failure_does_not_mask_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = "import time;time.sleep(10)"
    log_event = MagicMock()
    monkeypatch.setattr(supervisor, "_worker_command", fake_command(script))
    monkeypatch.setattr(
        supervisor,
        "_relay_stderr",
        AsyncMock(side_effect=ValueError("relay failed")),
    )
    monkeypatch.setattr(supervisor, "_TERMINATION_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(supervisor, "log_event", log_event)

    with pytest.raises(HTTPException) as exc_info:
        await supervisor.run_diarization(
            str(tmp_path / "audio.wav"),
            timeout_seconds=0.05,
            request_id="request-relay-failure",
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Diarization timed out"
    assert any(
        call.args[1] == "diarize_worker_relay_failed"
        for call in log_event.call_args_list
    )


@pytest.mark.asyncio
async def test_run_diarization_nonzero_exit_is_generic_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = "import sys;sys.stderr.write('internal /secret/path\\n');raise SystemExit(7)"
    monkeypatch.setattr(supervisor, "_worker_command", fake_command(script))

    with pytest.raises(HTTPException) as exc_info:
        await supervisor.run_diarization(
            str(tmp_path / "audio.wav"),
            request_id="request-2",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Diarization worker failed"
    assert "/secret/path" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_diarization_timeout_terminates_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_info_path = tmp_path / "process.json"
    term_marker_path = tmp_path / "term-received"
    script = "\n".join(
        [
            "import json, os, pathlib, signal, sys, time",
            f"info_path = pathlib.Path({str(process_info_path)!r})",
            f"term_path = pathlib.Path({str(term_marker_path)!r})",
            "info_path.write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))",
            "def stop(_signum, _frame):",
            "    term_path.write_text('yes')",
            "    raise SystemExit(0)",
            "signal.signal(signal.SIGTERM, stop)",
            "while True:",
            "    time.sleep(1)",
        ]
    )
    monkeypatch.setattr(supervisor, "_worker_command", fake_command(script))
    monkeypatch.setattr(supervisor, "_TERMINATION_GRACE_SECONDS", 1.0)

    with pytest.raises(HTTPException) as exc_info:
        await supervisor.run_diarization(
            str(tmp_path / "audio.wav"),
            timeout_seconds=0.5,
            request_id="request-3",
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Diarization timed out"
    assert term_marker_path.read_text() == "yes"
    process_group_id = json.loads(process_info_path.read_text())["pgid"]
    for _ in range(50):
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail(f"process group {process_group_id} is still alive")


@pytest.mark.asyncio
async def test_lifespan_shutdown_terminates_registered_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 12345
    process.returncode = None
    terminate_process_group = AsyncMock()
    ensure_model_loaded = AsyncMock()
    unload_model = MagicMock()
    start_job_runner = AsyncMock()
    stop_job_runner = AsyncMock()
    monkeypatch.setattr(supervisor, "active_worker_process", process)
    monkeypatch.setattr(supervisor, "_process_group_exists", lambda _pid: True)
    monkeypatch.setattr(
        supervisor,
        "_terminate_process_group",
        terminate_process_group,
    )
    monkeypatch.setattr(
        "siren.server.config.load_model_name",
        lambda: "test/model",
    )
    monkeypatch.setattr(
        "siren.server.models.ensure_model_loaded",
        ensure_model_loaded,
    )
    monkeypatch.setattr("siren.server.models.unload_model", unload_model)
    monkeypatch.setattr("siren.server.job_runner.start", start_job_runner)
    monkeypatch.setattr("siren.server.job_runner.stop", stop_job_runner)

    async with lifespan(app):
        pass

    terminate_process_group.assert_awaited_once_with(process)
    assert supervisor.active_worker_process is None
    start_job_runner.assert_awaited_once()
    stop_job_runner.assert_awaited_once()
    unload_model.assert_called_once()

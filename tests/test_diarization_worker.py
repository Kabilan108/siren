import json
import subprocess
import sys
from pathlib import Path
from typing import TextIO
from unittest.mock import ANY, MagicMock

import pytest

from siren.diarization import worker


class FakeSortformerModel:
    raw_turns = [
        "5.0 7.5 speaker_1",
        "0.25 2.0 speaker_0",
        "2.0 4.5 speaker_12",
    ]

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FakeSortformerModel":
        assert model_name == "test/diarizer"
        return cls()

    def cuda(self) -> "FakeSortformerModel":
        return self

    def eval(self) -> "FakeSortformerModel":
        return self

    def diarize(self, *, audio: list[str], batch_size: int) -> list[list[str]]:
        assert len(audio) == 1
        assert batch_size == 1
        return [self.raw_turns]


@pytest.fixture
def mocked_worker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    set_memory_fraction = MagicMock()
    monkeypatch.setenv("SIREN_DIARIZE_MODEL", "test/diarizer")
    monkeypatch.setenv("SIREN_DIARIZE_MEMORY_FRACTION", "0.25")
    monkeypatch.setattr(worker, "_load_model_class", lambda: FakeSortformerModel)
    monkeypatch.setattr(worker.torch.cuda, "init", MagicMock())
    monkeypatch.setattr(
        worker.torch.cuda,
        "set_per_process_memory_fraction",
        set_memory_fraction,
    )
    monkeypatch.setattr(
        worker,
        "get_wav_info",
        lambda _path: {"audio_duration_sec": 9.75},
    )
    return set_memory_fraction


def test_normalize_speaker_label() -> None:
    assert worker.normalize_speaker_label("speaker_0") == "SPEAKER_00"
    assert worker.normalize_speaker_label("SPEAKER_12") == "SPEAKER_12"


def test_parse_turns_normalizes_and_sorts() -> None:
    turns = worker.parse_turns(
        [
            "8.0 9.0 speaker_2",
            "0.5 1.0 speaker_0",
            "3.0 4.0 speaker_1",
        ]
    )

    assert turns == [
        {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 8.0, "end": 9.0, "speaker": "SPEAKER_02"},
    ]


def test_diarize_writes_sorted_atomic_result(
    tmp_path: Path,
    mocked_worker: MagicMock,
) -> None:
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "result.json"

    result = worker.diarize(audio_path, output_path)

    assert json.loads(output_path.read_text()) == result
    assert result == {
        "duration": 9.75,
        "model": "test/diarizer",
        "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_12"],
        "turns": [
            {"start": 0.25, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.5, "speaker": "SPEAKER_12"},
            {"start": 5.0, "end": 7.5, "speaker": "SPEAKER_01"},
        ],
    }
    mocked_worker.assert_called_once_with(0.25)
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_worker: MagicMock,
) -> None:
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "result.json"

    def fail_during_dump(
        payload: object,
        output_file: TextIO,
        **kwargs: object,
    ) -> None:
        del payload, kwargs
        output_file.write("{")
        raise OSError("disk full")

    monkeypatch.setattr(worker.json, "dump", fail_during_dump)

    with pytest.raises(OSError, match="disk full"):
        worker.diarize(audio_path, output_path)

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []
    mocked_worker.assert_called_once_with(0.25)


def test_parent_watchdog_exits_reparented_worker() -> None:
    script = "\n".join(
        [
            "import os, time",
            "from siren.diarization.worker import configure_worker_logging, _start_parent_watchdog",
            "configure_worker_logging()",
            "_start_parent_watchdog(os.getppid() + 1, poll_seconds=0.01)",
            "time.sleep(5)",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 1
    stderr_lines = [
        json.loads(line) for line in result.stderr.splitlines() if line.strip()
    ]
    assert stderr_lines == [
        {
            "event": "diarize_worker_parent_changed",
            "parent_pid": ANY,
            "current_parent_pid": ANY,
        }
    ]

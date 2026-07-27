import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

from siren.jobs import worker
from siren.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


def test_silence_aligned_cut_points_prefers_nearest_midpoint() -> None:
    cuts = worker.silence_aligned_cut_points(
        [
            (287.0, 289.0),
            (302.0, 306.0),
            (596.0, 598.0),
        ],
        910.0,
    )

    assert cuts == [304.0, 597.0, 900.0]


def test_silence_aligned_cut_points_uses_hard_cut_outside_window() -> None:
    cuts = worker.silence_aligned_cut_points(
        [(270.0, 272.0), (321.0, 323.0)],
        650.0,
    )

    assert cuts == [300.0, 600.0]


def test_silence_aligned_cut_points_handles_exact_boundaries_and_short_audio() -> None:
    assert worker.silence_aligned_cut_points([(299.0, 301.0)], 300.0) == []
    assert worker.silence_aligned_cut_points([], 299.999) == []
    assert worker.silence_aligned_cut_points([(280.0, 280.0)], 601.0) == [
        280.0,
        600.0,
    ]
    assert worker.silence_aligned_cut_points([(590.0, 620.0)], 610.0) == [
        300.0,
        600.0,
    ]


@pytest.mark.parametrize(
    ("duration", "chunk_seconds", "window_seconds"),
    [
        (math.inf, 300.0, 20.0),
        (1.0, 0.0, 20.0),
        (1.0, 300.0, -1.0),
    ],
)
def test_silence_aligned_cut_points_rejects_invalid_bounds(
    duration: float,
    chunk_seconds: float,
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        worker.silence_aligned_cut_points(
            [],
            duration,
            chunk_seconds=chunk_seconds,
            window_seconds=window_seconds,
        )


def test_all_ffmpeg_inputs_use_restricted_protocol_whitelist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def record_command(
        command: list[str] | tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(worker, "_run_command", record_command)
    input_path = tmp_path / "input.m3u8"
    wav_path = tmp_path / "audio.wav"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    worker._convert_to_wav(input_path, wav_path)
    worker._detect_silences(wav_path, 2.0)
    worker._split_wav(wav_path, scratch_dir, [1.0], 2.0)

    assert len(commands) == 4
    for command in commands:
        input_index = command.index("-i")
        whitelist_index = command.index("-protocol_whitelist")
        assert whitelist_index < input_index
        assert command[whitelist_index + 1] == "file,pipe,crypto"


@pytest.mark.asyncio
async def test_transcription_offsets_words_by_actual_chunk_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeBackend:
        call_index = 0

        async def transcribe(
            self,
            audio_path: str,
            *,
            language: str | None,
            word_timestamps: bool,
        ) -> TranscriptionResult:
            del audio_path, language
            assert word_timestamps is True
            index = self.call_index
            self.call_index += 1
            return TranscriptionResult(
                text=f"chunk {index}",
                language="en",
                duration=1.0,
                segments=[
                    TranscriptionSegment(
                        id=0,
                        start=0.25,
                        end=0.75,
                        text=f"word{index}",
                        words=[
                            TranscriptionWord(
                                start=0.25,
                                end=0.75,
                                word=f"word{index}",
                            )
                        ],
                    )
                ],
            )

    empty_cache = MagicMock()
    monkeypatch.setattr(worker, "load_backend", lambda _model: FakeBackend())
    monkeypatch.setattr(worker.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(worker, "_emit_phase", MagicMock())

    text, language, words = await worker._transcribe_chunks(
        [
            (tmp_path / "chunk-0.wav", 0.0),
            (tmp_path / "chunk-1.wav", 287.5),
        ],
        model_name="test/parakeet",
        language=None,
    )

    assert text == "chunk 0 chunk 1"
    assert language == "en"
    assert [(word.start, word.end, word.word) for word in words] == [
        (0.25, 0.75, "word0"),
        (287.75, 288.25, "word1"),
    ]
    empty_cache.assert_called_once()


def test_parent_watchdog_exits_reparented_job_worker() -> None:
    script = "\n".join(
        [
            "import os, time",
            "from siren.jobs.worker import configure_worker_logging, _start_parent_watchdog",
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
            "event": "job_worker_parent_changed",
            "parent_pid": ANY,
            "current_parent_pid": ANY,
        }
    ]

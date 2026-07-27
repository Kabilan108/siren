import logging
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from siren.audio import get_wav_info
from siren.diarization import get_diarization_model, get_memory_fraction
from siren.logging_utils import log_event
from siren.io import atomic_write_json

_SPEAKER_LABEL = re.compile(r"^speaker_(\d+)$", re.IGNORECASE)
_PARENT_POLL_SECONDS = 5.0


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
                    "diarize_worker_parent_changed",
                    parent_pid=parent_pid,
                    current_parent_pid=current_parent_pid,
                )
                os._exit(1)

    watchdog = threading.Thread(
        target=watch_parent,
        name="diarize-parent-watchdog",
        daemon=True,
    )
    watchdog.start()
    return watchdog


def normalize_speaker_label(label: str) -> str:
    match = _SPEAKER_LABEL.fullmatch(label.strip())
    if match is None:
        raise ValueError(f"Invalid speaker label: {label!r}")
    return f"SPEAKER_{int(match.group(1)):02d}"


def parse_turns(raw_turns: Sequence[str]) -> list[dict[str, float | str]]:
    turns: list[dict[str, float | str]] = []
    for raw_turn in raw_turns:
        parts = raw_turn.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid diarization turn: {raw_turn!r}")
        start, end, speaker = parts
        turns.append(
            {
                "start": float(start),
                "end": float(end),
                "speaker": normalize_speaker_label(speaker),
            }
        )
    return sorted(
        turns,
        key=lambda turn: (
            float(turn["start"]),
            float(turn["end"]),
            str(turn["speaker"]),
        ),
    )


def _load_model_class() -> type[Any]:
    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel


def _initialize_cuda() -> None:
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(get_memory_fraction())


def diarize(audio_path: Path, output_path: Path) -> dict[str, object]:
    model_name = get_diarization_model()
    _initialize_cuda()
    model = _load_model_class().from_pretrained(model_name).cuda().eval()
    with torch.inference_mode():
        outputs = model.diarize(audio=[str(audio_path)], batch_size=1)

    turns = parse_turns(outputs[0])
    audio_info = get_wav_info(str(audio_path))
    payload: dict[str, object] = {
        "duration": float(audio_info.get("audio_duration_sec", 0.0)),
        "model": model_name,
        "speakers": sorted({str(turn["speaker"]) for turn in turns}),
        "turns": turns,
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    configure_worker_logging()
    _start_parent_watchdog(os.getppid())
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) != 2:
            raise ValueError("Expected arguments: <audio.wav> <out.json>")
        diarize(Path(arguments[0]), Path(arguments[1]))
    except BaseException as exc:
        log_event(
            logging.ERROR,
            "diarize_worker_failed",
            error=str(exc),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

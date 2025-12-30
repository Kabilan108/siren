#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

DEFAULT_RECORDINGS_DIR = Path("~/.local/share/dictator/recordings").expanduser()


def duration_seconds(path: Path, ffprobe_path: str | None) -> float | None:
    ext = path.suffix.lower()
    if ext == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate:
                    return frames / float(rate)
        except (wave.Error, EOFError):
            return None

    if ffprobe_path:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                try:
                    return float(value)
                except ValueError:
                    return None

    return None


def main() -> int:
    recordings_dir = (
        Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_RECORDINGS_DIR
    )
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}", file=sys.stderr)
        return 1

    ffprobe_path = shutil.which("ffprobe")
    results: list[dict[str, float | str]] = []
    for file_path in sorted(recordings_dir.rglob("*")):
        if not file_path.is_file():
            continue
        seconds = duration_seconds(file_path, ffprobe_path)
        if seconds is None:
            print(f"Skipping unreadable file: {file_path}", file=sys.stderr)
            continue
        results.append({"file": str(file_path), "duration": round(seconds / 60.0, 4)})

    json.dump(results, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

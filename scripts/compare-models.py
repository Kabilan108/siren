#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///

import json
import mimetypes
import os
import re
import statistics as stats
import time
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

import typer

app = typer.Typer(add_completion=False)

AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aifc",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


@dataclass
class TranscriptionResult:
    text: str
    words: List[str]
    elapsed_ms: int


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def is_parakeet_model(model_name: str) -> bool:
    return model_name.startswith("nvidia/parakeet")


def encode_multipart(
    fields: Dict[str, str],
    files: Dict[str, Tuple[str, bytes, str]],
) -> Tuple[bytes, str]:
    boundary = f"----siren-{uuid.uuid4().hex}"
    body = bytearray()

    def add_str(value: str) -> None:
        body.extend(value.encode("utf-8"))

    for name, value in fields.items():
        add_str(f"--{boundary}\r\n")
        add_str(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        add_str(str(value))
        add_str("\r\n")

    for name, (filename, content, content_type) in files.items():
        add_str(f"--{boundary}\r\n")
        add_str(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        )
        add_str(f"Content-Type: {content_type}\r\n\r\n")
        body.extend(content)
        add_str("\r\n")

    add_str(f"--{boundary}--\r\n")
    return bytes(body), boundary


def post_transcription(
    base_url: str,
    token: str,
    model: str,
    file_path: Path,
    language: str | None,
    timeout: float,
) -> TranscriptionResult:
    endpoint = base_url.rstrip("/") + "/v1/audio/transcriptions"
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with file_path.open("rb") as handle:
        content = handle.read()

    fields = {"model": model}
    if language and not is_parakeet_model(model):
        fields["language"] = language

    body, boundary = encode_multipart(
        fields,
        {"file": (filename, content, content_type)},
    )

    req = request.Request(endpoint, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    data = json.loads(payload)
    text = data.get("text")
    if text is None:
        raise RuntimeError(f"Missing 'text' in response: {data}")
    return TranscriptionResult(
        text=text, words=tokenize_words(text), elapsed_ms=elapsed_ms
    )


def sanitize_label(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


@app.command()
def main(
    models: List[str] = typer.Option(
        ...,
        "-m",
        "--model",
        help="Model name (repeatable). First model is the baseline.",
    ),
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        help="Base server URL.",
    ),
    input_dir: Path = typer.Option(
        Path("data/recordings"),
        "--input-dir",
        help="Directory containing audio files.",
    ),
    language: str | None = typer.Option(
        "en",
        "--language",
        help="Language for Whisper models (ignored for Parakeet).",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Bearer token. Defaults to SIREN_API_KEY or dev_token.",
    ),
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        help="Request timeout in seconds.",
    ),
) -> None:
    if len(models) < 2:
        raise typer.BadParameter("Provide at least two -m/--model values.")

    if not input_dir.exists():
        raise typer.BadParameter(f"Input dir not found: {input_dir}")

    token_value = token or os.environ.get("SIREN_API_KEY", "dev_token")

    files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda p: p.name,
    )

    if not files:
        raise typer.BadParameter(f"No audio files found in {input_dir}")

    results: Dict[str, Dict[str, TranscriptionResult]] = {}

    for file_path in files:
        file_key = file_path.stem
        results[file_key] = {}
        for model in models:
            result = post_transcription(
                base_url=url,
                token=token_value,
                model=model,
                file_path=file_path,
                language=language,
                timeout=timeout,
            )
            results[file_key][model] = result

    baseline = models[0]
    baseline_label = sanitize_label(baseline)

    header: List[str] = ["file", f"words_{baseline_label}"]
    for model in models[1:]:
        label = sanitize_label(model)
        header.append(f"words_{label}")
        header.append("word_sim" if len(models) == 2 else f"word_sim_{label}")
    header.append(f"ms_{baseline_label}")
    for model in models[1:]:
        label = sanitize_label(model)
        header.append(f"ms_{label}")

    typer.echo("\t".join(header))

    word_sims: Dict[str, List[float]] = {m: [] for m in models[1:]}
    word_ratios: Dict[str, List[float]] = {m: [] for m in models[1:]}
    timings: Dict[str, List[int]] = {m: [] for m in models}
    empty_counts: Dict[str, int] = {m: 0 for m in models}

    for file_key in sorted(results.keys()):
        row: List[str] = [file_key]
        base_result = results[file_key][baseline]
        row.append(str(len(base_result.words)))
        timings[baseline].append(base_result.elapsed_ms)
        if not base_result.text.strip():
            empty_counts[baseline] += 1

        for model in models[1:]:
            other_result = results[file_key][model]
            row.append(str(len(other_result.words)))
            sim = SequenceMatcher(
                None,
                base_result.words,
                other_result.words,
            ).ratio()
            row.append(f"{sim:.3f}")
            word_sims[model].append(sim)
            if base_result.words:
                ratio = len(other_result.words) / len(base_result.words)
                word_ratios[model].append(ratio)
            timings[model].append(other_result.elapsed_ms)
            if not other_result.text.strip():
                empty_counts[model] += 1

        row.append(str(base_result.elapsed_ms))
        for model in models[1:]:
            row.append(str(results[file_key][model].elapsed_ms))

        typer.echo("\t".join(row))

    typer.echo("")
    typer.echo("STATS")
    typer.echo(f"files\t{len(results)}")

    for model in models:
        values = timings[model]
        typer.echo(
            "\t".join(
                [
                    f"timing_{sanitize_label(model)}",
                    f"avg_ms={round(stats.mean(values))}",
                    f"median_ms={round(stats.median(values))}",
                    f"min_ms={min(values)}",
                    f"max_ms={max(values)}",
                ]
            )
        )

    for model in models[1:]:
        sim_values = word_sims[model]
        ratio_values = word_ratios[model]
        typer.echo(
            "\t".join(
                [
                    f"word_sim_{sanitize_label(model)}",
                    f"avg={stats.mean(sim_values):.3f}",
                    f"min={min(sim_values):.3f}",
                    f"max={max(sim_values):.3f}",
                ]
            )
        )
        if ratio_values:
            typer.echo(
                "\t".join(
                    [
                        f"word_ratio_{sanitize_label(model)}",
                        f"avg={stats.mean(ratio_values):.3f}",
                        f"min={min(ratio_values):.3f}",
                        f"max={max(ratio_values):.3f}",
                    ]
                )
            )

    for model in models:
        typer.echo(
            "\t".join(
                [
                    f"empty_{sanitize_label(model)}",
                    str(empty_counts[model]),
                ]
            )
        )


if __name__ == "__main__":
    app()

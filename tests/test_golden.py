import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from jiwer import wer

REPO_ROOT = Path(__file__).parents[1]
GOLDEN_ROOT = REPO_ROOT / "tests" / "goldens"
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER_MODEL = "tiny"


@dataclass(frozen=True)
class GoldenCase:
    golden_path: Path
    audio_path: Path
    model: str
    response_format: str

    @property
    def id(self) -> str:
        return f"{self.golden_path.parent.name}/{self.golden_path.name}"


def golden_cases(directory: str, model: str) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for golden_path in sorted((GOLDEN_ROOT / directory).glob("*.json")):
        if golden_path.name.startswith("."):
            continue
        clip, response_format = golden_path.name.removesuffix(".json").rsplit(".", 1)
        audio_path = (
            REPO_ROOT / "test.wav"
            if clip == "test"
            else REPO_ROOT / "data" / "recordings" / f"{clip}.wav"
        )
        cases.append(
            GoldenCase(
                golden_path=golden_path,
                audio_path=audio_path,
                model=model,
                response_format=response_format,
            )
        )
    return cases


PARAKEET_CASES = golden_cases("parakeet", PARAKEET_MODEL)
WHISPER_CASES = golden_cases("whisper-tiny", WHISPER_MODEL)


def request_transcription(
    client: httpx.Client,
    case: GoldenCase,
    token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected: dict[str, Any] = json.loads(case.golden_path.read_text())
    with case.audio_path.open("rb") as audio_file:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (case.audio_path.name, audio_file, "audio/wav")},
            data={
                "model": case.model,
                "response_format": case.response_format,
            },
        )
    assert response.status_code == 200, (
        f"{case.id}: expected HTTP 200, got {response.status_code}: {response.text}"
    )
    live: dict[str, Any] = response.json()
    return expected, live


@pytest.mark.golden
def test_golden_transcriptions() -> None:
    assert len(PARAKEET_CASES) == 28
    assert len(WHISPER_CASES) == 4

    base_url = os.environ.get("SIREN_GOLDEN_URL", "http://127.0.0.1:8399")
    token = os.environ.get("SIREN_GOLDEN_TOKEN", "dev_token")
    references: list[str] = []
    hypotheses: list[str] = []
    comparisons: list[tuple[GoldenCase, dict[str, Any], dict[str, Any]]] = []

    with httpx.Client(
        base_url=base_url,
        timeout=120.0,
        trust_env=False,
    ) as client:
        try:
            health_status = client.get("/health").status_code
        except httpx.RequestError as exc:
            pytest.fail(f"golden server unreachable at {base_url}: {exc}")
        assert health_status == 200, (
            f"golden server unhealthy at {base_url}: {health_status}"
        )

        for case in [*PARAKEET_CASES, *WHISPER_CASES]:
            expected, live = request_transcription(client, case, token)
            references.append(str(expected["text"]))
            hypotheses.append(str(live.get("text", "")))
            comparisons.append((case, expected, live))

        restore_case = PARAKEET_CASES[0]
        expected, live = request_transcription(client, restore_case, token)
        comparisons.append((restore_case, expected, live))

    aggregate_wer = wer(references, hypotheses)
    print(f"aggregate WER: {aggregate_wer:.12f}")
    assert aggregate_wer == 0
    for case, expected, live in comparisons:
        assert live == expected, case.id

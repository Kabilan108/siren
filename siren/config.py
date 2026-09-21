import json
import logging
import os
from pathlib import Path
from typing import Any

from siren.logging_utils import log_event

TOKEN = os.environ.get("SIREN_API_KEY", "dev_token")
CONFIG_FILE = Path(
    os.environ.get("SIREN_CONFIG_FILE", "~/config.json")
).expanduser()
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
UPLOAD_CHUNK_BYTES = 1024 * 1024
SIREN_VERSION = "1.2.0"

# Shared by interactive dictation and durable transcript jobs. Keep vocabulary
# bounded; these are recognition hints, never unconditional text replacements.
WHISPER_INITIAL_PROMPT = (
    "Claude, Codex, OpenCode, CodeRabbit, NixOS, Nix, Siren, Dictator, "
    "Jacurutu, Sietch, pnpm, PyTorch, Kubernetes, PostgreSQL, Moberg, "
    "qEEG, EDF, CNSUtils, Tailscale, tmux, Bitbucket, E-BOOST."
)


def whisper_speech_gate_enabled() -> bool:
    # Enable after audio review; independent of the recognition recipe so a
    # false rejection can be disabled without reverting the model/settings.
    value = os.environ.get("SIREN_WHISPER_SPEECH_GATE", "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("SIREN_WHISPER_SPEECH_GATE must be true/false or 1/0")
    return value in {"true", "1"}

PARAKEET_MODELS = [
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-1.1b",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-ctc-0.6b",
]

WHISPER_MODELS = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo",
]


def load_model_name() -> str:
    try:
        if CONFIG_FILE.exists():
            config: Any = json.loads(CONFIG_FILE.read_text())
            return config.get("model", DEFAULT_MODEL)
    except Exception as exc:
        log_event(
            logging.WARNING,
            "config_load_failed",
            error=str(exc),
        )
    return DEFAULT_MODEL


def save_model_name(
    model_name: str,
    *,
    request_id: str | None = None,
) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps({"model": model_name}))
    except Exception as exc:
        fields: dict[str, object] = {
            "model": model_name,
            "error": str(exc),
        }
        if request_id is not None:
            fields["request_id"] = request_id
        log_event(
            logging.WARNING,
            "config_save_failed",
            **fields,
        )

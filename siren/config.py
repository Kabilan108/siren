import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from siren.logging_utils import log_event
from siren.vocabulary import parse_vocabulary

TOKEN = os.environ.get("SIREN_API_KEY", "dev_token")
CONFIG_FILE = Path(
    os.environ.get("SIREN_CONFIG_FILE", "~/config.json")
).expanduser()
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
UPLOAD_CHUNK_BYTES = 1024 * 1024
SIREN_VERSION = "1.2.1"

# Shared by interactive dictation and durable transcript jobs. Keep vocabulary
# bounded; these are recognition hints, never unconditional text replacements.
WHISPER_INITIAL_PROMPT = (
    "Claude, Codex, OpenCode, CodeRabbit, NixOS, Nix, Siren, Dictator, "
    "Jacurutu, Sietch, pnpm, PyTorch, Kubernetes, PostgreSQL, Moberg, "
    "qEEG, EDF, CNSUtils, Tailscale, tmux, Bitbucket, E-BOOST."
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "true" if default else "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        bound = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be a finite number{bound}")
    return value


def whisper_speech_gate_enabled() -> bool:
    # Enable after audio review; independent of the recognition recipe so a
    # false rejection can be disabled without reverting the model/settings.
    return _env_flag("SIREN_WHISPER_SPEECH_GATE", False)


def parakeet_speech_gate_enabled() -> bool:
    """Skip Parakeet inference on sync requests whose audio has no speech."""
    return _env_flag("SIREN_PARAKEET_SPEECH_GATE", True)


def parakeet_claude_hint_enabled() -> bool:
    return _env_flag("SIREN_PARAKEET_CLAUDE_HINT", False)


def parakeet_default_vocabulary() -> list[str]:
    """Boost terms used when a request omits `vocabulary`.

    An explicit SIREN_PARAKEET_BOOST_TERMS (including `[]`) wins; otherwise the
    legacy Claude hint keeps its evaluated single-term list.
    """
    raw = os.environ.get("SIREN_PARAKEET_BOOST_TERMS")
    if raw is not None and raw.strip():
        return parse_vocabulary(raw, source="SIREN_PARAKEET_BOOST_TERMS")
    return ["Claude"] if parakeet_claude_hint_enabled() else []


@dataclass(frozen=True)
class BoostingParams:
    context_score: float = 1.0
    depth_scaling: float = 2.0
    alpha: float = 0.125


def parakeet_boosting_params() -> BoostingParams:
    defaults = BoostingParams()
    return BoostingParams(
        context_score=_env_float(
            "SIREN_PARAKEET_BOOST_CONTEXT_SCORE", defaults.context_score
        ),
        depth_scaling=_env_float(
            "SIREN_PARAKEET_BOOST_DEPTH_SCALING", defaults.depth_scaling
        ),
        alpha=_env_float("SIREN_PARAKEET_BOOST_ALPHA", defaults.alpha, minimum=0.0),
    )


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

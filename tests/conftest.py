import pytest

from siren import config, models

TOKEN = "dev_token"

_PARAKEET_ENV = (
    "SIREN_PARAKEET_CLAUDE_HINT",
    "SIREN_PARAKEET_BOOST_TERMS",
    "SIREN_PARAKEET_BOOST_CONTEXT_SCORE",
    "SIREN_PARAKEET_BOOST_DEPTH_SCALING",
    "SIREN_PARAKEET_BOOST_ALPHA",
)


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch, tmp_path):
    """Set up environment variables and config file for tests."""
    monkeypatch.setattr(config, "TOKEN", TOKEN)
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    for name in _PARAKEET_ENV:
        monkeypatch.delenv(name, raising=False)
    # Endpoint tests upload placeholder bytes, not audio; gate tests opt back in.
    monkeypatch.setenv("SIREN_PARAKEET_SPEECH_GATE", "false")
    models.reset_model_state()
    return config_file

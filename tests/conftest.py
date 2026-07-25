import pytest

from siren import config, models

TOKEN = "dev_token"


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch, tmp_path):
    """Set up environment variables and config file for tests."""
    monkeypatch.setattr(config, "TOKEN", TOKEN)
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    models.reset_model_state()
    return config_file

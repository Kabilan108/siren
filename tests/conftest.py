import pytest

import server

TOKEN = "dev_token"


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch, tmp_path):
    """Set up environment variables and config file for tests."""
    monkeypatch.setattr(server, "TOKEN", TOKEN)
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(server, "CONFIG_FILE", config_file)
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(server, "current_model_name", None)
    monkeypatch.setattr(server, "model_loading_task", None)
    monkeypatch.setattr(server, "model_loading_target", None)
    monkeypatch.setattr(server, "model_load_error", None)
    server.model_ready.clear()
    return config_file

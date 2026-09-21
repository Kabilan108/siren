from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from siren.backends.parakeet import configure_claude_hint


def test_hint_matches_evaluated_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_CLAUDE_HINT", "true")
    model = SimpleNamespace(
        cfg=SimpleNamespace(decoding=OmegaConf.create({"durations": [0, 1, 2, 3, 4]})),
        change_decoding_strategy=MagicMock(),
    )
    configure_claude_hint(model, "nvidia/parakeet-tdt-0.6b-v2")
    cfg = model.change_decoding_strategy.call_args.args[0]
    assert cfg.durations == [0, 1, 2, 3, 4]
    assert cfg.strategy == "greedy_batch"
    assert cfg.compute_timestamps is True
    assert cfg.greedy.boosting_tree.key_phrases_list == ["Claude"]
    assert cfg.greedy.boosting_tree_alpha == 0.125
    assert cfg.greedy.boosting_tree.context_score == 1.0
    assert cfg.greedy.boosting_tree.depth_scaling == 2.0
    assert cfg.greedy.boosting_tree.use_triton is True
    assert "greedy" not in model.cfg.decoding


@pytest.mark.parametrize("model_name,enabled", [
    ("nvidia/parakeet-tdt-0.6b-v2", "false"),
    ("nvidia/parakeet-ctc-0.6b", "true"),
    ("nvidia/parakeet-tdt-1.1b", "true"),
])
def test_hint_leaves_disabled_and_other_models_unchanged(
    monkeypatch: pytest.MonkeyPatch, model_name: str, enabled: str,
) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_CLAUDE_HINT", enabled)
    model = MagicMock()
    configure_claude_hint(model, model_name)
    model.change_decoding_strategy.assert_not_called()


def test_hint_rejects_invalid_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_CLAUDE_HINT", "typo")
    with pytest.raises(ValueError):
        configure_claude_hint(MagicMock(), "nvidia/parakeet-tdt-0.6b-v2")

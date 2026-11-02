import json

import pytest

from siren import config
from siren.vocabulary import (
    MAX_TERM_CHARS,
    MAX_TERMS,
    parse_vocabulary,
    vocabulary_key,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[]", []),
        ('["Claude"]', ["Claude"]),
        ('["  Claude  ", "tmux"]', ["Claude", "tmux"]),
        ('["NixOS", "nixos", "NIXOS", "Sietch"]', ["NixOS", "Sietch"]),
        ('["Straße", "STRASSE"]', ["Straße"]),
        ('["E-BOOST", "home manager"]', ["E-BOOST", "home manager"]),
        (json.dumps(["x" * MAX_TERM_CHARS]), ["x" * MAX_TERM_CHARS]),
        (json.dumps([f"t{i}" for i in range(MAX_TERMS)]), [f"t{i}" for i in range(MAX_TERMS)]),
    ],
)
def test_parse_vocabulary_normalizes_valid_lists(raw: str, expected: list[str]) -> None:
    assert parse_vocabulary(raw) == expected


@pytest.mark.parametrize(
    "raw,message",
    [
        ("", "JSON array of strings"),
        ("Claude, tmux", "JSON array of strings"),
        ('"Claude"', "JSON array of strings"),
        ('{"terms": ["Claude"]}', "JSON array of strings"),
        ("[1]", r"vocabulary\[0\] must be a string"),
        ('["ok", null]', r"vocabulary\[1\] must be a string"),
        ('["ok", "   "]', r"vocabulary\[1\] must be 1-64 characters"),
        (json.dumps(["x" * (MAX_TERM_CHARS + 1)]), "1-64 characters"),
        (json.dumps([f"t{i}" for i in range(MAX_TERMS + 1)]), "at most 200 terms; got 201"),
        ('["line\\nbreak"]', "control characters"),
        ("[" * 10000 + "]" * 10000, "nested too deeply"),
    ],
)
def test_parse_vocabulary_rejects_malformed_input(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_vocabulary(raw)


def test_vocabulary_key_ignores_order_but_not_spelling() -> None:
    assert vocabulary_key(["tmux", "Claude"]) == vocabulary_key(["Claude", "tmux"])
    assert vocabulary_key(["Claude"]) != vocabulary_key(["claude"])
    assert vocabulary_key([]) == ()


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, []),
        ({"SIREN_PARAKEET_CLAUDE_HINT": "false"}, []),
        ({"SIREN_PARAKEET_CLAUDE_HINT": "true"}, ["Claude"]),
        ({"SIREN_PARAKEET_CLAUDE_HINT": "1", "SIREN_PARAKEET_BOOST_TERMS": ""}, ["Claude"]),
        ({"SIREN_PARAKEET_CLAUDE_HINT": "true", "SIREN_PARAKEET_BOOST_TERMS": "[]"}, []),
        (
            {"SIREN_PARAKEET_CLAUDE_HINT": "true", "SIREN_PARAKEET_BOOST_TERMS": '["tmux", "Sietch"]'},
            ["tmux", "Sietch"],
        ),
        ({"SIREN_PARAKEET_BOOST_TERMS": '["Claude", "claude"]'}, ["Claude"]),
    ],
)
def test_default_vocabulary_resolution(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: list[str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert config.parakeet_default_vocabulary() == expected


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("SIREN_PARAKEET_BOOST_TERMS", "Claude", "SIREN_PARAKEET_BOOST_TERMS must be a JSON array"),
        ("SIREN_PARAKEET_BOOST_TERMS", '[""]', r"SIREN_PARAKEET_BOOST_TERMS\[0\]"),
        ("SIREN_PARAKEET_CLAUDE_HINT", "typo", "SIREN_PARAKEET_CLAUDE_HINT must be"),
    ],
)
def test_default_vocabulary_rejects_bad_settings(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        config.parakeet_default_vocabulary()


def test_boosting_params_default_to_evaluated_values() -> None:
    assert config.parakeet_boosting_params() == config.BoostingParams(
        context_score=1.0, depth_scaling=2.0, alpha=0.125,
    )


def test_boosting_params_read_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIREN_PARAKEET_BOOST_CONTEXT_SCORE", "1.5")
    monkeypatch.setenv("SIREN_PARAKEET_BOOST_DEPTH_SCALING", "1")
    monkeypatch.setenv("SIREN_PARAKEET_BOOST_ALPHA", "0.3")
    assert config.parakeet_boosting_params() == config.BoostingParams(
        context_score=1.5, depth_scaling=1.0, alpha=0.3,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("SIREN_PARAKEET_BOOST_ALPHA", "-0.1"),
        ("SIREN_PARAKEET_BOOST_ALPHA", "strong"),
        ("SIREN_PARAKEET_BOOST_CONTEXT_SCORE", "nan"),
        ("SIREN_PARAKEET_BOOST_DEPTH_SCALING", "inf"),
    ],
)
def test_boosting_params_reject_bad_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        config.parakeet_boosting_params()

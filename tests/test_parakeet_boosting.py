import asyncio
import time
import wave
from pathlib import Path
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf, open_dict

from siren import config
from siren.backends.parakeet import (
    BoostingDecoder,
    ParakeetBackend,
    boosting_decoder_for,
)

EVALUATED = config.BoostingParams(context_score=1.0, depth_scaling=2.0, alpha=0.125)


class FakeModel:
    """Mimics the attributes NeMo's change_decoding_strategy reassigns."""

    def __init__(self) -> None:
        self.cfg = OmegaConf.create({"decoding": {"durations": [0, 1, 2, 3, 4]}})
        self.joint = SimpleNamespace(fuse_loss_wer=False, temperature=1.0)
        self.decoding: Any = "original"
        self.wer: Any = "original-wer"
        self.loss: Any = None
        self.builds: list[DictConfig] = []
        self.on_transcribe: Callable[[], object] | None = None

    def change_decoding_strategy(self, cfg: DictConfig, verbose: bool = True) -> None:
        self.builds.append(cfg)
        terms = cfg.greedy.boosting_tree.key_phrases_list
        self.decoding = SimpleNamespace(terms=None if terms is None else list(terms))
        self.wer = SimpleNamespace(decoding=self.decoding)
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = cfg

    def transcribe(self, paths: list[str], **kwargs: object) -> list[Any]:
        if self.on_transcribe is not None:
            self.on_transcribe()
        return [SimpleNamespace(text=" ".join(self.decoding.terms or []), timestamp={})]


@pytest.fixture
def clip(tmp_path: Path) -> str:
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return str(path)


def make_decoder(
    default_terms: list[str] | None = None,
    params: config.BoostingParams = EVALUATED,
) -> tuple[FakeModel, BoostingDecoder]:
    model = FakeModel()
    decoder = BoostingDecoder(
        model,
        default_terms=["Claude"] if default_terms is None else default_terms,
        params=params,
    )
    return model, decoder


def test_default_list_builds_the_evaluated_configuration() -> None:
    model, decoder = make_decoder()
    fields = decoder.activate(None)
    cfg = model.builds[0]
    assert cfg.durations == [0, 1, 2, 3, 4]
    assert cfg.strategy == "greedy_batch"
    assert cfg.compute_timestamps is True
    assert cfg.greedy.boosting_tree.key_phrases_list == ["Claude"]
    assert cfg.greedy.boosting_tree_alpha == 0.125
    assert cfg.greedy.boosting_tree.context_score == 1.0
    assert cfg.greedy.boosting_tree.depth_scaling == 2.0
    assert cfg.greedy.boosting_tree.use_triton is True
    assert fields["decoding_swap"] == "built"
    assert fields["boost_source"] == "default"
    assert fields["boost_terms"] == 1


def test_configured_params_reach_the_tree() -> None:
    params = config.BoostingParams(context_score=1.5, depth_scaling=1.0, alpha=0.3)
    model, decoder = make_decoder(params=params)
    decoder.activate(["tmux"])
    cfg = model.builds[0]
    assert cfg.greedy.boosting_tree.context_score == 1.5
    assert cfg.greedy.boosting_tree.depth_scaling == 1.0
    assert cfg.greedy.boosting_tree_alpha == 0.3


def test_empty_list_decodes_without_a_tree() -> None:
    model, decoder = make_decoder()
    decoder.activate([])
    cfg = model.builds[0]
    assert cfg.greedy.boosting_tree.key_phrases_list is None
    assert cfg.compute_timestamps is True
    assert cfg.strategy == "greedy_batch"


def test_empty_default_decodes_without_a_tree() -> None:
    model, decoder = make_decoder(default_terms=[])
    decoder.activate(None)
    assert model.builds[0].greedy.boosting_tree.key_phrases_list is None


def test_unchanged_list_skips_change_decoding_strategy() -> None:
    model, decoder = make_decoder()
    decoder.activate(None)
    fields = decoder.activate(["Claude"])
    assert len(model.builds) == 1
    assert fields["decoding_swap"] == "none"
    assert fields["boost_source"] == "request"


def test_term_order_shares_one_cache_entry() -> None:
    model, decoder = make_decoder()
    decoder.activate(["tmux", "Sietch"])
    decoder.activate([])
    fields = decoder.activate(["Sietch", "tmux"])
    assert len(model.builds) == 2
    assert fields["decoding_swap"] == "cached"


def test_cached_state_is_reinstalled_without_rebuilding() -> None:
    model, decoder = make_decoder()
    decoder.activate(["tmux"])
    tmux_decoding = model.decoding
    tmux_wer = model.wer
    decoder.activate(["Sietch"])
    assert model.decoding.terms == ["Sietch"]

    fields = decoder.activate(["tmux"])
    assert fields["decoding_swap"] == "cached"
    assert len(model.builds) == 2
    assert model.decoding is tmux_decoding
    assert model.wer is tmux_wer
    assert list(model.cfg.decoding.greedy.boosting_tree.key_phrases_list) == ["tmux"]
    # transcribe(timestamps=True) rebuilds decoding unless this stays true.
    assert model.cfg.decoding.compute_timestamps is True


def test_fused_joint_gets_the_reinstalled_wer() -> None:
    model, decoder = make_decoder()
    model.joint = SimpleNamespace(
        fuse_loss_wer=True,
        temperature=1.0,
        set_loss=lambda loss: setattr(model.joint, "loss", loss),
        set_wer=lambda wer: setattr(model.joint, "wer", wer),
    )
    model.loss = "loss"
    decoder.activate(["tmux"])
    tmux_wer = model.wer
    decoder.activate(["Sietch"])
    decoder.activate(["tmux"])
    assert model.joint.wer is tmux_wer
    assert model.joint.loss == "loss"


def test_cache_evicts_the_least_recently_used_list() -> None:
    model, decoder = make_decoder()
    for term in ["a", "b", "c", "d"]:
        decoder.activate([term])
    decoder.activate(["a"])  # refresh "a"; "b" is now the oldest
    decoder.activate(["e"])  # evicts "b"
    assert len(model.builds) == 5

    assert decoder.activate(["a"])["decoding_swap"] == "cached"
    assert decoder.activate(["b"])["decoding_swap"] == "built"
    assert len(model.builds) == 6


def test_failed_build_forces_a_reinstall_of_the_next_list() -> None:
    model, decoder = make_decoder()
    decoder.activate(["tmux"])
    tmux_decoding = model.decoding

    def broken(cfg: DictConfig, verbose: bool = True) -> None:
        model.decoding = "half-built"
        raise RuntimeError("tree build failed")

    model.change_decoding_strategy = broken
    with pytest.raises(RuntimeError):
        decoder.activate(["Sietch"])

    assert decoder.activate(["tmux"])["decoding_swap"] == "cached"
    assert model.decoding is tmux_decoding


def test_boosting_is_limited_to_the_evaluated_checkpoint() -> None:
    model = FakeModel()
    for name in ["nvidia/parakeet-ctc-0.6b", "nvidia/parakeet-tdt-1.1b"]:
        assert boosting_decoder_for(model, name, default_terms=["Claude"], params=EVALUATED) is None
    decoder = boosting_decoder_for(
        model, "nvidia/parakeet-tdt-0.6b-v2", default_terms=["Claude"], params=EVALUATED,
    )
    assert isinstance(decoder, BoostingDecoder)


@pytest.mark.asyncio
async def test_inference_runs_under_the_decoder_lock_with_its_own_list(clip: str) -> None:
    model, decoder = make_decoder()
    backend = ParakeetBackend(model, decoder=decoder)
    locked_during_inference: list[bool] = []
    model.on_transcribe = lambda: locked_during_inference.append(decoder.lock.locked())

    assert (await backend.transcribe(clip, vocabulary=["tmux"])).text == "tmux"
    assert (await backend.transcribe(clip)).text == "Claude"
    assert locked_during_inference == [True, True]


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_swap_decoding_mid_inference(clip: str) -> None:
    """Even without the API semaphore, a swap waits for in-flight inference."""
    model, decoder = make_decoder()
    backend = ParakeetBackend(model, decoder=decoder)
    model.on_transcribe = lambda: time.sleep(0.05)

    results = await asyncio.gather(
        *(backend.transcribe(clip, vocabulary=[term]) for term in ["alpha", "beta", "gamma"])
    )
    assert [result.text for result in results] == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_other_parakeet_models_ignore_vocabulary(clip: str) -> None:
    model = FakeModel()
    model.decoding = SimpleNamespace(terms=["untouched"])
    backend = ParakeetBackend(model)
    result = await backend.transcribe(clip, vocabulary=["tmux"])
    assert result.text == "untouched"
    assert model.builds == []

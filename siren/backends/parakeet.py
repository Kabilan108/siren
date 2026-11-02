import copy
import logging
import math
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from siren import config
from siren.audio import get_wav_info
from siren.concurrency import run_in_worker_thread
from siren.logging_utils import log_event
from siren.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from siren.vocabulary import vocabulary_key


def get_cuda_stats() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
    }


def parakeet_words(hypothesis: Any) -> list[TranscriptionWord]:
    timestamps = getattr(hypothesis, "timestamp", {}) or {}
    raw_words = timestamps.get("word") or []
    words = [
        TranscriptionWord(
            start=float(raw_word.get("start", 0.0)),
            end=float(raw_word.get("end", 0.0)),
            word=str(raw_word.get("word") or raw_word.get("text") or ""),
        )
        for raw_word in raw_words
        if str(raw_word.get("word") or raw_word.get("text") or "").strip()
    ]
    finite_words = [
        word
        for word in words
        if math.isfinite(word.start) and math.isfinite(word.end)
    ]
    if len(finite_words) != len(words):
        log_event(
            logging.WARNING,
            "word_timestamps_nonfinite_dropped",
            dropped=len(words) - len(finite_words),
        )
    return sorted(finite_words, key=lambda word: word.start)


def parakeet_segments(
    hypothesis: Any,
    *,
    word_timestamps: bool = False,
) -> list[TranscriptionSegment]:
    timestamps = getattr(hypothesis, "timestamp", {}) or {}
    raw_segments = timestamps.get("segment") or timestamps.get("word") or []
    segments: list[TranscriptionSegment] = []
    for segment in raw_segments:
        text = str(
            segment.get("segment")
            or segment.get("word")
            or segment.get("text")
            or ""
        ).strip()
        if not text:
            continue
        segments.append(
            TranscriptionSegment(
                id=len(segments),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=text,
            )
        )

    if not word_timestamps:
        return segments

    words = parakeet_words(hypothesis)
    if timestamps.get("segment"):
        word_index = 0
        for index, segment in enumerate(segments):
            next_start = (
                segments[index + 1].start
                if index + 1 < len(segments)
                else float("inf")
            )
            segment.words = []
            while word_index < len(words) and words[word_index].start < next_start:
                segment.words.append(words[word_index])
                word_index += 1
    else:
        for segment, word in zip(segments, words, strict=False):
            segment.words = [word]
    return segments


BOOSTING_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
DECODING_CACHE_SIZE = 4


@dataclass
class _DecodingState:
    decoding: Any
    wer: Any
    cfg: Any


class BoostingDecoder:
    """Keeps a small LRU of built decoding states keyed by boost-term list.

    Building a state through `change_decoding_strategy` constructs the GPU
    boosting tree and a fresh greedy decoder whose CUDA graphs are captured on
    first use. Reinstalling a cached state only reassigns the model attributes
    that `change_decoding_strategy` sets. Callers must hold `lock` from
    `activate` through the end of inference.
    """

    def __init__(
        self,
        model: Any,
        *,
        default_terms: Sequence[str],
        params: config.BoostingParams,
        cache_size: int = DECODING_CACHE_SIZE,
    ) -> None:
        from nemo.collections.asr.parts.submodules.rnnt_decoding import (
            RNNTBPEDecodingConfig,
        )
        from omegaconf import OmegaConf

        defaults = OmegaConf.create(
            OmegaConf.to_container(OmegaConf.structured(RNNTBPEDecodingConfig))
        )
        self.model = model
        self.default_terms = list(default_terms)
        self.params = params
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self._base_cfg = OmegaConf.merge(defaults, model.cfg.decoding)
        self._cache: OrderedDict[tuple[str, ...], _DecodingState] = OrderedDict()
        self._active: tuple[str, ...] | None = None

    def decoding_cfg(self, key: tuple[str, ...]) -> Any:
        from omegaconf import open_dict

        cfg = copy.deepcopy(self._base_cfg)
        with open_dict(cfg):
            # compute_timestamps must already be true: transcribe(timestamps=True)
            # otherwise rebuilds the decoder and silently drops the cached state.
            cfg.strategy = "greedy_batch"
            cfg.compute_timestamps = True
            if key:
                cfg.greedy.boosting_tree.key_phrases_list = list(key)
                cfg.greedy.boosting_tree.context_score = self.params.context_score
                cfg.greedy.boosting_tree.depth_scaling = self.params.depth_scaling
                cfg.greedy.boosting_tree.use_triton = True
                cfg.greedy.boosting_tree_alpha = self.params.alpha
            else:
                cfg.greedy.boosting_tree.key_phrases_list = None
                cfg.greedy.boosting_tree_alpha = 0.0
        return cfg

    def activate(self, terms: Sequence[str] | None) -> dict[str, object]:
        resolved = self.default_terms if terms is None else terms
        key = vocabulary_key(resolved)
        fields: dict[str, object] = {
            "boost_terms": len(key),
            "boost_source": "default" if terms is None else "request",
        }
        if key == self._active:
            self._cache.move_to_end(key)
            fields["decoding_swap"] = "none"
            return fields

        start = time.perf_counter()
        state = self._cache.get(key)
        # A failed build can leave the model half-switched; force a reinstall.
        self._active = None
        if state is None:
            self.model.change_decoding_strategy(self.decoding_cfg(key), verbose=False)
            state = _DecodingState(
                decoding=self.model.decoding,
                wer=self.model.wer,
                cfg=copy.deepcopy(self.model.cfg.decoding),
            )
            self._cache[key] = state
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            fields["decoding_swap"] = "built"
        else:
            self._install(state)
            self._cache.move_to_end(key)
            fields["decoding_swap"] = "cached"
        self._active = key
        fields["decoding_swap_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return fields

    def _install(self, state: _DecodingState) -> None:
        """Mirror the assignments NeMo's change_decoding_strategy makes."""
        from omegaconf import open_dict

        model = self.model
        model.decoding = state.decoding
        model.wer = state.wer
        joint_fused_batch_size = getattr(state.decoding, "joint_fused_batch_size", None)
        if model.joint.fuse_loss_wer or (
            joint_fused_batch_size is not None and joint_fused_batch_size > 0
        ):
            model.joint.set_loss(model.loss)
            model.joint.set_wer(state.wer)
        model.joint.temperature = state.cfg.get("temperature", 1.0)
        with open_dict(model.cfg.decoding):
            model.cfg.decoding = state.cfg


async def process_parakeet_transcription(
    audio_path: str,
    model: Any,
    *,
    word_timestamps: bool = False,
    request_id: str | None = None,
    decoder: BoostingDecoder | None = None,
    vocabulary: Sequence[str] | None = None,
) -> TranscriptionResult:
    audio_info = get_wav_info(audio_path, request_id=request_id)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    decoding_fields: dict[str, object] = {}

    def run() -> Any:
        if decoder is None:
            return model.transcribe([audio_path], timestamps=True, verbose=False)
        with decoder.lock:
            decoding_fields.update(decoder.activate(vocabulary))
            return model.transcribe([audio_path], timestamps=True, verbose=False)

    start = time.perf_counter()
    output = await run_in_worker_thread(run)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    log_event(
        logging.INFO,
        "parakeet_transcribe",
        request_id=request_id,
        latency_ms=elapsed_ms,
        **decoding_fields,
        **audio_info,
        **get_cuda_stats(),
    )
    if not output:
        return TranscriptionResult(
            text="",
            language="en",
            duration=float(audio_info.get("audio_duration_sec", 0.0)),
            segments=[],
        )
    hypothesis = output[0]
    return TranscriptionResult(
        text=str(hypothesis.text),
        language="en",
        duration=float(audio_info.get("audio_duration_sec", 0.0)),
        segments=parakeet_segments(
            hypothesis,
            word_timestamps=word_timestamps,
        ),
    )


class ParakeetBackend:
    def __init__(self, model: Any, *, decoder: BoostingDecoder | None = None) -> None:
        self.model = model
        self.decoder = decoder

    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        request_id: str | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> TranscriptionResult:
        if vocabulary is not None and self.decoder is None:
            log_event(
                logging.INFO,
                "vocabulary_ignored",
                request_id=request_id,
                reason="model does not support boosting",
            )
        return await process_parakeet_transcription(
            audio_path,
            self.model,
            word_timestamps=word_timestamps,
            request_id=request_id,
            decoder=self.decoder,
            vocabulary=vocabulary,
        )


def boosting_decoder_for(
    model: Any,
    model_name: str,
    *,
    default_terms: Sequence[str],
    params: config.BoostingParams,
) -> BoostingDecoder | None:
    """Only the evaluated TDT checkpoint gets vocabulary boosting."""
    if model_name != BOOSTING_MODEL:
        return None
    return BoostingDecoder(model, default_terms=default_terms, params=params)


def load_parakeet_backend(model_name: str) -> ParakeetBackend:
    warnings.filterwarnings("ignore", module="nemo")
    warnings.filterwarnings("ignore", message=".*torchaudio.*")
    # Read settings before the slow model load so a malformed value fails fast.
    default_terms = config.parakeet_default_vocabulary()
    params = config.parakeet_boosting_params()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    decoder = boosting_decoder_for(
        model, model_name, default_terms=default_terms, params=params
    )
    return ParakeetBackend(model, decoder=decoder)

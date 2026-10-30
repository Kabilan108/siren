import math
import os
import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from siren.schemas import (
    AlignedSegment,
    AlignedWord,
    AlignmentResponse,
    DiarizationTurn,
    TranscriptionWord,
)
from siren.segmentation import segment_words

_TIE_TOLERANCE = 1e-6
_DEFAULT_SPEAKER = "SPEAKER_00"
_IndexedTurn = tuple[int, DiarizationTurn]
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


@dataclass(frozen=True)
class SpeakerSmoothing:
    """Limits for folding a short speaker run back into the surrounding turn.

    A run is a candidate when it is short on both counts and the same speaker
    talks on either side of it. It is folded when the speech on both sides
    flows into it without a pause of at least ``max_gap`` seconds, or, with
    ``fold_mid_sentence``, when the preceding word does not end a sentence.
    A reply that follows a finished sentence and a real pause keeps its own
    speaker.
    """

    max_words: int = 3
    max_seconds: float = 1.0
    max_gap: float = 0.3
    fold_mid_sentence: bool = True


def get_speaker_smoothing() -> SpeakerSmoothing | None:
    defaults = SpeakerSmoothing()
    smoothing = SpeakerSmoothing(
        max_words=int(
            os.environ.get("SIREN_SMOOTH_MAX_WORDS", str(defaults.max_words))
        ),
        max_seconds=float(
            os.environ.get("SIREN_SMOOTH_MAX_SECONDS", str(defaults.max_seconds))
        ),
        max_gap=float(
            os.environ.get("SIREN_SMOOTH_MAX_GAP", str(defaults.max_gap))
        ),
        fold_mid_sentence=_env_flag(
            "SIREN_SMOOTH_MID_SENTENCE", defaults.fold_mid_sentence
        ),
    )
    if smoothing.max_words <= 0 or smoothing.max_seconds <= 0.0:
        return None
    return smoothing


def smooth_speaker_runs(
    words: Sequence[AlignedWord],
    smoothing: SpeakerSmoothing,
) -> list[AlignedWord]:
    """Reassign short runs sandwiched inside one speaker's continuous turn."""
    runs = [list(run) for _speaker, run in groupby(words, key=lambda w: w.speaker)]
    smoothed: list[AlignedWord] = []
    for index, run in enumerate(runs):
        if 0 < index < len(runs) - 1 and _is_flicker(
            runs[index - 1], run, runs[index + 1], smoothing
        ):
            speaker = runs[index - 1][0].speaker
            smoothed.extend(word.model_copy(update={"speaker": speaker}) for word in run)
        else:
            smoothed.extend(run)
    return smoothed


def _is_flicker(
    before: Sequence[AlignedWord],
    run: Sequence[AlignedWord],
    after: Sequence[AlignedWord],
    smoothing: SpeakerSmoothing,
) -> bool:
    if before[0].speaker != after[0].speaker:
        return False
    if len(run) > smoothing.max_words:
        return False
    if run[-1].end - run[0].start > smoothing.max_seconds:
        return False
    if smoothing.fold_mid_sentence and not _ends_sentence(before[-1].word):
        return True
    gap_before = run[0].start - before[-1].end
    gap_after = after[0].start - run[-1].end
    return gap_before < smoothing.max_gap and gap_after < smoothing.max_gap


def _ends_sentence(word: str) -> bool:
    return _SENTENCE_END.search(word.strip()) is not None


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


def assign_speakers_to_words(
    words: Sequence[TranscriptionWord],
    turns: Sequence[DiarizationTurn],
) -> list[AlignedWord]:
    """Normalize timestamps and assign every word to a speaker."""
    sorted_words = sorted(words, key=lambda word: (word.start, word.end))
    sorted_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    _validate_intervals(sorted_words, "words")
    _validate_intervals(sorted_turns, "turns")
    return _assign_speakers_to_sorted_words(sorted_words, sorted_turns)


def _assign_speakers_to_sorted_words(
    words: Sequence[TranscriptionWord],
    turns: Sequence[DiarizationTurn],
) -> list[AlignedWord]:
    """Assign speakers to validated words and turns in timestamp order."""

    if not turns:
        return [
            AlignedWord(
                start=word.start,
                end=word.end,
                word=word.word,
                speaker=_DEFAULT_SPEAKER,
            )
            for word in words
        ]

    aligned_words: list[AlignedWord] = []
    previous_speaker: str | None = None
    indexed_turns = list(enumerate(turns))
    turn_starts = [turn.start for turn in turns]
    turns_by_end = sorted(
        indexed_turns,
        key=lambda item: (item[1].end, item[1].start, item[0]),
    )
    turn_ends = [turn.end for _index, turn in turns_by_end]
    maximum_turn_duration = max(turn.end - turn.start for turn in turns)

    for word in words:
        candidate_start = bisect_left(
            turn_starts,
            word.start - maximum_turn_duration,
        )
        candidate_end = bisect_right(turn_starts, word.end)
        nearby_turns = indexed_turns[candidate_start:candidate_end]
        intersections = [
            (index, turn, _intersection_weight(word, turn))
            for index, turn in nearby_turns
        ]
        overlapping = [item for item in intersections if item[2] > 0.0]

        if overlapping:
            maximum_intersection = max(item[2] for item in overlapping)
            candidates = [
                (index, turn)
                for index, turn, intersection in overlapping
                if maximum_intersection - intersection <= _TIE_TOLERANCE
            ]
        else:
            touching = [
                (index, turn)
                for index, turn in nearby_turns
                if _interval_gap(word, turn) == 0.0
            ]
            candidates = touching or _nearest_gap_candidates(
                word,
                indexed_turns,
                turn_starts,
                turns_by_end,
                turn_ends,
            )

        selected_turn = _select_turn(candidates, previous_speaker)
        previous_speaker = selected_turn.speaker
        aligned_words.append(
            AlignedWord(
                start=word.start,
                end=word.end,
                word=word.word,
                speaker=selected_turn.speaker,
            )
        )

    return aligned_words


def segment_aligned_words(
    words: Sequence[AlignedWord],
    *,
    pause_threshold: float = 0.6,
    max_segment_seconds: float = 30.0,
) -> list[AlignedSegment]:
    """Segment speaker runs using the shared pause and maximum-span rules."""
    segments: list[AlignedSegment] = []

    for speaker, grouped_words in groupby(words, key=lambda word: word.speaker):
        speaker_words = list(grouped_words)
        word_offset = 0
        for segment in segment_words(
            speaker_words,
            pause_threshold=pause_threshold,
            max_segment_seconds=max_segment_seconds,
        ):
            segment_word_count = len(segment.words or [])
            aligned_segment_words = speaker_words[
                word_offset : word_offset + segment_word_count
            ]
            word_offset += segment_word_count
            segments.append(
                AlignedSegment(
                    id=len(segments),
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker,
                    text=segment.text,
                    words=aligned_segment_words,
                )
            )

    return segments


def align_words(
    words: Sequence[TranscriptionWord],
    turns: Sequence[DiarizationTurn],
    *,
    pause_threshold: float = 0.6,
    max_segment_seconds: float = 30.0,
    smoothing: SpeakerSmoothing | None = None,
) -> AlignmentResponse:
    """Assign speakers and build speaker-aware transcript segments."""
    sorted_words = sorted(words, key=lambda word: (word.start, word.end))
    sorted_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    _validate_intervals(sorted_words, "words")
    _validate_intervals(sorted_turns, "turns")
    aligned_words = _assign_speakers_to_sorted_words(sorted_words, sorted_turns)
    if smoothing is not None:
        aligned_words = smooth_speaker_runs(aligned_words, smoothing)
    segments = segment_aligned_words(
        aligned_words,
        pause_threshold=pause_threshold,
        max_segment_seconds=max_segment_seconds,
    )
    return AlignmentResponse(
        speakers=sorted({segment.speaker for segment in segments}),
        segments=segments,
    )


def _intersection_weight(
    word: TranscriptionWord,
    turn: DiarizationTurn,
) -> float:
    intersection = min(word.end, turn.end) - max(word.start, turn.start)
    if intersection > 0.0:
        return intersection
    if word.start == word.end and turn.start <= word.start and word.start < turn.end:
        return _TIE_TOLERANCE
    return 0.0


def _interval_gap(
    word: TranscriptionWord,
    turn: DiarizationTurn,
) -> float:
    if turn.end < word.start:
        return word.start - turn.end
    if word.end < turn.start:
        return turn.start - word.end
    return 0.0


def _nearest_gap_candidates(
    word: TranscriptionWord,
    turns_by_start: Sequence[_IndexedTurn],
    turn_starts: Sequence[float],
    turns_by_end: Sequence[_IndexedTurn],
    turn_ends: Sequence[float],
) -> list[_IndexedTurn]:
    candidates: list[_IndexedTurn] = []

    left_end = bisect_right(turn_ends, word.start)
    if left_end:
        nearest_end = turn_ends[left_end - 1]
        left_start = bisect_left(
            turn_ends,
            nearest_end - _TIE_TOLERANCE,
        )
        candidates.extend(turns_by_end[left_start:left_end])

    right_start = bisect_left(turn_starts, word.end)
    if right_start < len(turns_by_start):
        nearest_start = turn_starts[right_start]
        right_end = bisect_right(
            turn_starts,
            nearest_start + _TIE_TOLERANCE,
        )
        candidates.extend(turns_by_start[right_start:right_end])

    unique_candidates = list(
        {index: (index, turn) for index, turn in candidates}.values()
    )
    minimum_gap = min(_interval_gap(word, turn) for _index, turn in unique_candidates)
    return [
        (index, turn)
        for index, turn in unique_candidates
        if _interval_gap(word, turn) - minimum_gap <= _TIE_TOLERANCE
    ]


def _select_turn(
    candidates: Sequence[_IndexedTurn],
    previous_speaker: str | None,
) -> DiarizationTurn:
    if previous_speaker is not None:
        for _index, turn in candidates:
            if turn.speaker == previous_speaker:
                return turn

    return min(candidates, key=lambda item: (item[1].start, item[0]))[1]


def _validate_intervals(
    intervals: Sequence[TranscriptionWord] | Sequence[DiarizationTurn],
    field_name: str,
) -> None:
    for index, interval in enumerate(intervals):
        if not math.isfinite(interval.start) or not math.isfinite(interval.end):
            raise ValueError(
                f"{field_name}[{index}] has non-finite start or end values."
            )
        if interval.end < interval.start:
            raise ValueError(f"{field_name}[{index}] has end before start.")

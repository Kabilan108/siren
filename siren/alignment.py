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
_SENTENCE_END = re.compile(r"[.?!\u2026][\"'\u201d\u2019\u00bb)\]}]*$")
_QUESTION_END = re.compile(r"\?[\"'\u201d\u2019\u00bb)\]}]*$")
_PUNCTUATION = re.compile(r"[.,?!;:\u2026]")
_WORD_MARKS = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb()[]{}.,?!;:\u2026"
# Words that open a reply or backchannel; a run starting with one is never folded.
_REPLY_WORDS = frozenset(
    {
        "absolutely", "agreed", "alright", "correct", "cool", "definitely",
        "exactly", "gotcha", "great", "hmm", "mhm", "mm", "nah", "no", "nope",
        "ok", "okay", "oh", "perfect", "right", "sure", "thank", "thanks",
        "totally", "true", "uh-huh", "wow", "yeah", "yep", "yes", "yup",
    }
)
# Parakeet TDT timestamps step in 80 ms frames; a smaller gap means no silent
# frame between the words.
_CONTINUOUS_GAP = 0.04
_CONTEXT_WORDS = 8


@dataclass(frozen=True)
class SpeakerSmoothing:
    """Limits for folding a short speaker run back into the surrounding turn.

    A candidate run has at most ``max_words`` words and ``max_seconds``
    seconds and sits between two runs of one other speaker. It is never folded
    when it starts with a reply word ("yeah", "no", "okay", ...) or follows a
    question. A run that ends a sentence counts as a standalone reply unless
    the word before it is unfinished and flows into it with no silent frame.
    Otherwise it is folded when it is glued to an unfinished sentence on one
    side by a gap under ``max_pause``; where the surrounding words carry no
    punctuation, both gaps must be under ``max_pause``.
    """

    max_words: int = 3
    max_seconds: float = 1.0
    max_pause: float = 0.5


def get_speaker_smoothing() -> SpeakerSmoothing | None:
    enabled = os.environ.get("SIREN_SPEAKER_SMOOTHING", "true").lower()
    if enabled not in {"true", "false", "1", "0"}:
        raise ValueError("SIREN_SPEAKER_SMOOTHING must be true/false or 1/0")
    if enabled in {"false", "0"}:
        return None
    defaults = SpeakerSmoothing()
    smoothing = SpeakerSmoothing(
        max_words=int(
            os.environ.get("SIREN_SMOOTH_MAX_WORDS", str(defaults.max_words))
        ),
        max_seconds=float(
            os.environ.get("SIREN_SMOOTH_MAX_SECONDS", str(defaults.max_seconds))
        ),
        max_pause=float(
            os.environ.get("SIREN_SMOOTH_MAX_PAUSE", str(defaults.max_pause))
        ),
    )
    if smoothing.max_words <= 0 or smoothing.max_seconds <= 0.0:
        return None
    return smoothing


def smooth_speaker_runs(
    words: Sequence[AlignedWord],
    smoothing: SpeakerSmoothing,
) -> list[AlignedWord]:
    """Reassign short runs glued inside another speaker's sentence.

    Runs are decided left to right against the already merged turn, so a
    folded run joins its neighbours before the next run is judged.
    """
    runs = [list(run) for _speaker, run in groupby(words, key=lambda w: w.speaker)]
    merged: list[list[AlignedWord]] = []
    for index, run in enumerate(runs):
        if (
            merged
            and index < len(runs) - 1
            and _is_flicker(merged[-1], run, runs[index + 1], smoothing)
        ):
            speaker = merged[-1][0].speaker
            run = [word.model_copy(update={"speaker": speaker}) for word in run]
        if merged and merged[-1][0].speaker == run[0].speaker:
            merged[-1].extend(run)
        else:
            merged.append(run)
    return [word for run in merged for word in run]


def _is_flicker(
    before: Sequence[AlignedWord],
    run: Sequence[AlignedWord],
    after: Sequence[AlignedWord],
    smoothing: SpeakerSmoothing,
) -> bool:
    speaker = before[0].speaker
    if speaker != after[0].speaker or speaker == run[0].speaker:
        return False
    if len(run) > smoothing.max_words:
        return False
    if run[-1].end - run[0].start > smoothing.max_seconds:
        return False
    if _bare_word(run[0].word) in _REPLY_WORDS:
        return False
    previous = before[-1].word
    if _QUESTION_END.search(previous.strip()):
        return False
    gap_before = run[0].start - before[-1].end
    gap_after = after[0].start - run[-1].end
    context = [*before[-_CONTEXT_WORDS:], *run, *after[:_CONTEXT_WORDS]]
    if not any(_PUNCTUATION.search(word.word) for word in context):
        return gap_before < smoothing.max_pause and gap_after < smoothing.max_pause
    if _ends_sentence(run[-1].word):
        return not _ends_sentence(previous) and gap_before < _CONTINUOUS_GAP
    glued_before = not _ends_sentence(previous) and gap_before < smoothing.max_pause
    glued_after = gap_after < smoothing.max_pause
    return glued_before or glued_after


def _ends_sentence(word: str) -> bool:
    return _SENTENCE_END.search(word.strip()) is not None


def _bare_word(word: str) -> str:
    return word.strip().strip(_WORD_MARKS).lower()


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

import math
import time

import pytest

from siren.alignment import (
    align_words,
    assign_speakers_to_words,
    segment_aligned_words,
)
from siren.schemas import AlignedWord, DiarizationTurn, TranscriptionWord


def word(start: float, end: float, text: str) -> TranscriptionWord:
    return TranscriptionWord(start=start, end=end, word=text)


def turn(start: float, end: float, speaker: str) -> DiarizationTurn:
    return DiarizationTurn(start=start, end=end, speaker=speaker)


def aligned_word(start: float, end: float, text: str, speaker: str) -> AlignedWord:
    return AlignedWord(start=start, end=end, word=text, speaker=speaker)


def test_maximum_intersection_assigns_speaker() -> None:
    aligned = assign_speakers_to_words(
        [word(0.8, 1.8, "hello")],
        [
            turn(0.0, 1.0, "SPEAKER_00"),
            turn(1.0, 2.0, "SPEAKER_01"),
        ],
    )

    assert [item.speaker for item in aligned] == ["SPEAKER_01"]


def test_uncovered_word_uses_nearest_interval_gap() -> None:
    aligned = assign_speakers_to_words(
        [word(4.0, 5.0, "gap")],
        [
            turn(0.0, 1.0, "SPEAKER_00"),
            turn(10.0, 11.0, "SPEAKER_01"),
        ],
    )

    assert aligned[0].speaker == "SPEAKER_00"


def test_interval_gap_beats_turn_midpoint_distance() -> None:
    aligned = assign_speakers_to_words(
        [word(10.0, 11.0, "gap")],
        [
            turn(0.0, 9.0, "SPEAKER_00"),
            turn(11.5, 100.0, "SPEAKER_01"),
        ],
    )

    assert aligned[0].speaker == "SPEAKER_01"


def test_tied_intersection_sticks_to_previous_speaker() -> None:
    aligned = assign_speakers_to_words(
        [
            word(0.1, 0.4, "first"),
            word(0.5, 1.5, "tied"),
        ],
        [
            turn(0.0, 0.5, "SPEAKER_01"),
            turn(0.4, 1.0, "SPEAKER_00"),
            turn(1.0, 2.0, "SPEAKER_01"),
        ],
    )

    assert [item.speaker for item in aligned] == ["SPEAKER_01", "SPEAKER_01"]


def test_tied_gap_sticks_to_previous_speaker() -> None:
    aligned = assign_speakers_to_words(
        [
            word(0.1, 0.2, "first"),
            word(3.0, 3.0, "tied-gap"),
        ],
        [
            turn(0.0, 0.5, "SPEAKER_01"),
            turn(1.0, 2.0, "SPEAKER_00"),
            turn(4.0, 5.0, "SPEAKER_01"),
        ],
    )

    assert [item.speaker for item in aligned] == ["SPEAKER_01", "SPEAKER_01"]


def test_tied_intersection_without_sticky_candidate_uses_earlier_turn() -> None:
    aligned = assign_speakers_to_words(
        [word(0.5, 1.5, "tied")],
        [
            turn(1.0, 2.0, "SPEAKER_01"),
            turn(0.0, 1.0, "SPEAKER_00"),
        ],
    )

    assert aligned[0].speaker == "SPEAKER_00"


def test_intersections_within_tolerance_are_tied() -> None:
    aligned = assign_speakers_to_words(
        [
            word(0.1, 0.2, "first"),
            word(0.5, 1.5000005, "nearly-tied"),
        ],
        [
            turn(0.0, 0.3, "SPEAKER_01"),
            turn(0.4, 1.0, "SPEAKER_00"),
            turn(1.0, 2.0000005, "SPEAKER_01"),
        ],
    )

    assert [item.speaker for item in aligned] == ["SPEAKER_01", "SPEAKER_01"]


def test_zero_duration_word_prefers_containing_turn_over_midpoint() -> None:
    aligned = assign_speakers_to_words(
        [word(1.0, 1.0, "point")],
        [
            turn(2.0, 3.0, "SPEAKER_01"),
            turn(0.0, 100.0, "SPEAKER_00"),
        ],
    )

    assert aligned[0].speaker == "SPEAKER_00"


def test_zero_duration_containment_is_end_exclusive() -> None:
    aligned = assign_speakers_to_words(
        [word(1.0, 1.0, "boundary")],
        [
            turn(0.0, 1.0, "SPEAKER_00"),
            turn(1.0, 2.0, "SPEAKER_01"),
        ],
    )

    assert aligned[0].speaker == "SPEAKER_01"


def test_no_turns_uses_default_speaker_for_all_words() -> None:
    aligned = assign_speakers_to_words(
        [word(0.0, 0.2, "one"), word(0.3, 0.5, "two")],
        [],
    )

    assert [item.speaker for item in aligned] == ["SPEAKER_00", "SPEAKER_00"]


def test_empty_words_returns_empty_response() -> None:
    response = align_words([], [turn(0.0, 1.0, "SPEAKER_00")])

    assert response.model_dump() == {"speakers": [], "segments": []}


def test_unsorted_inputs_are_stably_normalized_before_segmentation() -> None:
    response = align_words(
        [
            word(10.0, 11.0, "later"),
            word(0.0, 1.0, "earlier"),
        ],
        [
            turn(9.0, 12.0, "SPEAKER_00"),
            turn(0.0, 2.0, "SPEAKER_00"),
        ],
    )

    assert [
        (
            segment.start,
            segment.end,
            [item.word for item in segment.words],
        )
        for segment in response.segments
    ] == [
        (0.0, 1.0, ["earlier"]),
        (10.0, 11.0, ["later"]),
    ]


def test_speaker_change_and_pause_force_separate_segments() -> None:
    segments = segment_aligned_words(
        [
            aligned_word(0.0, 0.2, "one", "SPEAKER_00"),
            aligned_word(0.3, 0.5, "two", "SPEAKER_01"),
            aligned_word(1.2, 1.4, "three", "SPEAKER_01"),
        ]
    )

    assert [(segment.id, segment.speaker, segment.text) for segment in segments] == [
        (0, "SPEAKER_00", "one"),
        (1, "SPEAKER_01", "two"),
        (2, "SPEAKER_01", "three"),
    ]


def test_same_speaker_run_reuses_maximum_span_rule() -> None:
    segments = segment_aligned_words(
        [
            aligned_word(0.0, 10.0, "one", "SPEAKER_00"),
            aligned_word(10.1, 20.0, "two", "SPEAKER_00"),
            aligned_word(20.1, 30.1, "three", "SPEAKER_00"),
        ]
    )

    assert [segment.text for segment in segments] == ["one two", "three"]


def test_alignment_performance_smoke() -> None:
    words = [
        word(index * 0.1, index * 0.1 + 0.05, str(index)) for index in range(10_000)
    ]
    turns = [
        turn(
            float(index),
            float(index + 1),
            f"SPEAKER_{index % 2:02d}",
        )
        for index in range(1_000)
    ]

    started_at = time.perf_counter()
    response = align_words(words, turns)
    elapsed = time.perf_counter() - started_at

    aligned_word_count = sum(len(segment.words) for segment in response.segments)
    assert aligned_word_count == 10_000
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("words", "turns", "message"),
    [
        (
            [word(math.inf, 1.0, "bad")],
            [],
            "words[0] has non-finite start or end values.",
        ),
        (
            [word(2.0, 1.0, "bad")],
            [],
            "words[0] has end before start.",
        ),
        (
            [],
            [turn(0.0, math.nan, "SPEAKER_00")],
            "turns[0] has non-finite start or end values.",
        ),
        (
            [],
            [turn(2.0, 1.0, "SPEAKER_00")],
            "turns[0] has end before start.",
        ),
    ],
)
def test_invalid_intervals_raise_clear_error(
    words: list[TranscriptionWord],
    turns: list[DiarizationTurn],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        align_words(words, turns)

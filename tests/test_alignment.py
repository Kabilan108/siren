import math
import time

import pytest

from siren.alignment import (
    SpeakerSmoothing,
    _ends_sentence,
    align_words,
    assign_speakers_to_words,
    get_speaker_smoothing,
    segment_aligned_words,
    smooth_speaker_runs,
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


def speakers(words: list[AlignedWord]) -> list[str]:
    return [item.speaker for item in words]


def utterance(
    start: float,
    texts: list[str],
    speaker: str,
    *,
    step: float = 0.3,
) -> list[AlignedWord]:
    return [
        aligned_word(start + index * step, start + index * step + 0.25, text, speaker)
        for index, text in enumerate(texts)
    ]


def test_smoothing_folds_fragment_after_unfinished_word() -> None:
    words = [
        *utterance(0.0, ["the", "methods", "you're", "talking"], "SPEAKER_00"),
        aligned_word(1.15, 1.39, "about,", "SPEAKER_01"),
        *utterance(3.47, ["but", "still", "accessible."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == ["SPEAKER_00"] * len(words)
    assert [item.word for item in smoothed] == [item.word for item in words]


def test_smoothing_folds_unfinished_fragment_that_runs_into_next_word() -> None:
    words = [
        *utterance(0.0, ["makes", "sense."], "SPEAKER_00"),
        aligned_word(0.8, 0.95, "I", "SPEAKER_01"),
        *utterance(0.95, ["know", "who", "this", "is."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == ["SPEAKER_00"] * len(words)


def test_smoothing_resolves_alternating_fragments_against_merged_turn() -> None:
    words = [
        *utterance(0.0, ["we", "could"], "SPEAKER_00"),
        aligned_word(0.6, 0.85, "make", "SPEAKER_01"),
        aligned_word(0.9, 1.15, "the", "SPEAKER_00"),
        aligned_word(1.2, 1.45, "whole", "SPEAKER_01"),
        *utterance(1.5, ["thing", "work."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == ["SPEAKER_00"] * len(words)


def test_smoothing_never_moves_surrounding_speaker_words() -> None:
    words = [
        *utterance(0.0, ["we", "could"], "SPEAKER_00"),
        aligned_word(0.6, 0.85, "make", "SPEAKER_01"),
        aligned_word(0.9, 1.15, "the", "SPEAKER_00"),
        aligned_word(1.2, 1.45, "whole", "SPEAKER_01"),
        *utterance(1.5, ["thing", "work."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert all(
        after.speaker == "SPEAKER_00"
        for before, after in zip(words, smoothed)
        if before.speaker == "SPEAKER_00"
    )


def test_smoothing_keeps_backchannel_after_finished_sentence() -> None:
    words = [
        *utterance(0.0, ["next", "fall,", "I'm", "assuming."], "SPEAKER_00"),
        aligned_word(1.31, 1.63, "Yeah.", "SPEAKER_01"),
        *utterance(1.95, ["And,", "you", "know,"], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_smoothing_keeps_rapid_punctuated_reply() -> None:
    words = [
        *utterance(0.0, ["Are", "you", "ready?"], "SPEAKER_00"),
        aligned_word(0.9, 1.1, "Yes.", "SPEAKER_01"),
        *utterance(1.15, ["Great,", "let's", "start."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_smoothing_keeps_interruption_set_off_by_pauses() -> None:
    words = [
        *utterance(0.0, ["so", "if", "we"], "SPEAKER_00"),
        aligned_word(3.0, 3.2, "no", "SPEAKER_01"),
        *utterance(5.5, ["okay", "then."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_smoothing_keeps_whole_sentence_interruption() -> None:
    words = [
        *utterance(0.0, ["and", "then", "if"], "SPEAKER_00"),
        aligned_word(0.9, 1.1, "No.", "SPEAKER_01"),
        *utterance(1.3, ["okay.", "Never", "mind."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_smoothing_pause_bound_protects_unfinished_fragment() -> None:
    words = [
        *utterance(0.0, ["rely", "on", "what", "it", "learned"], "SPEAKER_00"),
        aligned_word(2.09, 2.24, "to", "SPEAKER_01"),
        *utterance(2.88, ["perform", "well."], "SPEAKER_00"),
    ]

    assert speakers(smooth_speaker_runs(words, SpeakerSmoothing())) == speakers(
        words
    )
    assert speakers(
        smooth_speaker_runs(words, SpeakerSmoothing(max_pause=0.8))
    ) == ["SPEAKER_00"] * len(words)


def test_smoothing_treats_closing_quote_as_finished_sentence() -> None:
    words = [
        *utterance(0.0, ["he", "said", "\u201cwe\u2019re", "done.\u201d"], "SPEAKER_00"),
        aligned_word(1.3, 1.5, "right", "SPEAKER_01"),
        *utterance(2.2, ["So", "anyway."], "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


@pytest.mark.parametrize(
    ("text", "finished"),
    [
        ("done.", True),
        ("done.\u201d", True),
        ("why?\u2019", True),
        ("(really!)", True),
        ("wait\u2026", True),
        ("done", False),
        ("about,", False),
        ("\u201cquote", False),
    ],
)
def test_ends_sentence_handles_closing_marks(text: str, finished: bool) -> None:
    assert _ends_sentence(text) is finished


def test_smoothing_keeps_run_with_too_many_words() -> None:
    words = [
        aligned_word(0.0, 0.1, "a", "SPEAKER_00"),
        aligned_word(0.1, 0.2, "b", "SPEAKER_01"),
        aligned_word(0.2, 0.3, "c", "SPEAKER_01"),
        aligned_word(0.3, 0.4, "d", "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing(max_words=1))

    assert speakers(smoothed) == speakers(words)


def test_smoothing_keeps_run_longer_than_max_seconds() -> None:
    words = [
        aligned_word(0.0, 0.5, "well", "SPEAKER_00"),
        aligned_word(0.5, 2.0, "absolutely", "SPEAKER_01"),
        aligned_word(2.0, 2.5, "anyway", "SPEAKER_00"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing(max_seconds=1.0))

    assert speakers(smoothed) == speakers(words)


def test_smoothing_keeps_short_run_between_different_speakers() -> None:
    words = [
        aligned_word(0.0, 0.3, "done", "SPEAKER_00"),
        aligned_word(0.3, 0.5, "okay", "SPEAKER_01"),
        aligned_word(0.5, 0.8, "next", "SPEAKER_02"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_smoothing_leaves_edge_runs_alone() -> None:
    words = [
        aligned_word(0.0, 0.2, "hi", "SPEAKER_01"),
        aligned_word(0.2, 0.4, "so", "SPEAKER_00"),
        aligned_word(0.4, 0.6, "about", "SPEAKER_00"),
        aligned_word(0.6, 0.8, "that", "SPEAKER_00"),
        aligned_word(0.8, 1.0, "plan", "SPEAKER_00"),
        aligned_word(1.0, 1.2, "bye", "SPEAKER_01"),
    ]

    smoothed = smooth_speaker_runs(words, SpeakerSmoothing())

    assert speakers(smoothed) == speakers(words)


def test_align_words_smoothing_avoids_segment_split() -> None:
    words = [
        word(0.0, 0.4, "this"),
        word(0.45, 0.8, "is"),
        word(0.85, 1.1, "one"),
        word(1.15, 1.6, "sentence."),
    ]
    turns = [
        turn(0.0, 0.8, "SPEAKER_00"),
        turn(0.8, 1.12, "SPEAKER_01"),
        turn(1.12, 2.0, "SPEAKER_00"),
    ]

    unsmoothed = align_words(words, turns)
    smoothed = align_words(words, turns, smoothing=SpeakerSmoothing())

    assert [segment.speaker for segment in unsmoothed.segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert [(segment.speaker, segment.text) for segment in smoothed.segments] == [
        ("SPEAKER_00", "this is one sentence."),
    ]
    assert smoothed.speakers == ["SPEAKER_00"]


def test_get_speaker_smoothing_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIREN_SMOOTH_MAX_WORDS", "2")
    monkeypatch.setenv("SIREN_SMOOTH_MAX_SECONDS", "0.8")
    monkeypatch.setenv("SIREN_SMOOTH_MAX_PAUSE", "0.4")

    assert get_speaker_smoothing() == SpeakerSmoothing(
        max_words=2,
        max_seconds=0.8,
        max_pause=0.4,
    )


def test_get_speaker_smoothing_disabled_by_zero_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIREN_SMOOTH_MAX_WORDS", "0")

    assert get_speaker_smoothing() is None


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

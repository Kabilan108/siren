import pytest

from siren.schemas import TranscriptionWord
from siren.segmentation import segment_words


def word(start: float, end: float, text: str) -> TranscriptionWord:
    return TranscriptionWord(start=start, end=end, word=text)


@pytest.mark.parametrize(
    ("gap", "expected_segment_count"),
    [(0.6, 1), (0.6001, 2)],
)
def test_pause_split_uses_strict_threshold(
    gap: float,
    expected_segment_count: int,
) -> None:
    words = [word(0.0, 0.4, "one"), word(0.4 + gap, 1.3, "two")]

    assert len(segment_words(words, pause_threshold=0.6)) == expected_segment_count


def test_max_length_forces_split() -> None:
    words = [
        word(0.0, 10.0, "one"),
        word(10.1, 20.0, "two"),
        word(20.1, 30.1, "three"),
    ]

    segments = segment_words(words, max_segment_seconds=30.0)

    assert [segment.text for segment in segments] == ["one two", "three"]
    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 20.0),
        (20.1, 30.1),
    ]


def test_single_oversized_word_is_allowed() -> None:
    oversized = word(1.0, 32.0, "long")

    segments = segment_words([oversized], max_segment_seconds=30.0)

    assert len(segments) == 1
    assert segments[0].start == 1.0
    assert segments[0].end == 32.0
    assert segments[0].words == [oversized]


def test_empty_input_returns_no_segments() -> None:
    assert segment_words([]) == []


def test_words_are_never_lost_or_reordered() -> None:
    words = [
        word(0.0, 0.2, "one"),
        word(0.3, 0.5, "two"),
        word(1.2, 1.4, "three"),
        word(31.5, 31.8, "four"),
    ]

    segments = segment_words(words)
    segmented_words = [
        segmented_word
        for segment in segments
        for segmented_word in (segment.words or [])
    ]

    assert segmented_words == words
    assert all(
        actual is expected
        for actual, expected in zip(segmented_words, words, strict=True)
    )
    assert [segment.id for segment in segments] == list(range(len(segments)))


def test_segment_text_joins_words_with_single_spaces() -> None:
    words = [word(0.0, 0.2, " one "), word(0.3, 0.5, "two")]

    assert segment_words(words)[0].text == "one two"


def test_segment_end_covers_overlapping_word_ends():
    words = [
        TranscriptionWord(start=0.0, end=31.0, word="long"),
        TranscriptionWord(start=1.0, end=2.0, word="inside"),
    ]
    segments = segment_words(words)
    for segment in segments:
        assert segment.end == max(word.end for word in segment.words or [])
    all_words = [word for segment in segments for word in segment.words or []]
    assert [word.word for word in all_words] == ["long", "inside"]

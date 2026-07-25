from collections.abc import Sequence

from siren.schemas import TranscriptionSegment, TranscriptionWord


def segment_words(
    words: Sequence[TranscriptionWord],
    *,
    pause_threshold: float = 0.6,
    max_segment_seconds: float = 30.0,
) -> list[TranscriptionSegment]:
    if not words:
        return []

    segments: list[TranscriptionSegment] = []
    segment_words = [words[0]]

    segment_end = words[0].end

    for word in words[1:]:
        exceeds_pause_threshold = word.start - segment_end > pause_threshold
        exceeds_max_segment_seconds = (
            max(word.end, segment_end) - segment_words[0].start
            > max_segment_seconds
        )
        if exceeds_pause_threshold or exceeds_max_segment_seconds:
            segments.append(_build_segment(len(segments), segment_words))
            segment_words = [word]
            segment_end = word.end
        else:
            segment_words.append(word)
            segment_end = max(segment_end, word.end)

    segments.append(_build_segment(len(segments), segment_words))
    return segments


def _build_segment(
    segment_id: int,
    words: list[TranscriptionWord],
) -> TranscriptionSegment:
    return TranscriptionSegment(
        id=segment_id,
        start=words[0].start,
        end=max(word.end for word in words),
        text=" ".join(word.word.strip() for word in words),
        words=words,
    )

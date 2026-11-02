"""Validation for Parakeet boosting vocabularies (request field and env default)."""

import json
from collections.abc import Sequence

MAX_TERMS = 200
MAX_TERM_CHARS = 64


def normalize_vocabulary(terms: object, *, source: str = "vocabulary") -> list[str]:
    """Trim terms and drop case-insensitive duplicates, keeping the first spelling."""
    if not isinstance(terms, list):
        raise ValueError(f"{source} must be a JSON array of strings")
    if len(terms) > MAX_TERMS:
        raise ValueError(
            f"{source} accepts at most {MAX_TERMS} terms; got {len(terms)}"
        )
    seen: set[str] = set()
    normalized: list[str] = []
    for index, term in enumerate(terms):
        if not isinstance(term, str):
            raise ValueError(f"{source}[{index}] must be a string")
        trimmed = term.strip()
        if not 1 <= len(trimmed) <= MAX_TERM_CHARS:
            raise ValueError(
                f"{source}[{index}] must be 1-{MAX_TERM_CHARS} characters after trimming"
            )
        if any(not character.isprintable() for character in trimmed):
            raise ValueError(f"{source}[{index}] must not contain control characters")
        folded = trimmed.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(trimmed)
    return normalized


def parse_vocabulary(raw: str, *, source: str = "vocabulary") -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must be a JSON array of strings ({exc.msg})")
    return normalize_vocabulary(value, source=source)


def vocabulary_key(terms: Sequence[str]) -> tuple[str, ...]:
    """Cache key for an already-normalized list; the boosting tree ignores order."""
    return tuple(sorted(terms))

"""Reference-based WER/CER helpers for reproducible ASR evaluation."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorRateScore:
    """Edit counts for one recording or a corpus aggregate."""

    word_errors: int
    reference_words: int
    character_errors: int
    reference_characters: int

    @property
    def wer(self) -> float:
        return self.word_errors / self.reference_words

    @property
    def cer(self) -> float:
        return self.character_errors / self.reference_characters


def normalize_text(text: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace without removing accents."""
    text = unicodedata.normalize("NFC", text).casefold()
    normalized: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isspace():
            normalized.append(" ")
        elif category[0] in {"L", "M", "N"}:
            normalized.append(character)
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def score_text(reference: str, hypothesis: str) -> ErrorRateScore:
    """Score a transcript pair; WER/CER are ratios, not percentages."""
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    if not reference_words:
        raise ValueError("Reference is empty after normalization; WER/CER are undefined.")

    hypothesis_words = normalized_hypothesis.split()
    reference_characters = list(normalized_reference)
    hypothesis_characters = list(normalized_hypothesis)
    return ErrorRateScore(
        word_errors=_edit_distance(reference_words, hypothesis_words),
        reference_words=len(reference_words),
        character_errors=_edit_distance(reference_characters, hypothesis_characters),
        reference_characters=len(reference_characters),
    )


def aggregate_scores(scores: Iterable[ErrorRateScore]) -> ErrorRateScore:
    """Compute corpus-level (micro-averaged) WER/CER from total edit counts."""
    items = list(scores)
    if not items:
        raise ValueError("Cannot aggregate an empty collection of transcript scores.")
    return ErrorRateScore(
        word_errors=sum(item.word_errors for item in items),
        reference_words=sum(item.reference_words for item in items),
        character_errors=sum(item.character_errors for item in items),
        reference_characters=sum(item.reference_characters for item in items),
    )

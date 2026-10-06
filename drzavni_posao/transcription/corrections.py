from __future__ import annotations

import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .constants import CORE_CHARACTERS
from .transcript import Transcript


class CrossSegmentCorrectionError(ValueError):
    """Raised when a corrected turn cannot retain its source segment count."""


@dataclass(slots=True)
class CorrectionStats:
    applied: int = 0
    skipped: int = 0
    ignored_confidence: int = 0
    changed_segments: int = 0

    def add(self, other: CorrectionStats) -> None:
        self.applied += other.applied
        self.skipped += other.skipped
        self.ignored_confidence += other.ignored_confidence
        self.changed_segments += other.changed_segments


@dataclass(slots=True)
class _Turn:
    speaker: str
    segment_indices: list[int]


def _normalize_text(text: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


def _core_turns(
    segments: list[dict[str, Any]],
    working_texts: list[str],
) -> list[_Turn]:
    """Match generation.dataset's filter-then-collapse turn construction."""
    turns: list[_Turn] = []
    for index, segment in enumerate(segments):
        if not working_texts[index]:
            continue
        speaker = segment.get("speaker")
        if speaker not in CORE_CHARACTERS:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1].segment_indices.append(index)
        else:
            turns.append(_Turn(str(speaker), [index]))
    return turns


def _turn_text(turn: _Turn, working_texts: list[str]) -> str:
    return " ".join(working_texts[index] for index in turn.segment_indices)


def _project_turn_text(
    turn: _Turn,
    corrected_text: str,
    working_texts: list[str],
    *,
    transcript_file: str | None,
) -> set[int]:
    """Redistribute a corrected turn using the source segments' word counts."""
    corrected_text = _normalize_text(corrected_text)
    tokens = corrected_text.split()
    indices = turn.segment_indices
    if len(tokens) < len(indices):
        raise CrossSegmentCorrectionError(
            f"Corrected turn for {transcript_file!r} has fewer words than its "
            "timestamped source segments"
        )

    changed: set[int] = set()
    cursor = 0
    for position, segment_index in enumerate(indices):
        if position == len(indices) - 1:
            projected = " ".join(tokens[cursor:])
        else:
            remaining_segments = len(indices) - position - 1
            available = len(tokens) - cursor - remaining_segments
            source_word_count = len(working_texts[segment_index].split())
            width = max(1, min(source_word_count, available))
            projected = " ".join(tokens[cursor : cursor + width])
            cursor += width
        if projected != working_texts[segment_index]:
            working_texts[segment_index] = projected
            changed.add(segment_index)

    if _turn_text(turn, working_texts) != corrected_text:
        raise AssertionError("Corrected turn projection changed its text")
    return changed


def apply_episode_corrections(
    transcript: Transcript,
    edits: list[dict[str, Any]],
    *,
    confidences: frozenset[str] = frozenset({"high"}),
) -> tuple[Transcript, CorrectionStats]:
    """Apply generation-style turn edits while preserving source timestamps.

    Corrections are evaluated against core-character turns after filtering and
    collapsing, exactly as in the screenplay generation notebook. Corrected
    turns are redistributed over their original timestamped source segments
    according to the original word counts.
    """
    segments = deepcopy(transcript.segments)
    working_texts = [_normalize_text(segment.get("text")) for segment in segments]
    turns = _core_turns(segments, working_texts)
    turn_texts = [_turn_text(turn, working_texts) for turn in turns]
    stats = CorrectionStats()

    for edit in edits:
        if edit.get("confidence") not in confidences:
            stats.ignored_confidence += 1
            continue

        turn_id = edit.get("turn_id")
        original = edit.get("original")
        replacement = edit.get("replacement")
        if (
            type(turn_id) is not int
            or turn_id < 0
            or turn_id >= len(turns)
            or not isinstance(original, str)
            or not original
            or not isinstance(replacement, str)
            or not replacement
            or original == replacement
        ):
            stats.skipped += 1
            continue

        before = turn_texts[turn_id]
        if before.count(original) != 1:
            stats.skipped += 1
            continue

        turn_texts[turn_id] = before.replace(original, replacement, 1)
        stats.applied += 1

    changed_indices: set[int] = set()
    for turn, corrected_text in zip(turns, turn_texts):
        if corrected_text == _turn_text(turn, working_texts):
            continue
        changed_indices.update(
            _project_turn_text(
                turn,
                corrected_text,
                working_texts,
                transcript_file=transcript.file,
            )
        )

    for index in changed_indices:
        segments[index]["text"] = working_texts[index]
        # Corrected text no longer has trustworthy word-level ASR alignment.
        segments[index].pop("words", None)
    stats.changed_segments = len(changed_indices)
    return Transcript(segments, file=transcript.file), stats

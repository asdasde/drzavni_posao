import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from ..transcription.constants import ANONYMOUS_LABEL, CORE_CHARACTERS
from ..transcription.episode import Episode
from ..transcription.transcript import Transcript


SPLITS = ("train", "validation", "test")
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_DIGIT_RUN_RE = re.compile(r"\d+")


class TranscriptQualityError(ValueError):
    def __init__(self, episode_key: str, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__(
            f"Episode {episode_key!r} failed transcript quality checks: "
            f"{', '.join(reasons)}"
        )


def build_pair(
    episode: Episode,
    premise: str,
    *,
    check_quality: bool = True,
) -> dict[str, str]:
    if episode.transcript is None:
        raise ValueError(f"Episode {episode.key!r} has no transcript")

    premise = premise.strip()
    if not premise:
        raise ValueError("Premise cannot be empty")

    cleaned, issues = _prepare_transcript(episode.transcript)
    if check_quality and issues:
        raise TranscriptQualityError(episode.key, issues)

    dialogue = cleaned.collapse_speaker_turns().render()
    if not dialogue:
        raise ValueError(f"Episode {episode.key!r} has no dialogue")

    return {
        "episode_key": episode.key,
        "premise": premise,
        "dialogue": dialogue,
    }


def _prepare_transcript(transcript: Transcript) -> tuple[Transcript, list[str]]:
    source = Transcript(
        [
            {**segment, "text": _normalize_text(segment.get("text"))}
            for segment in transcript.segments
            if _normalize_text(segment.get("text"))
        ],
        file=transcript.file,
    )
    cleaned = Transcript(
        [
            segment.copy()
            for segment in source.segments
            if segment.get("speaker") in CORE_CHARACTERS
        ],
        file=transcript.file,
    )
    return cleaned, _find_transcript_quality_issues(source, cleaned)


def _find_transcript_quality_issues(
    source: Transcript,
    transcript: Transcript,
) -> list[str]:
    segments = transcript.segments
    texts = [_normalize_text(segment["text"]) for segment in segments]
    segment_words = [_words(text) for text in texts]
    all_words = [word for words in segment_words for word in words]
    issues = []

    if len(all_words) < 400:
        issues.append("short_transcript")
    if len({segment.get("speaker") for segment in segments}) < 2:
        issues.append("fewer_than_two_core_speakers")

    ngrams = [
        tuple(all_words[index : index + 4])
        for index in range(max(0, len(all_words) - 3))
    ]
    counts = Counter(ngrams)
    repeated_excess = sum(count - 1 for count in counts.values() if count > 1)
    if repeated_excess / max(1, len(ngrams)) > 0.06:
        issues.append("repeated_4grams_gt_6pct")

    adjacent_duplicates = sum(
        left.casefold() == right.casefold() for left, right in zip(texts, texts[1:])
    )
    if adjacent_duplicates / max(1, len(texts) - 1) > 0.10:
        issues.append("adjacent_duplicates_gt_10pct")

    collapsed = transcript.collapse_speaker_turns()
    if (
        max((len(_words(segment["text"])) for segment in collapsed.segments), default=0)
        > 300
    ):
        issues.append("single_turn_gt_300_words")

    full_text = "\n".join(texts)
    nonspace = [character for character in full_text if not character.isspace()]
    if (
        sum(character.isalpha() for character in nonspace) / max(1, len(nonspace))
        < 0.75
    ):
        issues.append("alphabetic_content_lt_75pct")
    if (
        max(
            (len(match.group()) for match in _DIGIT_RUN_RE.finditer(full_text)),
            default=0,
        )
        >= 10
    ):
        issues.append("digit_run_ge_10")

    starts = [
        segment.get("start")
        for segment in segments
        if isinstance(segment.get("start"), (int, float))
    ]
    ends = [
        segment.get("end")
        for segment in segments
        if isinstance(segment.get("end"), (int, float))
    ]
    if starts and ends and max(ends) > min(starts):
        words_per_minute = len(all_words) / ((max(ends) - min(starts)) / 60)
        if words_per_minute < 80 or words_per_minute > 250:
            issues.append("speech_rate_outside_80_250")

    total_words = sum(len(_words(segment["text"])) for segment in source.segments)
    unknown_words = sum(
        len(_words(segment["text"]))
        for segment in source.segments
        if segment.get("speaker") not in CORE_CHARACTERS
        and segment.get("speaker") != ANONYMOUS_LABEL
    )
    if total_words and unknown_words / total_words > 0.02:
        issues.append("unknown_speakers_gt_2pct")

    return issues


def _normalize_text(text: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_normalize_text(text).lower())


def assign_folds(
    episodes: Iterable[Episode],
    fold_count: int = 10,
) -> dict[str, int]:
    if fold_count < 3:
        raise ValueError("At least three folds are required")

    ordered = sorted(
        episodes,
        key=lambda episode: (episode.metadata.date, episode.key),
    )
    keys = [episode.key for episode in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode keys")

    return {episode.key: index % fold_count for index, episode in enumerate(ordered)}


def select_splits(
    fold_assignments: dict[str, int],
    test_fold: int,
    validation_fold: int,
) -> dict[str, str]:
    if test_fold == validation_fold:
        raise ValueError("Test and validation folds must differ")

    return {
        episode_key: (
            "test"
            if fold == test_fold
            else "validation"
            if fold == validation_fold
            else "train"
        )
        for episode_key, fold in fold_assignments.items()
    }

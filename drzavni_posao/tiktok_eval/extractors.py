"""High-precision extraction helpers for the organic TikTok eval dataset.

The functions in this module deliberately prefer skipping a post over assigning
an uncertain label.  They do not fetch TikTok, run OCR, or depend on collector
models; callers pass caption/OCR text in and receive small, serialisable result
objects back.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping


_DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "youtube_episodes.json"
)
_EPISODE_MARKER_RE = re.compile(
    r"(?<!\w)(?:ep\.?|epizoda(?:\s+br\.?)?|еп\.?|епизода(?:\s+бр\.?)?)"
    r"\s*[:#-]?\s*(\d{1,4})(?!\d)",
    re.IGNORECASE,
)
_REPLY_PREFIX_RE = re.compile(
    r"^.{0,80}?(?:(?:replying|reply)\s+to\s+@?[\w.-]+|"
    r"odgovor(?:i)?\s+na\s+komentar(?:\s+korisnika)?\s+@?\S+|"
    r"одговор(?:и)?\s+на\s+коментар(?:\s+корисника)?\s+@?\S+)"
    r"(?:['’]s\s+comment)?\s*[:\-–—]?\s*",
    re.IGNORECASE | re.DOTALL,
)
_REPLY_HEADER_LINE_RE = re.compile(
    r"^\s*(?:(?:replying|reply)\s+to\b.*|"
    r"odgovor(?:i)?\s+na\s+komentar\b.*|"
    r"одговор(?:и)?\s+на\s+коментар\b.*)\s*$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\S+")
_HASHTAG_RE = re.compile(r"(?<!\w)#\S+")
_STICKER_UI_RE = re.compile(r"\[\s*(?:sticker|nalepnica|налепница)\s*\]", re.IGNORECASE)
_QUERY_CUE_RE = re.compile(
    r"(?<!\w)(?:"
    r"kako\s+se\s+zove|како\s+се\s+зове|"
    r"(?:koja|која)\s+(?:(?:je|је)\s+)?(?:epizoda|епизода)|"
    r"(?:ona|onaj|оно|она|онај)\s+(?:epizoda|епизода)|"
    r"(?:u\s+kojoj\s+epizodi|у\s+којој\s+епизоди)|"
    r"(?:epizod[aei]|епизод[аеи])\s+"
    r"(?:kad(?:a)?|gde|gdje|sa|u\s+kojoj|кад(?:а)?|где|са|у\s+којој)"
    r")",
    re.IGNORECASE,
)

# Fraction of a typical frame's words a stable span must retain to be preferred
# over the full text of the most central frame.
_STABLE_SPAN_COVERAGE = 0.75

_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "ђ": "dj",
        "е": "e",
        "ж": "z",
        "з": "z",
        "и": "i",
        "ј": "j",
        "к": "k",
        "л": "l",
        "љ": "lj",
        "м": "m",
        "н": "n",
        "њ": "nj",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "ћ": "c",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "c",
        "џ": "dz",
        "ш": "s",
    }
)


@dataclass(frozen=True)
class Episode:
    episode_number: int | None
    title: str
    youtube_video_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Episode":
        number = value.get("episode_number")
        return cls(
            episode_number=number if isinstance(number, int) else None,
            title=str(value.get("title", "")).strip(),
            youtube_video_id=str(value.get("youtube_video_id", "")).strip(),
        )


class EpisodeCatalog:
    """Validated lookup indexes over ``youtube_episodes.json`` records."""

    def __init__(self, episodes: Iterable[Episode | Mapping[str, object]]) -> None:
        parsed = [
            item if isinstance(item, Episode) else Episode.from_mapping(item)
            for item in episodes
        ]
        self.episodes = tuple(e for e in parsed if e.title and e.youtube_video_id)
        self.by_number = {
            e.episode_number: e for e in self.episodes if e.episode_number is not None
        }
        by_title: dict[str, list[Episode]] = defaultdict(list)
        for episode in self.episodes:
            by_title[_normalise(episode.title)].append(episode)
        self.by_title = {key: tuple(value) for key, value in by_title.items() if key}

    @classmethod
    def from_json(cls, path: str | Path = _DEFAULT_CATALOG_PATH) -> "EpisodeCatalog":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = payload.get("episodes") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Episode catalog must contain an 'episodes' list")
        return cls(records)


@dataclass(frozen=True)
class EpisodeExtractionResult:
    status: str
    episode_number: int | None = None
    youtube_video_id: str | None = None
    title: str | None = None
    confidence: float = 0.0
    method: str | None = None
    reason: str | None = None

    @property
    def episode_id(self) -> str | None:
        """Collector-friendly alias for the canonical YouTube video id."""

        return self.youtube_video_id

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class QueryExtractionResult:
    status: str
    text: str | None = None
    confidence: float = 0.0
    method: str | None = None
    raw_candidates: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold()).translate(_CYRILLIC_TO_LATIN)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _clean_caption(caption: str) -> str:
    caption = _REPLY_PREFIX_RE.sub("", caption.strip())
    caption = _URL_RE.sub(" ", caption)
    caption = _MENTION_RE.sub(" ", caption)
    caption = _HASHTAG_RE.sub(" ", caption)
    caption = _EPISODE_MARKER_RE.sub(" ", caption)
    return " ".join(caption.split()).strip(" :-–—|.,")


def _title_match_groups(
    caption: str,
    catalog: EpisodeCatalog,
    *,
    fuzzy_threshold: float,
    fuzzy_margin: float,
) -> tuple[tuple[Episode, ...] | None, str | None, float]:
    cleaned = _normalise(_clean_caption(caption))
    if not cleaned:
        return None, None, 0.0

    exact: list[tuple[int, str, tuple[Episode, ...]]] = []
    padded_caption = f" {cleaned} "
    for title, episodes in catalog.by_title.items():
        # Very short/common titles must comprise the whole useful caption.
        if len(title) < 6 and cleaned != title:
            continue
        if f" {title} " in padded_caption:
            exact.append((len(title), title, episodes))
    if exact:
        longest = max(item[0] for item in exact)
        best = [item for item in exact if item[0] == longest]
        distinct_titles = {item[1] for item in best}
        if len(distinct_titles) != 1:
            return None, "ambiguous_title", 0.0
        return best[0][2], "title_exact", 0.98

    # Fuzzy matching is intentionally conservative: a typo-tolerant match is
    # accepted only when the best title is both strong and clearly separated.
    scores: list[tuple[float, str, tuple[Episode, ...]]] = []
    caption_tokens = cleaned.split()
    for title, episodes in catalog.by_title.items():
        if len(title) < 6:
            continue
        title_tokens = title.split()
        windows = [cleaned]
        if len(caption_tokens) > len(title_tokens):
            windows.extend(
                " ".join(caption_tokens[index : index + len(title_tokens)])
                for index in range(len(caption_tokens) - len(title_tokens) + 1)
            )
        score = max(SequenceMatcher(None, title, window).ratio() for window in windows)
        scores.append((score, title, episodes))
    scores.sort(key=lambda item: item[0], reverse=True)
    if not scores or scores[0][0] < fuzzy_threshold:
        return None, None, 0.0
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    if scores[0][0] - runner_up < fuzzy_margin:
        return None, "ambiguous_title", 0.0
    return scores[0][2], "title_fuzzy", min(0.95, scores[0][0])


def extract_episode(
    caption: str,
    catalog: EpisodeCatalog | str | Path | None = None,
    *,
    fuzzy_threshold: float = 0.92,
    fuzzy_margin: float = 0.06,
) -> EpisodeExtractionResult:
    """Extract and validate an episode label from a TikTok caption.

    Explicit episode markers always take precedence.  An unknown explicit
    number is rejected rather than rescued by title matching.  If both number
    and a confidently recognised title occur, they must describe the same
    catalogue record.
    """

    if not isinstance(caption, str) or not caption.strip():
        return EpisodeExtractionResult("not_found", reason="empty_caption")
    if catalog is None:
        catalog = EpisodeCatalog.from_json()
    elif not isinstance(catalog, EpisodeCatalog):
        catalog = EpisodeCatalog.from_json(catalog)

    numbers = {int(match) for match in _EPISODE_MARKER_RE.findall(caption)}
    if len(numbers) > 1:
        return EpisodeExtractionResult("conflict", reason="multiple_episode_numbers")

    title_group, title_method, title_confidence = _title_match_groups(
        caption,
        catalog,
        fuzzy_threshold=fuzzy_threshold,
        fuzzy_margin=fuzzy_margin,
    )

    if numbers:
        number = next(iter(numbers))
        episode = catalog.by_number.get(number)
        if episode is None:
            return EpisodeExtractionResult(
                "not_in_catalog",
                episode_number=number,
                method="episode_number",
                reason="episode_number_not_in_catalog",
            )
        cleaned_caption = _normalise(_clean_caption(caption))
        expected_title = _normalise(episode.title)
        expected_title_present = bool(
            expected_title and f" {expected_title} " in f" {cleaned_caption} "
        )
        if (
            title_group is not None
            and episode not in title_group
            and not expected_title_present
        ):
            return EpisodeExtractionResult(
                "conflict",
                episode_number=number,
                method="number_and_title",
                reason="episode_number_title_mismatch",
            )
        title_confirmed = expected_title_present or (
            title_group is not None and episode in title_group
        )
        method = "number_and_title" if title_confirmed else "episode_number"
        confidence = 1.0 if title_confirmed else 0.99
        return EpisodeExtractionResult(
            "ok",
            episode_number=episode.episode_number,
            youtube_video_id=episode.youtube_video_id,
            title=episode.title,
            confidence=confidence,
            method=method,
        )

    if title_group is None:
        status = "ambiguous" if title_method == "ambiguous_title" else "not_found"
        return EpisodeExtractionResult(
            status, reason=title_method or "episode_not_found"
        )
    if len(title_group) != 1:
        return EpisodeExtractionResult(
            "ambiguous",
            method=title_method,
            reason="title_maps_to_multiple_episodes",
        )
    episode = title_group[0]
    return EpisodeExtractionResult(
        "ok",
        episode_number=episode.episode_number,
        youtube_video_id=episode.youtube_video_id,
        title=episode.title,
        confidence=title_confidence,
        method=title_method,
    )


def _strip_reply_ui(text: str) -> str:
    # OCR sometimes collapses the sticker header and the comment onto one line.
    # Strip only a recognised prefix and keep the original comment text intact.
    text = _REPLY_PREFIX_RE.sub("", text.strip())
    text = _STICKER_UI_RE.sub(" ", text)
    text = re.sub(r"^[\s@=*_~|:;.,(){}\[\]<>-]{1,12}(?=\w)", "", text)
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    while lines and (not lines[0] or _REPLY_HEADER_LINE_RE.fullmatch(lines[0])):
        lines.pop(0)
    return " ".join(line for line in lines if line).strip()


def _plausible_query(text: str, *, min_chars: int, min_words: int) -> bool:
    words = [token for token in re.findall(r"[^\W\d_]+", text, re.UNICODE) if token]
    return len(text) >= min_chars and len(words) >= min_words


def _focus_query_candidate(text: str) -> str | None:
    """Trim changing full-frame OCR around an explicit episode-query cue."""

    cue = _QUERY_CUE_RE.search(text)
    if cue is None:
        return None
    end = len(text)
    for terminator in ("?", "？"):
        index = text.find(terminator, cue.start())
        if index >= 0:
            end = min(end, index + 1)
    return text[cue.start() : end].strip()


def _token_spans(text: str) -> tuple[list[str], list[re.Match[str]]]:
    matches = list(re.finditer(r"[^\W_]+", text, re.UNICODE))
    return [_normalise(match.group()) for match in matches], matches


def _contains_tokens(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    length = len(phrase)
    return any(
        tuple(tokens[index : index + length]) == phrase
        for index in range(len(tokens) - length + 1)
    )


def _stable_query_substring(
    candidates: tuple[str, ...],
    *,
    min_words: int,
) -> tuple[str, int] | None:
    """Find a majority-stable token span while retaining one frame's raw text."""

    if len(candidates) < 2:
        return None
    tokenised = [_token_spans(candidate) for candidate in candidates]
    required_support = max(2, math.ceil(len(candidates) * 0.6))
    # Rank support before length. This prevents a long accidental region shared
    # by two frames from beating the actual query present in every frame.
    best: tuple[int, int, int, int] | None = None
    seen: set[tuple[str, ...]] = set()
    for candidate_index, (tokens, _matches) in enumerate(tokenised):
        for length in range(len(tokens), min_words - 1, -1):
            for start in range(len(tokens) - length + 1):
                phrase = tuple(tokens[start : start + length])
                if phrase in seen or not all(phrase):
                    continue
                seen.add(phrase)
                support = sum(
                    _contains_tokens(other_tokens, phrase)
                    for other_tokens, _ in tokenised
                )
                if support < required_support:
                    continue
                contender = (support, length, -candidate_index, -start)
                current = (
                    (best[0], best[1], -best[2], -best[3]) if best is not None else None
                )
                if current is None or contender > current:
                    best = (support, length, candidate_index, start)
    if best is None:
        return None

    support, length, candidate_index, start = best
    candidate = candidates[candidate_index]
    _tokens, matches = tokenised[candidate_index]
    substring_start = matches[start].start()
    substring_end = matches[start + length - 1].end()
    next_token = matches[start + length] if start + length < len(matches) else None
    punctuation_end = next_token.start() if next_token is not None else len(candidate)
    punctuation = candidate[substring_end:punctuation_end]
    if punctuation.strip() and not any(char.isalnum() for char in punctuation):
        substring_end += len(punctuation.rstrip())
    return candidate[substring_start:substring_end].strip(), support


def extract_query(
    frame_texts: Iterable[str],
    *,
    agreement_threshold: float = 0.82,
    accept_confidence: float = 0.78,
    min_chars: int = 8,
    min_words: int = 1,
) -> QueryExtractionResult:
    """Build a conservative query consensus from OCR text across video frames.

    The selected output is an original OCR candidate (not corrected or
    transliterated), so casing, script, spelling mistakes and missing accents
    are preserved.  A single usable frame is returned as ``low_confidence``.

    A viewer's comment can be a single distinctive word — an episode name, or a
    catchphrase — so length rather than word count separates a real query from
    interface text: "Halkidiki" is a query, "Ep. 23" and "super" are not.
    """

    candidates = tuple(
        cleaned
        for raw in frame_texts
        if isinstance(raw, str) and (cleaned := _strip_reply_ui(raw))
    )
    valid = tuple(
        candidate
        for candidate in candidates
        if _plausible_query(candidate, min_chars=min_chars, min_words=min_words)
    )
    if not valid:
        return QueryExtractionResult(
            "not_found",
            raw_candidates=candidates,
            reason="no_plausible_ocr_candidate",
        )

    if len(valid) == 1:
        return QueryExtractionResult(
            "low_confidence",
            text=valid[0],
            confidence=0.65,
            method="single_frame_ocr",
            raw_candidates=candidates,
            reason="consensus_requires_multiple_frames",
        )

    focused = tuple(
        focused_text
        for candidate in valid
        if (focused_text := _focus_query_candidate(candidate)) is not None
        and _plausible_query(
            focused_text,
            min_chars=min_chars,
            min_words=min_words,
        )
    )
    required_focus_support = max(2, math.ceil(len(valid) * 0.6))
    if len(focused) >= required_focus_support:
        focused_normalised = [_normalise(candidate) for candidate in focused]
        similarities = [
            [SequenceMatcher(None, left, right).ratio() for right in focused_normalised]
            for left in focused_normalised
        ]
        medoid_index = max(
            range(len(focused)),
            key=lambda index: sum(similarities[index]) / len(focused),
        )
        agreement = [
            index
            for index, similarity in enumerate(similarities[medoid_index])
            if similarity >= agreement_threshold
        ]
        support_ratio = len(agreement) / len(valid)
        mean_similarity = sum(
            similarities[medoid_index][index] for index in agreement
        ) / len(agreement)
        confidence = min(1.0, 0.55 * support_ratio + 0.45 * mean_similarity)
        status = (
            "ok"
            if confidence >= accept_confidence and len(agreement) >= 2
            else "low_confidence"
        )
        return QueryExtractionResult(
            status,
            text=focused[medoid_index],
            confidence=round(confidence, 4),
            method="query_cue_consensus",
            raw_candidates=candidates,
            reason=None if status == "ok" else "insufficient_frame_agreement",
        )

    stable = _stable_query_substring(valid, min_words=max(min_words, 6))
    if stable is not None:
        stable_text, support = stable
        stable_words = len(re.findall(r"[^\W_]+", stable_text, re.UNICODE))
        original_words = [
            len(re.findall(r"[^\W_]+", candidate, re.UNICODE)) for candidate in valid
        ]
        # The stable span must be contiguous, so a single word that OCR drops
        # in one frame splits the query and leaves only the longer half. Trim
        # surrounding noise, but decline when most of the frame would be lost
        # and let whole-frame consensus keep the complete text instead.
        minimum_coverage = math.ceil(
            sorted(original_words)[len(original_words) // 2] * _STABLE_SPAN_COVERAGE
        )
        if (
            stable_words < max(original_words)
            and stable_words >= minimum_coverage
            and _plausible_query(stable_text, min_chars=min_chars, min_words=min_words)
        ):
            confidence = min(1.0, 0.55 * (support / len(valid)) + 0.45)
            status = "ok" if confidence >= accept_confidence else "low_confidence"
            return QueryExtractionResult(
                status,
                text=stable_text,
                confidence=round(confidence, 4),
                method="stable_substring_consensus",
                raw_candidates=candidates,
                reason=None if status == "ok" else "insufficient_frame_agreement",
            )

    normalised = [_normalise(candidate) for candidate in valid]
    similarities = [
        [SequenceMatcher(None, left, right).ratio() for right in normalised]
        for left in normalised
    ]
    medoid_index = max(
        range(len(valid)),
        key=lambda index: sum(similarities[index]) / len(valid),
    )
    agreement = [
        index
        for index, similarity in enumerate(similarities[medoid_index])
        if similarity >= agreement_threshold
    ]
    support_ratio = len(agreement) / len(valid)
    mean_similarity = sum(
        similarities[medoid_index][index] for index in agreement
    ) / len(agreement)
    confidence = min(1.0, 0.55 * support_ratio + 0.45 * mean_similarity)
    status = (
        "ok"
        if confidence >= accept_confidence and len(agreement) >= 2
        else "low_confidence"
    )
    return QueryExtractionResult(
        status,
        text=valid[medoid_index],
        confidence=round(confidence, 4),
        method="multi_frame_consensus",
        raw_candidates=candidates,
        reason=None if status == "ok" else "insufficient_frame_agreement",
    )


__all__ = [
    "Episode",
    "EpisodeCatalog",
    "EpisodeExtractionResult",
    "QueryExtractionResult",
    "extract_episode",
    "extract_query",
]

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..search.common.io import read_jsonl, write_jsonl
from .extractors import EpisodeCatalog
from .models import ProcessingRecord, ProcessingStatus
from .state import JsonlStateStore


@dataclass(frozen=True, slots=True)
class ExportSummary:
    input_records: int
    unique_posts: int
    accepted_posts: int
    exported: int
    skipped: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    admitted_below_query_bar: int = 0


@dataclass(frozen=True, slots=True)
class ExportResult:
    rows: tuple[dict[str, Any], ...]
    summary: ExportSummary


def detect_serbian_script(text: str) -> str:
    """Return a stable language tag for Serbian Latin or Cyrillic query text."""
    cyrillic = sum("\u0400" <= character <= "\u04ff" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    return "sr-Cyrl" if cyrillic > latin else "sr-Latn"


def _youtube_video_id_from_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").casefold()
    if host == "youtu.be":
        return parsed.path.strip("/").split("/", 1)[0] or None
    if host not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }:
        return None
    if parsed.path.rstrip("/") == "/watch":
        return (parse_qs(parsed.query).get("v") or [None])[0]
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
        return parts[1] or None
    return None


def corpus_video_ids_from_segments(path: Path | str) -> set[str]:
    video_ids: set[str] = set()
    for line_number, row in enumerate(read_jsonl(Path(path)), start=1):
        video_id = str(row.get("youtube_video_id") or "").strip()
        if not video_id:
            video_id = _youtube_video_id_from_url(row.get("youtube_url")) or ""
        if not video_id:
            raise ValueError(
                f"Corpus segment on line {line_number} has no YouTube video ID"
            )
        video_ids.add(video_id)
    return video_ids


def _episode_labels(
    catalog: EpisodeCatalog | Path | str | None,
) -> dict[str, dict[str, Any]]:
    """Map YouTube video IDs to human-readable episode number and title."""
    if catalog is None:
        return {}
    if not isinstance(catalog, EpisodeCatalog):
        catalog = EpisodeCatalog.from_json(catalog)
    labels: dict[str, dict[str, Any]] = {}
    for episode in catalog.episodes:
        labels[episode.youtube_video_id] = {
            "episode_number": episode.episode_number,
            "episode_title": episode.title,
        }
    return labels


def _clears_query_bar(
    record: ProcessingRecord, min_query_confidence: float | None
) -> bool:
    """Allow a held-back post through when its query is good enough anyway."""
    if min_query_confidence is None:
        return False
    if record.status is not ProcessingStatus.QUERY_LOW_CONFIDENCE:
        return False
    if record.query is None or record.query.confidence < min_query_confidence:
        return False
    return record.episode is not None and bool(record.episode.youtube_video_id)


def build_episode_eval_export(
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
    corpus_video_ids: Set[str] | None = None,
    segments_path: Path | str | None = None,
    min_query_confidence: float | None = None,
) -> ExportResult:
    """Build deterministic episode-level eval rows from append-only collector state.

    ``min_query_confidence`` additionally admits posts the collector held back
    for a weak query, when the query still clears the given bar and the episode
    was matched. A stored query's confidence does not depend on the collector's
    threshold, so re-deciding here avoids re-downloading a whole profile to
    change where the line sits.
    """
    if corpus_video_ids is not None and segments_path is not None:
        raise ValueError("Provide corpus_video_ids or segments_path, not both")
    coverage = (
        corpus_video_ids_from_segments(segments_path)
        if segments_path is not None
        else corpus_video_ids
    )
    records = list(source.records() if isinstance(source, JsonlStateStore) else source)
    latest: dict[str, ProcessingRecord] = {}
    for record in records:
        # Duplicate discovery entries are audit events, not new processing outcomes.
        if record.status is not ProcessingStatus.DUPLICATE:
            latest[record.post_id] = record

    rows: list[dict[str, Any]] = []
    accepted_posts = 0
    admitted = 0
    skip_reasons: Counter[str] = Counter()
    for post_id in sorted(latest):
        record = latest[post_id]
        if record.status is not ProcessingStatus.ACCEPTED:
            if not _clears_query_bar(record, min_query_confidence):
                skip_reasons[record.status.value] += 1
                continue
            admitted += 1
        else:
            accepted_posts += 1
        if record.query is None or record.episode is None:
            skip_reasons["invalid_accepted_record"] += 1
            continue
        video_id = (record.episode.youtube_video_id or "").strip()
        if not video_id:
            skip_reasons["invalid_accepted_record"] += 1
            continue
        if coverage is not None and video_id not in coverage:
            skip_reasons[ProcessingStatus.EPISODE_NOT_IN_CORPUS.value] += 1
            continue
        query = record.query.text
        rows.append(
            {
                "query_id": f"tt_{post_id}",
                "query": query,
                "language": detect_serbian_script(query),
                "category": "organic_tiktok",
                "split": split,
                "judgments": [{"youtube_video_id": video_id, "grade": 2}],
                "provenance": {
                    "platform": "tiktok",
                    "post_id": post_id,
                    "url": record.canonical_url,
                    "query_confidence": record.query.confidence,
                },
            }
        )

    summary = ExportSummary(
        input_records=len(records),
        unique_posts=len(latest),
        accepted_posts=accepted_posts,
        exported=len(rows),
        skipped=len(latest) - len(rows),
        skip_reasons=dict(sorted(skip_reasons.items())),
        admitted_below_query_bar=admitted,
    )
    return ExportResult(rows=tuple(rows), summary=summary)


def build_combined_export(
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
    corpus_video_ids: Set[str] | None = None,
    segments_path: Path | str | None = None,
    min_query_confidence: float | None = None,
    catalog: EpisodeCatalog | Path | str | None = None,
) -> ExportResult:
    """Build one file holding every recovered query, labelled or not.

    Rows share a single schema: ``judgments`` carries the episode when one was
    matched and is empty when it was not, so the file can be filled in by hand
    and used directly. An episode that exists but is not trusted enough to be a
    judgment is reported as ``episode_candidate`` rather than being dropped.
    """
    if corpus_video_ids is not None and segments_path is not None:
        raise ValueError("Provide corpus_video_ids or segments_path, not both")
    coverage = (
        corpus_video_ids_from_segments(segments_path)
        if segments_path is not None
        else corpus_video_ids
    )
    labels = _episode_labels(catalog)
    records = list(source.records() if isinstance(source, JsonlStateStore) else source)
    latest: dict[str, ProcessingRecord] = {}
    for record in records:
        if record.status is not ProcessingStatus.DUPLICATE:
            latest[record.post_id] = record

    rows: list[dict[str, Any]] = []
    labelled = 0
    admitted = 0
    skip_reasons: Counter[str] = Counter()
    for post_id in sorted(latest):
        record = latest[post_id]
        if record.query is None or not record.query.text.strip():
            skip_reasons[record.status.value] += 1
            continue
        trusted = record.status is ProcessingStatus.ACCEPTED
        if not trusted and _clears_query_bar(record, min_query_confidence):
            trusted = True
            admitted += 1
        video_id = (
            (record.episode.youtube_video_id or "").strip()
            if record.episode is not None
            else ""
        )
        if video_id and coverage is not None and video_id not in coverage:
            video_id = ""
        judgments: list[dict[str, Any]] = []
        if trusted and video_id:
            # The runner reads only youtube_video_id and grade; number and
            # title ride along so a row is readable without a catalog lookup.
            judgments = [
                {"youtube_video_id": video_id, "grade": 2, **labels.get(video_id, {})}
            ]
            labelled += 1
        query = record.query.text
        provenance: dict[str, Any] = {
            "platform": "tiktok",
            "post_id": post_id,
            "url": record.canonical_url,
            "query_confidence": record.query.confidence,
            "status": record.status.value,
            "caption": record.caption,
        }
        if not judgments and video_id:
            provenance["episode_candidate"] = video_id
            provenance.update(labels.get(video_id, {}))
        rows.append(
            {
                "query_id": f"tt_{post_id}",
                "query": query,
                "language": detect_serbian_script(query),
                "category": "organic_tiktok",
                "split": split,
                "judgments": judgments,
                "provenance": provenance,
            }
        )

    summary = ExportSummary(
        input_records=len(records),
        unique_posts=len(latest),
        accepted_posts=labelled,
        exported=len(rows),
        skipped=len(latest) - len(rows),
        skip_reasons=dict(sorted(skip_reasons.items())),
        admitted_below_query_bar=admitted,
    )
    return ExportResult(rows=tuple(rows), summary=summary)


def export_combined_jsonl(
    output_path: Path | str,
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
    corpus_video_ids: Set[str] | None = None,
    segments_path: Path | str | None = None,
    min_query_confidence: float | None = None,
    catalog: EpisodeCatalog | Path | str | None = None,
) -> ExportResult:
    """Atomically write the combined labelled/unlabelled JSONL."""
    result = build_combined_export(
        source,
        split=split,
        corpus_video_ids=corpus_video_ids,
        segments_path=segments_path,
        min_query_confidence=min_query_confidence,
        catalog=catalog,
    )
    write_jsonl(Path(output_path), result.rows)
    return result


def build_review_export(
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
) -> ExportResult:
    """Build rows for posts that yielded a query but no accepted episode.

    Judgments are deliberately left empty.  These rows are candidates for
    manual labelling, not ground truth, so the caption and source URL travel
    with each one and no episode guess is ever written.
    """
    records = list(source.records() if isinstance(source, JsonlStateStore) else source)
    latest: dict[str, ProcessingRecord] = {}
    for record in records:
        if record.status is not ProcessingStatus.DUPLICATE:
            latest[record.post_id] = record

    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    for post_id in sorted(latest):
        record = latest[post_id]
        if record.status is ProcessingStatus.ACCEPTED:
            skip_reasons["already_accepted"] += 1
            continue
        if record.query is None or not record.query.text.strip():
            skip_reasons[record.status.value] += 1
            continue
        query = record.query.text
        episode = record.episode
        rows.append(
            {
                "query_id": f"tt_{post_id}",
                "query": query,
                "language": detect_serbian_script(query),
                "category": "organic_tiktok",
                "split": split,
                "judgments": [],
                "review": {
                    "status": record.status.value,
                    "query_confidence": record.query.confidence,
                    "query_method": record.query.method,
                    "caption": record.caption,
                    "episode_candidate": (
                        episode.youtube_video_id if episode is not None else None
                    ),
                },
                "provenance": {
                    "platform": "tiktok",
                    "post_id": post_id,
                    "url": record.canonical_url,
                },
            }
        )

    summary = ExportSummary(
        input_records=len(records),
        unique_posts=len(latest),
        accepted_posts=sum(
            1 for r in latest.values() if r.status is ProcessingStatus.ACCEPTED
        ),
        exported=len(rows),
        skipped=len(latest) - len(rows),
        skip_reasons=dict(sorted(skip_reasons.items())),
    )
    return ExportResult(rows=tuple(rows), summary=summary)


def export_review_jsonl(
    output_path: Path | str,
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
) -> ExportResult:
    """Atomically write unlabelled review JSONL and return rows plus summary."""
    result = build_review_export(source, split=split)
    write_jsonl(Path(output_path), result.rows)
    return result


def export_episode_eval_jsonl(
    output_path: Path | str,
    source: Iterable[ProcessingRecord] | JsonlStateStore,
    *,
    split: str = "test",
    corpus_video_ids: Set[str] | None = None,
    segments_path: Path | str | None = None,
    min_query_confidence: float | None = None,
) -> ExportResult:
    """Atomically write eval JSONL and return the exact rows plus export summary."""
    result = build_episode_eval_export(
        source,
        split=split,
        corpus_video_ids=corpus_video_ids,
        segments_path=segments_path,
        min_query_confidence=min_query_confidence,
    )
    write_jsonl(Path(output_path), result.rows)
    return result

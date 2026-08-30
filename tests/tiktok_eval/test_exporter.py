from __future__ import annotations

import json

from drzavni_posao.tiktok_eval.exporter import (
    build_combined_export,
    build_episode_eval_export,
    build_review_export,
    export_episode_eval_jsonl,
    export_review_jsonl,
)
from drzavni_posao.tiktok_eval.models import (
    EpisodeMatch,
    ExtractedQuery,
    ProcessingRecord,
    ProcessingStatus,
)
from drzavni_posao.tiktok_eval.state import JsonlStateStore


def record(
    post_id: str,
    *,
    status: ProcessingStatus = ProcessingStatus.ACCEPTED,
    query: str = "ona epizoda kada Boškić slavi",
    video_id: str = "youtube-42",
) -> ProcessingRecord:
    accepted = status is ProcessingStatus.ACCEPTED
    return ProcessingRecord(
        post_id=post_id,
        canonical_url=f"https://www.tiktok.com/@owner/video/{post_id}",
        status=status,
        outcome="accepted" if accepted else "skipped",
        query=(
            ExtractedQuery(text=query, confidence=0.9, method="ocr")
            if accepted
            else None
        ),
        episode=(
            EpisodeMatch(episode_id="42", youtube_video_id=video_id, method="caption")
            if accepted
            else None
        ),
    )


def test_exports_only_latest_accepted_record_per_post() -> None:
    result = build_episode_eval_export(
        [
            record("2"),
            record("1", status=ProcessingStatus.QUERY_NOT_FOUND),
            record("2", status=ProcessingStatus.FETCH_FAILED),
            record("3", query="Она епизода са јубилејом", video_id="youtube-43"),
        ]
    )

    assert [row["query_id"] for row in result.rows] == ["tt_3"]
    assert result.rows[0]["language"] == "sr-Cyrl"
    assert result.summary.input_records == 4
    assert result.summary.unique_posts == 3
    assert result.summary.accepted_posts == 1
    assert result.summary.skip_reasons == {
        "fetch_failed": 1,
        "query_not_found": 1,
    }


def test_duplicate_audit_event_does_not_replace_latest_result() -> None:
    duplicate = record("10", status=ProcessingStatus.DUPLICATE)
    result = build_episode_eval_export([record("10"), duplicate])

    assert [row["query_id"] for row in result.rows] == ["tt_10"]
    assert result.summary.unique_posts == 1


def test_corpus_coverage_skips_out_of_corpus_episode() -> None:
    result = build_episode_eval_export(
        [record("1", video_id="present"), record("2", video_id="missing")],
        corpus_video_ids={"present"},
    )

    assert [row["query_id"] for row in result.rows] == ["tt_1"]
    assert result.summary.skip_reasons == {"episode_not_in_corpus": 1}
    assert result.summary.exported == 1
    assert result.summary.skipped == 1


def test_export_is_deterministic_and_has_episode_eval_schema(tmp_path) -> None:
    state = JsonlStateStore(tmp_path / "state.jsonl")
    state.append(record("20", query="koja je epizoda", video_id="video-b"))
    state.append(record("10", query="neki drugi komentar", video_id="video-a"))
    output = tmp_path / "eval.jsonl"

    first = export_episode_eval_jsonl(output, state, split="real-test")
    first_bytes = output.read_bytes()
    second = export_episode_eval_jsonl(output, state, split="real-test")

    assert output.read_bytes() == first_bytes
    assert first == second
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == list(first.rows)
    assert rows[0] == {
        "query_id": "tt_10",
        "query": "neki drugi komentar",
        "language": "sr-Latn",
        "category": "organic_tiktok",
        "split": "real-test",
        "judgments": [{"youtube_video_id": "video-a", "grade": 2}],
        "provenance": {
            "platform": "tiktok",
            "post_id": "10",
            "url": "https://www.tiktok.com/@owner/video/10",
            "query_confidence": 0.9,
        },
    }
    assert "start_seconds" not in rows[0]["judgments"][0]
    assert "segment_id" not in rows[0]["judgments"][0]


def test_segments_jsonl_can_supply_corpus_coverage(tmp_path) -> None:
    segments = tmp_path / "segments.jsonl"
    segments.write_text(
        json.dumps({"youtube_video_id": "video-a", "segment_id": "s1"}) + "\n",
        encoding="utf-8",
    )

    result = build_episode_eval_export(
        [record("1", video_id="video-a"), record("2", video_id="video-b")],
        segments_path=segments,
    )

    assert [row["query_id"] for row in result.rows] == ["tt_1"]
    assert result.summary.skip_reasons == {"episode_not_in_corpus": 1}


def test_legacy_segments_can_supply_coverage_from_youtube_url(tmp_path) -> None:
    segments = tmp_path / "segments.jsonl"
    segments.write_text(
        json.dumps(
            {
                "youtube_url": "https://youtube.com/watch?v=video-a&t=15s",
                "segment_id": "s1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_episode_eval_export(
        [record("1", video_id="video-a"), record("2", video_id="video-b")],
        segments_path=segments,
    )

    assert [row["query_id"] for row in result.rows] == ["tt_1"]


def _query_only(post_id: str, *, status: ProcessingStatus, caption: str = "") -> ProcessingRecord:
    return ProcessingRecord(
        post_id=post_id,
        canonical_url=f"https://www.tiktok.com/@owner/video/{post_id}",
        caption=caption,
        status=status,
        outcome="skipped",
        query=ExtractedQuery(
            text="koja je epizoda kad Dragan pravi kokice?",
            confidence=0.97,
            method="query_cue_consensus",
        ),
    )


def test_review_export_carries_queries_that_have_no_episode() -> None:
    result = build_review_export(
        [
            record("1"),
            _query_only(
                "2",
                status=ProcessingStatus.EPISODE_NOT_FOUND,
                caption="Replying to @x  TRLA-BABA-LAN!",
            ),
            _query_only("3", status=ProcessingStatus.QUERY_LOW_CONFIDENCE),
        ]
    )

    assert [row["query_id"] for row in result.rows] == ["tt_2", "tt_3"]
    # Judgments stay empty: these are labelling candidates, never ground truth.
    assert all(row["judgments"] == [] for row in result.rows)
    assert result.rows[0]["review"]["status"] == "episode_not_found"
    assert result.rows[0]["review"]["caption"] == "Replying to @x  TRLA-BABA-LAN!"
    assert result.rows[0]["provenance"]["url"].endswith("/video/2")
    assert result.summary.skip_reasons == {"already_accepted": 1}


def test_review_export_omits_posts_without_any_query_text() -> None:
    result = build_review_export(
        [
            record("1", status=ProcessingStatus.QUERY_NOT_FOUND),
            _query_only("2", status=ProcessingStatus.EPISODE_NOT_FOUND),
        ]
    )

    assert [row["query_id"] for row in result.rows] == ["tt_2"]
    assert result.summary.skip_reasons == {"query_not_found": 1}


def test_review_export_writes_jsonl(tmp_path) -> None:
    output = tmp_path / "review.jsonl"

    result = export_review_jsonl(
        output, [_query_only("2", status=ProcessingStatus.EPISODE_NOT_FOUND)]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["query"] == "koja je epizoda kad Dragan pravi kokice?"
    assert row["judgments"] == []
    assert result.summary.exported == 1


def _weak_query_with_episode(post_id: str, confidence: float) -> ProcessingRecord:
    return ProcessingRecord(
        post_id=post_id,
        canonical_url=f"https://www.tiktok.com/@owner/video/{post_id}",
        status=ProcessingStatus.QUERY_LOW_CONFIDENCE,
        outcome="skipped",
        query=ExtractedQuery(
            text="može epozoda oranijaaaaaa",
            confidence=confidence,
            method="multi_frame_consensus",
        ),
        episode=EpisodeMatch(
            episode_id="60", youtube_video_id="youtube-60", method="caption"
        ),
    )


def test_weak_query_with_an_episode_is_admitted_above_the_export_bar() -> None:
    result = build_episode_eval_export(
        [record("1"), _weak_query_with_episode("2", 0.83)],
        min_query_confidence=0.75,
    )

    assert [row["query_id"] for row in result.rows] == ["tt_1", "tt_2"]
    assert result.summary.admitted_below_query_bar == 1
    assert result.rows[1]["provenance"]["query_confidence"] == 0.83


def test_weak_query_below_the_export_bar_stays_out() -> None:
    result = build_episode_eval_export(
        [_weak_query_with_episode("2", 0.60)], min_query_confidence=0.75
    )

    assert result.rows == ()
    assert result.summary.admitted_below_query_bar == 0
    assert result.summary.skip_reasons == {"query_low_confidence": 1}


def test_weak_query_without_an_episode_is_never_admitted() -> None:
    """A trusted query still needs an episode; the exporter never guesses one."""
    weak = _weak_query_with_episode("2", 0.99)
    no_episode = weak.model_copy(update={"episode": None})

    result = build_episode_eval_export([no_episode], min_query_confidence=0.75)

    assert result.rows == ()
    assert result.summary.admitted_below_query_bar == 0


def test_export_bar_is_opt_in() -> None:
    result = build_episode_eval_export([_weak_query_with_episode("2", 0.99)])

    assert result.rows == ()


def test_combined_export_holds_labelled_and_unlabelled_rows_in_one_schema() -> None:
    result = build_combined_export(
        [
            record("1"),
            _query_only("2", status=ProcessingStatus.EPISODE_NOT_FOUND, caption="hi"),
            _weak_query_with_episode("3", 0.60),
        ]
    )

    rows = {row["query_id"]: row for row in result.rows}
    assert set(rows) == {"tt_1", "tt_2", "tt_3"}
    # Matched episode becomes a judgment.
    assert rows["tt_1"]["judgments"] == [
        {"youtube_video_id": "youtube-42", "grade": 2}
    ]
    # No episode at all: empty judgments, nothing invented.
    assert rows["tt_2"]["judgments"] == []
    assert "episode_candidate" not in rows["tt_2"]["provenance"]
    # Episode known but query not trusted: surfaced, not silently dropped.
    assert rows["tt_3"]["judgments"] == []
    assert rows["tt_3"]["provenance"]["episode_candidate"] == "youtube-60"
    assert rows["tt_2"]["provenance"]["status"] == "episode_not_found"
    assert result.summary.exported == 3
    assert result.summary.accepted_posts == 1


def test_combined_export_promotes_a_weak_query_above_the_bar() -> None:
    result = build_combined_export(
        [_weak_query_with_episode("3", 0.83)], min_query_confidence=0.75
    )

    row = result.rows[0]
    assert row["judgments"] == [{"youtube_video_id": "youtube-60", "grade": 2}]
    assert "episode_candidate" not in row["provenance"]
    assert result.summary.admitted_below_query_bar == 1


def test_combined_export_omits_posts_with_no_query_text() -> None:
    result = build_combined_export([record("1", status=ProcessingStatus.NOT_A_REPLY)])

    assert result.rows == ()
    assert result.summary.skip_reasons == {"not_a_reply": 1}


def test_combined_export_names_the_episode_when_a_catalog_is_given() -> None:
    from drzavni_posao.tiktok_eval.extractors import Episode, EpisodeCatalog

    catalog = EpisodeCatalog([Episode(1847, "Kojekude", "youtube-42")])
    result = build_combined_export([record("1")], catalog=catalog)

    judgment = result.rows[0]["judgments"][0]
    assert judgment["youtube_video_id"] == "youtube-42"
    assert judgment["episode_number"] == 1847
    assert judgment["episode_title"] == "Kojekude"


def test_combined_export_without_a_catalog_keeps_the_bare_judgment() -> None:
    result = build_combined_export([record("1")])

    assert result.rows[0]["judgments"][0] == {
        "youtube_video_id": "youtube-42",
        "grade": 2,
    }

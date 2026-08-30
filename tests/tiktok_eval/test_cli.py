from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from drzavni_posao.tiktok_eval import cli
from drzavni_posao.tiktok_eval.extractors import (
    Episode,
    EpisodeCatalog,
    QueryExtractionResult,
)
from drzavni_posao.tiktok_eval.models import (
    EpisodeMatch,
    ExtractedQuery,
    ProcessingRecord,
    ProcessingStatus,
    RawPost,
)
from drzavni_posao.tiktok_eval.state import JsonlStateStore


def _post() -> RawPost:
    return RawPost(
        post_id="123",
        canonical_url="https://www.tiktok.com/@owner/video/123",
        caption="Replying to @viewer Epizoda 42 Košarka",
        is_reply=True,
    )


def test_url_manifest_supports_comments_and_rejects_empty(tmp_path) -> None:
    manifest = tmp_path / "posts.txt"
    manifest.write_text(
        "# browser export\n\nhttps://www.tiktok.com/@owner/video/123\n",
        encoding="utf-8",
    )

    assert cli._read_url_manifest(manifest) == [
        "https://www.tiktok.com/@owner/video/123"
    ]

    manifest.write_text("# no posts\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is empty"):
        cli._read_url_manifest(manifest)


def test_browser_post_manifest_preserves_caption_without_metadata_fetch(
    tmp_path,
) -> None:
    manifest = tmp_path / "posts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "canonical_url": "https://www.tiktok.com/@owner/video/123",
                "caption": "Replying to @viewer Epizoda 42 Košarka",
                "published_at": "2026-08-01T12:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    posts = cli._read_post_manifest(manifest)

    assert len(posts) == 1
    assert posts[0].post_id == "123"
    assert posts[0].is_reply is True
    assert posts[0].caption.endswith("Košarka")
    assert posts[0].published_at is not None


def test_browser_post_manifest_rejects_mismatched_id(tmp_path) -> None:
    manifest = tmp_path / "posts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "post_id": "999",
                "url": "https://www.tiktok.com/@owner/video/123",
                "caption": "Replying to @viewer Ep. 42",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        cli._read_post_manifest(manifest)


def test_browser_post_manifest_recognises_localised_reply_marker(tmp_path) -> None:
    manifest = tmp_path / "posts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "url": "https://www.tiktok.com/@owner/video/123",
                "caption": "Odgovor na komentar korisnika @viewer Epizoda 42",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert cli._read_post_manifest(manifest)[0].is_reply is True


def test_episode_adapter_maps_catalog_result_to_collector_model() -> None:
    catalog = EpisodeCatalog([Episode(42, "Košarka", "youtube-42")])

    result = cli._episode_adapter(catalog)(_post())

    assert result is not None
    assert result.youtube_video_id == "youtube-42"
    assert result.episode_id == "youtube-42"
    assert result.method == "number_and_title"


def test_query_adapter_preserves_successful_ocr_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "extract_query_from_url",
        lambda *args, **kwargs: QueryExtractionResult(
            "ok",
            text="ona epizoda u kojoj slave?",
            confidence=0.94,
            method="multi_frame_consensus",
        ),
    )
    args = argparse.Namespace(
        yt_dlp_binary="yt-dlp",
        tesseract_binary="tesseract",
        min_word_confidence=70.0,
        download_delay=0.0,
        download_attempts=1,
        min_query_chars=8,
        min_query_words=1,
        emoji_saturation=0,
        frame_seconds=12.0,
        frame_count=5,
        temp_dir=tmp_path,
    )

    result = cli._query_adapter(args)(_post())

    assert result == ExtractedQuery(
        text="ona epizoda u kojoj slave?",
        confidence=0.94,
        method="multi_frame_consensus",
    )


def test_query_adapter_preserves_low_consensus_for_state_audit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        cli,
        "extract_query_from_url",
        lambda *args, **kwargs: QueryExtractionResult(
            "low_confidence",
            text="unclear text",
            confidence=0.55,
            reason="insufficient_frame_agreement",
        ),
    )
    args = argparse.Namespace(
        yt_dlp_binary="yt-dlp",
        tesseract_binary="tesseract",
        min_word_confidence=70.0,
        download_delay=0.0,
        download_attempts=1,
        min_query_chars=8,
        min_query_words=1,
        emoji_saturation=0,
        frame_seconds=12.0,
        frame_count=5,
        temp_dir=tmp_path,
    )

    result = cli._query_adapter(args)(_post())

    assert result is not None
    assert result.text == "unclear text"
    assert result.confidence == 0.55


def test_query_adapter_preserves_not_found_reason(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "extract_query_from_url",
        lambda *args, **kwargs: QueryExtractionResult(
            "not_found",
            reason="no_plausible_ocr_candidate",
        ),
    )
    args = argparse.Namespace(
        yt_dlp_binary="yt-dlp",
        tesseract_binary="tesseract",
        min_word_confidence=70.0,
        download_delay=0.0,
        download_attempts=1,
        min_query_chars=8,
        min_query_words=1,
        emoji_saturation=0,
        frame_seconds=12.0,
        frame_count=5,
        temp_dir=tmp_path,
    )

    with pytest.raises(cli.QueryNotFoundError, match="no_plausible"):
        cli._query_adapter(args)(_post())


def test_export_command_writes_episode_only_judgment(tmp_path, capsys) -> None:
    state_path = tmp_path / "state.jsonl"
    output_path = tmp_path / "queries.jsonl"
    JsonlStateStore(state_path).append(
        ProcessingRecord(
            post_id="123",
            canonical_url="https://www.tiktok.com/@owner/video/123",
            status=ProcessingStatus.ACCEPTED,
            outcome="accepted",
            query=ExtractedQuery(
                text="она епизода када славе?",
                confidence=0.95,
                method="ocr",
            ),
            episode=EpisodeMatch(
                episode_id="youtube-42",
                youtube_video_id="youtube-42",
                method="caption",
            ),
        )
    )
    args = argparse.Namespace(
        state=state_path,
        output=output_path,
        segments=None,
        split="test",
        review_output=None,
        min_query_confidence=None,
        include_unlabelled=False,
        catalog=Path("data/youtube_episodes.json"),
    )

    assert cli.run_export(args) == 0

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["query_id"] == "tt_123"
    assert row["language"] == "sr-Cyrl"
    assert row["judgments"] == [{"grade": 2, "youtube_video_id": "youtube-42"}]
    summary = json.loads(capsys.readouterr().out)
    assert summary["exported"] == 1


def test_parser_exposes_collect_and_export_defaults() -> None:
    parser = cli.build_argument_parser()

    collect = parser.parse_args(
        ["collect", "--profile-url", "https://www.tiktok.com/@owner"]
    )
    export = parser.parse_args(["export"])

    assert collect.state == cli.DEFAULT_STATE
    assert collect.frame_count == 8
    assert collect.frame_seconds == 8.0
    assert collect.min_query_confidence == 0.85
    assert export.output == cli.DEFAULT_OUTPUT


def test_parser_accepts_browser_metadata_manifest() -> None:
    parser = cli.build_argument_parser()

    collect = parser.parse_args(
        ["collect", "--posts-jsonl", "artifacts/tiktok_eval/posts.jsonl"]
    )

    assert collect.posts_jsonl.name == "posts.jsonl"
    assert collect.profile_url is None
    assert collect.urls_file is None

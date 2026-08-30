from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .collector import (
    EpisodeNotFoundError,
    EpisodeNotInCatalogError,
    QueryNotFoundError,
    TikTokEvalCollector,
)
from .discovery import (
    DiscoveryError,
    PostManifestSource,
    PostSource,
    PublicProfileSource,
    YtDlpPostUrlSource,
    YtDlpProfileSource,
    is_reply_caption,
)
from .exporter import (
    export_combined_jsonl,
    export_episode_eval_jsonl,
    export_review_jsonl,
)
from .extractors import EpisodeCatalog, extract_episode
from .media import (
    PyAVFrameSampler,
    SaturationMasker,
    TesseractOCR,
    YtDlpDownloader,
    extract_query_from_url,
)
from .models import EpisodeMatch, ExtractedQuery, PostRef, RawPost
from .state import JsonlStateStore

DEFAULT_STATE = Path("artifacts/tiktok_eval/collector.jsonl")
DEFAULT_OUTPUT = Path("artifacts/tiktok_eval/evaluation_queries.jsonl")
DEFAULT_CATALOG = Path("data/youtube_episodes.json")
DEFAULT_TEMP_DIR = Path("artifacts/tiktok_eval/tmp")


class _SourceFactory(Protocol):
    def __call__(self, args: argparse.Namespace) -> PostSource: ...


class LimitedPostSource:
    """Limit discovery while preserving the selected source's fetch cache."""

    def __init__(self, source: PostSource, limit: int | None):
        self.source = source
        self.limit = limit

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        refs = self.source.discover(profile_url)
        if self.limit is None:
            return refs
        selected: list[PostRef] = []
        for ref in refs:
            if len(selected) == self.limit:
                break
            selected.append(ref)
        return selected

    def fetch(self, ref: PostRef) -> RawPost:
        return self.source.fetch(ref)


class FallbackProfileSource:
    """Use the first public discovery adapter that successfully returns posts."""

    def __init__(self, sources: Iterable[PostSource]):
        self.sources = tuple(sources)
        self.active: PostSource | None = None

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        failures: list[str] = []
        for source in self.sources:
            try:
                refs = list(source.discover(profile_url))
            except DiscoveryError as error:
                failures.append(f"{type(source).__name__}: {error}")
                continue
            self.active = source
            return refs
        detail = "; ".join(failures) or "no discovery adapters configured"
        raise DiscoveryError(f"All public profile discovery adapters failed: {detail}")

    def fetch(self, ref: PostRef) -> RawPost:
        if self.active is None:
            raise DiscoveryError("Call discover() before fetch()")
        return self.active.fetch(ref)


def _read_url_manifest(path: Path) -> list[str]:
    urls: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if "\t" in value or " " in value:
                raise ValueError(
                    f"Expected one TikTok URL per line in {path}, line {line_number}"
                )
            urls.append(value)
    if not urls:
        raise ValueError(f"URL manifest is empty: {path}")
    return urls


def _read_post_manifest(path: Path) -> list[RawPost]:
    posts: list[RawPost] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} on line {line_number}"
                )
            url = str(row.get("canonical_url") or row.get("url") or "").strip()
            match = PublicProfileSource._VIDEO_URL_RE.fullmatch(url)
            if match is None:
                raise ValueError(
                    f"Expected a full TikTok post URL in {path} on line {line_number}"
                )
            post_id = match.group(2)
            declared_id = str(row.get("post_id") or post_id).strip()
            if declared_id != post_id:
                raise ValueError(
                    f"Post ID does not match its URL in {path} on line {line_number}"
                )
            caption = str(row.get("caption") or "").strip()
            reply_value = row.get("is_reply")
            is_reply = (
                reply_value
                if isinstance(reply_value, bool)
                else is_reply_caption(caption)
            )
            posts.append(
                RawPost.model_validate(
                    {
                        "post_id": post_id,
                        "canonical_url": match.group(1),
                        "caption": caption,
                        "is_reply": is_reply,
                        "published_at": row.get("published_at"),
                        "source_payload": row,
                    }
                )
            )
    if not posts:
        raise ValueError(f"Post manifest is empty: {path}")
    return posts


def _build_source(args: argparse.Namespace) -> PostSource:
    if args.posts_jsonl is not None:
        source: PostSource = PostManifestSource(_read_post_manifest(args.posts_jsonl))
        return LimitedPostSource(source, args.limit)
    if args.urls_file is not None:
        source: PostSource = YtDlpPostUrlSource(
            _read_url_manifest(args.urls_file),
            executable=args.yt_dlp_binary,
        )
        return LimitedPostSource(source, args.limit)

    html_source = LimitedPostSource(
        PublicProfileSource(timeout=args.http_timeout), args.limit
    )
    yt_dlp_source = YtDlpProfileSource(
        max_posts=args.limit,
        executable=args.yt_dlp_binary,
    )
    if args.discovery == "yt-dlp":
        return yt_dlp_source
    if args.discovery == "html":
        return html_source
    return FallbackProfileSource((yt_dlp_source, html_source))


def _query_adapter(args: argparse.Namespace):
    downloader = YtDlpDownloader(
        binary=args.yt_dlp_binary,
        delay_seconds=args.download_delay,
        max_attempts=args.download_attempts,
    )
    sampler = PyAVFrameSampler(
        duration_seconds=args.frame_seconds,
        frame_count=args.frame_count,
    )
    ocr = TesseractOCR(
        binary=args.tesseract_binary,
        min_word_confidence=args.min_word_confidence,
    )
    masker = (
        SaturationMasker(threshold=args.emoji_saturation)
        if args.emoji_saturation > 0
        else None
    )
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract(post: RawPost) -> ExtractedQuery | None:
        result = extract_query_from_url(
            post.canonical_url,
            downloader=downloader,
            sampler=sampler,
            ocr=ocr,
            masker=masker,
            temp_parent=args.temp_dir,
            min_chars=args.min_query_chars,
            min_words=args.min_query_words,
        )
        if result.text is None:
            raise QueryNotFoundError(
                result.reason or f"query extraction returned {result.status}"
            )
        return ExtractedQuery(
            text=result.text,
            confidence=result.confidence,
            method=result.method or "ocr_low_confidence",
        )

    return extract


def _episode_adapter(catalog: EpisodeCatalog):
    def extract(post: RawPost) -> EpisodeMatch | None:
        result = extract_episode(post.caption, catalog)
        if result.status == "not_in_catalog":
            raise EpisodeNotInCatalogError(
                result.reason or "episode is not in the canonical catalog"
            )
        if not result.ok or result.youtube_video_id is None or result.method is None:
            raise EpisodeNotFoundError(
                result.reason or f"episode extraction returned {result.status}"
            )
        return EpisodeMatch(
            episode_id=result.youtube_video_id,
            youtube_video_id=result.youtube_video_id,
            title=result.title,
            confidence=result.confidence,
            method=result.method,
        )

    return extract


def _summary_payload(summary: object) -> dict[str, object]:
    if hasattr(summary, "statuses"):
        statuses = getattr(summary, "statuses")
        return {
            "discovered": getattr(summary, "discovered"),
            "resumed": getattr(summary, "resumed"),
            "accepted": getattr(summary, "accepted"),
            "skipped": getattr(summary, "skipped"),
            "statuses": {
                status.value: count
                for status, count in sorted(
                    statuses.items(), key=lambda item: item[0].value
                )
            },
        }
    return {
        "input_records": getattr(summary, "input_records"),
        "unique_posts": getattr(summary, "unique_posts"),
        "accepted_posts": getattr(summary, "accepted_posts"),
        "exported": getattr(summary, "exported"),
        "skipped": getattr(summary, "skipped"),
        "skip_reasons": getattr(summary, "skip_reasons"),
    }


def run_collect(
    args: argparse.Namespace,
    *,
    source_factory: _SourceFactory = _build_source,
) -> int:
    source = source_factory(args)
    catalog = EpisodeCatalog.from_json(args.catalog)
    collector = TikTokEvalCollector(
        source=source,
        state=JsonlStateStore(args.state),
        query_extractor=_query_adapter(args),
        episode_extractor=_episode_adapter(catalog),
        extractor_version=args.extractor_version,
        min_query_confidence=args.min_query_confidence,
        min_episode_confidence=args.min_episode_confidence,
    )
    source_label = args.profile_url or str(args.posts_jsonl or args.urls_file)
    summary = collector.collect(source_label, retry_failed=args.retry_failed)
    print(json.dumps(_summary_payload(summary), ensure_ascii=False, indent=2))
    return 0


def run_export(args: argparse.Namespace) -> int:
    state = JsonlStateStore(args.state)
    if args.include_unlabelled:
        result = export_combined_jsonl(
            args.output,
            state,
            split=args.split,
            segments_path=args.segments,
            min_query_confidence=args.min_query_confidence,
            catalog=args.catalog if args.catalog.exists() else None,
        )
    else:
        result = export_episode_eval_jsonl(
            args.output,
            state,
            split=args.split,
            segments_path=args.segments,
            min_query_confidence=args.min_query_confidence,
        )
    payload = _summary_payload(result.summary)
    payload["admitted_below_query_bar"] = result.summary.admitted_below_query_bar
    payload["output"] = str(args.output)
    if args.review_output is not None:
        review = export_review_jsonl(args.review_output, state, split=args.split)
        payload["review_exported"] = review.summary.exported
        payload["review_output"] = str(args.review_output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a high-precision, episode-level eval dataset from public TikTok "
            "reply videos"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect", help="discover posts and append per-post extraction outcomes"
    )
    source = collect.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile-url", help="public TikTok profile URL")
    source.add_argument(
        "--urls-file",
        type=Path,
        help="text file containing one full /@user/video/<id> URL per line",
    )
    source.add_argument(
        "--posts-jsonl",
        type=Path,
        help="browser-exported JSONL with canonical_url/url and caption fields",
    )
    collect.add_argument("--state", type=Path, default=DEFAULT_STATE)
    collect.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    collect.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    collect.add_argument("--limit", type=int)
    collect.add_argument(
        "--discovery",
        choices=("auto", "yt-dlp", "html"),
        default="auto",
        help="profile discovery adapter; auto tries yt-dlp and then public HTML",
    )
    collect.add_argument("--retry-failed", action="store_true")
    collect.add_argument("--extractor-version", default="1")
    collect.add_argument("--http-timeout", type=float, default=20.0)
    collect.add_argument("--frame-seconds", type=float, default=8.0)
    collect.add_argument("--frame-count", type=int, default=8)
    collect.add_argument("--min-query-confidence", type=float, default=0.85)
    collect.add_argument("--min-episode-confidence", type=float, default=0.90)
    collect.add_argument("--yt-dlp-binary", default="yt-dlp")
    collect.add_argument("--tesseract-binary", default="tesseract")
    collect.add_argument("--min-word-confidence", type=float, default=70.0)
    collect.add_argument(
        "--min-query-chars",
        type=int,
        default=8,
        help="shortest OCR text treated as a query; separates real one-word "
        "queries such as an episode name from interface text",
    )
    collect.add_argument("--min-query-words", type=int, default=1)
    collect.add_argument(
        "--emoji-saturation",
        type=int,
        default=60,
        help="saturation above which pixels are treated as emoji and whitened "
        "in a second OCR pass; 0 disables the pass",
    )
    collect.add_argument(
        "--download-delay",
        type=float,
        default=6.0,
        help="minimum seconds between post downloads; 0 disables throttling",
    )
    collect.add_argument(
        "--download-attempts",
        type=int,
        default=4,
        help="attempts per post before recording a download failure",
    )
    collect.set_defaults(handler=run_collect)

    export = subparsers.add_parser(
        "export", help="write accepted posts as episode-only evaluation JSONL"
    )
    export.add_argument("--state", type=Path, default=DEFAULT_STATE)
    export.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    export.add_argument(
        "--segments",
        type=Path,
        help="optional segments JSONL; skip labels whose episode is outside the corpus",
    )
    export.add_argument("--split", default="test")
    export.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    export.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="write one file holding every recovered query, with empty "
        "judgments where no episode was matched",
    )
    export.add_argument(
        "--min-query-confidence",
        type=float,
        help="also export posts held back for a weak query when the query "
        "reaches this confidence and the episode was matched",
    )
    export.add_argument(
        "--review-output",
        type=Path,
        help=(
            "also write posts that produced a query but no accepted episode, "
            "with empty judgments, for manual labelling"
        ),
    )
    export.set_defaults(handler=run_export)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.command == "collect":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.http_timeout <= 0:
            parser.error("--http-timeout must be positive")
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()

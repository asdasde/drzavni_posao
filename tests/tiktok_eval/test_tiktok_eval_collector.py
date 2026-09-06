from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from drzavni_posao.tiktok_eval.collector import (
    EpisodeNotInCatalogError,
    EpisodeNotInCorpusError,
    LowConfidenceQueryError,
    TikTokEvalCollector,
)
from drzavni_posao.tiktok_eval.discovery import (
    DiscoveryBlockedError,
    DiscoveryError,
    MissingDependencyError,
    PostManifestSource,
    PostUrlSource,
    PublicProfileSource,
    YtDlpPostUrlSource,
    YtDlpProfileSource,
)
from drzavni_posao.tiktok_eval.models import (
    EpisodeMatch,
    ExtractedQuery,
    PostRef,
    ProcessingStatus,
    RawPost,
)
from drzavni_posao.tiktok_eval.state import JsonlStateStore


class FakeSource:
    def __init__(self, posts, *, fetch_errors=None):
        self.posts = {post.post_id: post for post in posts}
        self.refs = [
            PostRef(post_id=post.post_id, canonical_url=post.canonical_url)
            for post in posts
        ]
        self.fetch_errors = fetch_errors or {}
        self.fetch_calls = []

    def discover(self, profile_url):
        assert profile_url == "https://www.tiktok.com/@example"
        return self.refs

    def fetch(self, ref):
        self.fetch_calls.append(ref.post_id)
        if ref.post_id in self.fetch_errors:
            raise self.fetch_errors[ref.post_id]
        return self.posts[ref.post_id]


def post(post_id, *, reply=True, caption="Replying to @viewer Ep. 42"):
    return RawPost(
        post_id=post_id,
        canonical_url=f"https://www.tiktok.com/@owner/video/{post_id}",
        caption=caption,
        is_reply=reply,
    )


def query_extractor(raw_post):
    return ExtractedQuery(
        text=f"query {raw_post.post_id}", confidence=0.9, method="ocr"
    )


def episode_extractor(raw_post):
    return EpisodeMatch(
        episode_id="42",
        youtube_video_id="youtube-42",
        title="Epizoda 42",
        method="caption_number",
    )


def collector(
    tmp_path,
    source,
    query=query_extractor,
    episode=episode_extractor,
    **options,
):
    state = JsonlStateStore(tmp_path / "collector.jsonl")
    return (
        TikTokEvalCollector(
            source=source,
            state=state,
            query_extractor=query,
            episode_extractor=episode,
            **options,
        ),
        state,
    )


def test_collects_accepted_and_skipped_records_without_stopping(tmp_path):
    posts = [post("1"), post("2", reply=False), post("3")]
    source = FakeSource(posts, fetch_errors={"3": TimeoutError("slow")})
    sut, state = collector(tmp_path, source)

    summary = sut.collect("https://www.tiktok.com/@example")

    records = list(state.records())
    assert [record.status for record in records] == [
        ProcessingStatus.ACCEPTED,
        ProcessingStatus.NOT_A_REPLY,
        ProcessingStatus.FETCH_FAILED,
    ]
    assert [record.outcome for record in records] == ["accepted", "skipped", "skipped"]
    assert records[2].errors[0].stage == "fetch"
    assert records[2].errors[0].error_type == "TimeoutError"
    assert summary.accepted == 1
    assert summary.skipped == 2


@pytest.mark.parametrize(
    ("query", "episode", "status"),
    [
        (lambda _: None, episode_extractor, ProcessingStatus.QUERY_NOT_FOUND),
        (
            lambda _: (_ for _ in ()).throw(LowConfidenceQueryError("unclear")),
            episode_extractor,
            ProcessingStatus.QUERY_LOW_CONFIDENCE,
        ),
        (query_extractor, lambda _: None, ProcessingStatus.EPISODE_NOT_FOUND),
        (
            query_extractor,
            lambda _: (_ for _ in ()).throw(EpisodeNotInCatalogError("missing")),
            ProcessingStatus.EPISODE_NOT_IN_CATALOG,
        ),
        (
            query_extractor,
            lambda _: EpisodeMatch(
                episode_id="999", method="caption", in_catalog=False
            ),
            ProcessingStatus.EPISODE_NOT_IN_CATALOG,
        ),
        (
            query_extractor,
            lambda _: (_ for _ in ()).throw(EpisodeNotInCorpusError("missing")),
            ProcessingStatus.EPISODE_NOT_IN_CORPUS,
        ),
        (
            query_extractor,
            lambda _: EpisodeMatch(episode_id="43", method="caption", in_corpus=False),
            ProcessingStatus.EPISODE_NOT_IN_CORPUS,
        ),
    ],
)
def test_stage_failures_have_precise_status(tmp_path, query, episode, status):
    sut, state = collector(tmp_path, FakeSource([post("10")]), query, episode)

    sut.collect("https://www.tiktok.com/@example")

    assert list(state.records())[0].status is status


@pytest.mark.parametrize("query_mode", ["none", "low_result", "raises"])
def test_episode_is_attempted_and_retained_when_query_fails(tmp_path, query_mode):
    calls = []

    def failing_query(_):
        calls.append("query")
        if query_mode == "none":
            return None
        if query_mode == "low_result":
            return ExtractedQuery(text="unclear", confidence=0.1, method="ocr")
        raise RuntimeError("OCR failed")

    def successful_episode(raw_post):
        calls.append("episode")
        return episode_extractor(raw_post)

    sut, state = collector(
        tmp_path,
        FakeSource([post("11")]),
        query=failing_query,
        episode=successful_episode,
    )

    sut.collect("https://www.tiktok.com/@example")

    record = list(state.records())[0]
    assert calls == ["query", "episode"]
    assert record.status in {
        ProcessingStatus.QUERY_NOT_FOUND,
        ProcessingStatus.QUERY_LOW_CONFIDENCE,
    }
    assert record.episode is not None
    assert record.episode.youtube_video_id == "youtube-42"


def test_query_is_retained_when_episode_fails(tmp_path):
    def failing_episode(_):
        raise RuntimeError("caption failed")

    sut, state = collector(
        tmp_path,
        FakeSource([post("12")]),
        episode=failing_episode,
    )

    sut.collect("https://www.tiktok.com/@example")

    record = list(state.records())[0]
    assert record.status is ProcessingStatus.EPISODE_NOT_FOUND
    assert record.query is not None
    assert record.query.text == "query 12"
    assert [(error.stage, error.message) for error in record.errors] == [
        ("episode", "caption failed")
    ]


def test_both_stage_errors_are_retained_with_query_status_primary(tmp_path):
    def failing_query(_):
        raise RuntimeError("OCR failed")

    def failing_episode(_):
        raise ValueError("caption failed")

    sut, state = collector(
        tmp_path,
        FakeSource([post("13")]),
        query=failing_query,
        episode=failing_episode,
    )

    sut.collect("https://www.tiktok.com/@example")

    record = list(state.records())[0]
    assert record.status is ProcessingStatus.QUERY_NOT_FOUND
    assert [(error.stage, error.message) for error in record.errors] == [
        ("query", "OCR failed"),
        ("episode", "caption failed"),
    ]


def test_both_empty_stage_results_are_retained_for_audit(tmp_path):
    sut, state = collector(
        tmp_path,
        FakeSource([post("14")]),
        query=lambda _: None,
        episode=lambda _: None,
    )

    sut.collect("https://www.tiktok.com/@example")

    record = list(state.records())[0]
    assert record.status is ProcessingStatus.QUERY_NOT_FOUND
    assert [(error.stage, error.error_type) for error in record.errors] == [
        ("query", "QueryNotFoundError"),
        ("episode", "EpisodeNotFoundError"),
    ]


def test_resume_does_not_refetch_terminal_posts(tmp_path):
    source = FakeSource([post("20")])
    sut, state = collector(tmp_path, source)

    first = sut.collect("https://www.tiktok.com/@example")
    second = sut.collect("https://www.tiktok.com/@example")

    assert first.accepted == 1
    assert second.resumed == 1
    assert source.fetch_calls == ["20"]
    assert len(list(state.records())) == 1


def test_changed_extractor_version_reprocesses_append_only_state(tmp_path):
    source = FakeSource([post("22")])
    first, state = collector(tmp_path, source, extractor_version="1")
    first.collect("https://www.tiktok.com/@example")
    second, _ = collector(tmp_path, source, extractor_version="2")

    summary = second.collect("https://www.tiktok.com/@example")

    records = list(state.records())
    assert summary.resumed == 0
    assert summary.accepted == 1
    assert source.fetch_calls == ["22", "22"]
    assert [record.extractor_version for record in records] == ["1", "2"]
    assert state.latest_by_post()["22"].extractor_version == "2"


def test_confidence_thresholds_prevent_weak_acceptance(tmp_path):
    def weak_query(_):
        return ExtractedQuery(text="weak", confidence=0.69, method="ocr")

    sut, state = collector(tmp_path, FakeSource([post("19")]), weak_query)

    sut.collect("https://www.tiktok.com/@example")

    record = list(state.records())[0]
    assert record.status is ProcessingStatus.QUERY_LOW_CONFIDENCE
    assert record.query is not None
    assert "below 0.700" in record.errors[0].message


def test_retry_failed_retries_fetch_failures(tmp_path):
    source = FakeSource([post("21")], fetch_errors={"21": TimeoutError("slow")})
    sut, state = collector(tmp_path, source)
    sut.collect("https://www.tiktok.com/@example")
    source.fetch_errors.clear()

    summary = sut.collect("https://www.tiktok.com/@example", retry_failed=True)

    assert summary.accepted == 1
    assert [r.status for r in state.records()] == [
        ProcessingStatus.FETCH_FAILED,
        ProcessingStatus.ACCEPTED,
    ]
    assert state.latest_by_post()["21"].status is ProcessingStatus.ACCEPTED


def test_retry_failed_retries_extraction_skips_but_not_accepted_posts(tmp_path):
    source = FakeSource([post("23")])
    attempts = 0

    def query_after_retry(raw_post):
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else query_extractor(raw_post)

    sut, state = collector(tmp_path, source, query=query_after_retry)
    sut.collect("https://www.tiktok.com/@example")

    retried = sut.collect("https://www.tiktok.com/@example", retry_failed=True)
    accepted_is_terminal = sut.collect(
        "https://www.tiktok.com/@example", retry_failed=True
    )

    assert retried.accepted == 1
    assert accepted_is_terminal.resumed == 1
    assert attempts == 2
    assert [record.status for record in state.records()] == [
        ProcessingStatus.QUERY_NOT_FOUND,
        ProcessingStatus.ACCEPTED,
    ]
    assert state.latest_by_post()["23"].status is ProcessingStatus.ACCEPTED


def test_duplicate_discovery_is_audited_but_does_not_replace_result(tmp_path):
    source = FakeSource([post("30")])
    source.refs.append(source.refs[0])
    sut, state = collector(tmp_path, source)

    summary = sut.collect("https://www.tiktok.com/@example")

    assert summary.statuses[ProcessingStatus.DUPLICATE] == 1
    assert [r.status for r in state.records()] == [
        ProcessingStatus.ACCEPTED,
        ProcessingStatus.DUPLICATE,
    ]
    assert state.latest_by_post()["30"].status is ProcessingStatus.ACCEPTED


def test_state_reports_line_number_for_corrupt_input(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        list(JsonlStateStore(path).records())


def test_public_profile_source_parses_embedded_posts(monkeypatch):
    payload = {
        "__DEFAULT_SCOPE__": {
            "webapp.user-detail": {
                "itemList": [
                    {
                        "id": "123456",
                        "desc": "Replying to @viewer Ep. 42",
                        "createTime": "1",
                        "author": {"uniqueId": "owner"},
                    },
                    {"id": "999999", "desc": "ordinary clip", "video": {}},
                ]
            }
        }
    }
    html = (
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">'
        + json.dumps(payload)
        + "</script>"
    )
    source = PublicProfileSource()
    monkeypatch.setattr(source, "_get", lambda _: html)

    refs = list(source.discover("https://www.tiktok.com/@example"))

    assert [ref.post_id for ref in refs] == ["123456", "999999"]
    assert refs[0].canonical_url == "https://www.tiktok.com/@owner/video/123456"
    assert source.fetch(refs[0]).is_reply is True
    assert source.fetch(refs[0]).published_at is not None
    assert source.fetch(refs[1]).is_reply is False


def test_public_profile_source_refuses_captcha(monkeypatch):
    source = PublicProfileSource()
    monkeypatch.setattr(
        source, "_get", lambda _: "<html>Verify to continue CAPTCHA</html>"
    )

    with pytest.raises(DiscoveryBlockedError, match="will not bypass"):
        list(source.discover("https://www.tiktok.com/@example"))


def test_public_profile_source_explains_dynamic_empty_page(monkeypatch):
    source = PublicProfileSource()
    monkeypatch.setattr(source, "_get", lambda _: "<html>JavaScript app shell</html>")

    with pytest.raises(DiscoveryError, match="No public posts"):
        list(source.discover("https://www.tiktok.com/@example"))


def test_public_profile_source_rejects_lookalike_host():
    with pytest.raises(DiscoveryError, match="TikTok profile URL"):
        list(PublicProfileSource().discover("https://tiktok.com.evil/@example"))


def test_manual_url_manifest_preserves_urls_and_duplicates():
    url = "https://www.tiktok.com/@owner/video/123456"
    source = PostUrlSource([url, url])

    refs = list(source.discover("unused"))

    assert [ref.post_id for ref in refs] == ["123456", "123456"]
    assert all(ref.canonical_url == url for ref in refs)


def test_browser_metadata_manifest_fetches_from_memory():
    raw = post("123456", caption="Replying to @viewer Ep. 42")
    source = PostManifestSource([raw])

    refs = list(source.discover("unused"))

    assert refs == [PostRef(post_id="123456", canonical_url=raw.canonical_url)]
    assert source.fetch(refs[0]) is raw


def test_browser_metadata_manifest_rejects_conflicting_urls_for_one_post():
    with pytest.raises(DiscoveryError, match="conflicting canonical URLs"):
        PostManifestSource(
            [
                post("123456"),
                RawPost(
                    post_id="123456",
                    canonical_url="https://www.tiktok.com/@another/video/123456",
                    caption="Replying to @viewer Ep. 42",
                    is_reply=True,
                ),
            ]
        )


def test_ytdlp_profile_source_builds_public_flat_playlist_command_and_caches():
    commands = []
    payload = {
        "entries": [
            {
                "id": "123456",
                "description": "Replying to @viewer Ep. 42",
                "title": "ignored title",
                "timestamp": 1_700_000_000,
            },
            {
                "id": "234567",
                "title": "ordinary clip",
                "release_timestamp": 1_700_000_001,
            },
        ]
    }

    def runner(command):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    source = YtDlpProfileSource(
        max_posts=25,
        resolver=lambda name: f"/tools/{name}",
        runner=runner,
    )
    refs = list(source.discover("https://www.tiktok.com/@owner"))

    assert commands == [
        [
            "/tools/yt-dlp",
            "--flat-playlist",
            "--dump-single-json",
            "--playlist-end",
            "25",
            "--",
            "https://www.tiktok.com/@owner",
        ]
    ]
    assert [ref.canonical_url for ref in refs] == [
        "https://www.tiktok.com/@owner/video/123456",
        "https://www.tiktok.com/@owner/video/234567",
    ]
    first = source.fetch(refs[0])
    second = source.fetch(refs[1])
    assert first.caption == "Replying to @viewer Ep. 42"
    assert first.is_reply is True
    assert first.published_at is not None
    assert second.caption == "ordinary clip"
    assert second.published_at is not None


def test_ytdlp_profile_source_uses_no_limit_flag_when_unbounded():
    commands = []

    def runner(command):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"entries": [{"id": "123", "title": "clip"}]}),
            stderr="",
        )

    source = YtDlpProfileSource(resolver=lambda _: "/bin/yt-dlp", runner=runner)
    list(source.discover("https://tiktok.com/@owner/"))

    assert "--playlist-end" not in commands[0]


@pytest.mark.parametrize("max_posts", [0, -1, True, 1.5])
def test_ytdlp_profile_source_rejects_invalid_limit(max_posts):
    with pytest.raises(ValueError, match="positive integer"):
        YtDlpProfileSource(max_posts=max_posts)


def test_ytdlp_profile_source_reports_missing_optional_dependency():
    source = YtDlpProfileSource(resolver=lambda _: None)

    with pytest.raises(MissingDependencyError, match="not installed"):
        list(source.discover("https://www.tiktok.com/@owner"))


@pytest.mark.parametrize(
    "url",
    [
        "https://tiktok.com.evil/@owner",
        "https://www.tiktok.com/@owner/video/123",
        "https://www.tiktok.com/explore",
    ],
)
def test_ytdlp_profile_source_requires_real_profile_url(url):
    source = YtDlpProfileSource(resolver=lambda _: "/bin/yt-dlp")

    with pytest.raises(DiscoveryError, match="TikTok|profile"):
        list(source.discover(url))


def test_ytdlp_profile_source_reports_nonzero_exit_without_parsing():
    source = YtDlpProfileSource(
        resolver=lambda _: "/bin/yt-dlp",
        runner=lambda _: SimpleNamespace(
            returncode=1, stdout="not json", stderr="profile unavailable"
        ),
    )

    with pytest.raises(DiscoveryError, match="exit 1.*profile unavailable"):
        list(source.discover("https://www.tiktok.com/@owner"))


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("not json", "malformed JSON"),
        ("[]", "root must be an object"),
        ("{}", "entries list"),
        ('{"entries": []}', "no usable"),
        ('{"entries": [{"id": "invalid"}]}', "no usable"),
    ],
)
def test_ytdlp_profile_source_reports_malformed_or_empty_output(stdout, message):
    source = YtDlpProfileSource(
        resolver=lambda _: "/bin/yt-dlp",
        runner=lambda _: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    with pytest.raises(DiscoveryError, match=message):
        list(source.discover("https://www.tiktok.com/@owner"))


def test_ytdlp_fetch_requires_discovery():
    source = YtDlpProfileSource(resolver=lambda _: "/bin/yt-dlp")
    ref = PostRef(
        post_id="123", canonical_url="https://www.tiktok.com/@owner/video/123"
    )

    with pytest.raises(DiscoveryError, match="call discover"):
        source.fetch(ref)


def test_ytdlp_post_url_source_discovers_offline_then_fetches_exact_post():
    calls = {"resolver": 0, "commands": []}
    url = "https://www.tiktok.com/@owner/video/7654321"

    def resolver(name):
        calls["resolver"] += 1
        return f"/tools/{name}"

    def runner(command):
        calls["commands"].append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "id": "7654321",
                    "description": "Replying to @viewer episode?",
                    "timestamp": 1_700_000_000,
                    "channel": "owner",
                }
            ),
            stderr="",
        )

    source = YtDlpPostUrlSource([url], resolver=resolver, runner=runner)
    refs = list(source.discover("unused"))

    assert refs == [PostRef(post_id="7654321", canonical_url=url)]
    assert calls == {"resolver": 0, "commands": []}

    post = source.fetch(refs[0])
    assert calls["resolver"] == 1
    assert calls["commands"] == [
        [
            "/tools/yt-dlp",
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "--",
            url,
        ]
    ]
    assert post.post_id == "7654321"
    assert post.canonical_url == url
    assert post.caption == "Replying to @viewer episode?"
    assert post.is_reply is True
    assert post.published_at is not None
    assert post.source_payload["channel"] == "owner"
    assert source.fetch(refs[0]) is post
    assert len(calls["commands"]) == 1


def test_ytdlp_post_url_source_uses_title_and_release_timestamp_fallbacks():
    url = "https://www.tiktok.com/@owner/video/123"
    source = YtDlpPostUrlSource(
        [url],
        resolver=lambda _: "/bin/yt-dlp",
        runner=lambda _: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "id": "123",
                    "description": "",
                    "title": "ordinary title",
                    "timestamp": "bad",
                    "release_timestamp": 1_700_000_001,
                }
            ),
            stderr="",
        ),
    )

    post = source.fetch(list(source.discover("unused"))[0])

    assert post.caption == "ordinary title"
    assert post.published_at is not None


def test_ytdlp_post_url_source_rejects_invalid_manifest_without_network():
    calls = []
    source = YtDlpPostUrlSource(
        ["https://tiktok.com.evil/@owner/video/123"],
        resolver=lambda name: calls.append(name),
    )

    with pytest.raises(DiscoveryError, match="full TikTok"):
        list(source.discover("unused"))
    assert calls == []


def test_ytdlp_post_url_source_reports_missing_dependency_at_fetch():
    url = "https://www.tiktok.com/@owner/video/123"
    source = YtDlpPostUrlSource([url], resolver=lambda _: None)
    ref = list(source.discover("unused"))[0]

    with pytest.raises(MissingDependencyError, match="not installed"):
        source.fetch(ref)


def test_ytdlp_post_url_source_rejects_returned_id_mismatch():
    url = "https://www.tiktok.com/@owner/video/123"
    source = YtDlpPostUrlSource(
        [url],
        resolver=lambda _: "/bin/yt-dlp",
        runner=lambda _: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"id": "999", "title": "wrong post"}),
            stderr="",
        ),
    )
    ref = list(source.discover("unused"))[0]

    with pytest.raises(DiscoveryError, match="returned post ID 999, expected 123"):
        source.fetch(ref)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            SimpleNamespace(returncode=2, stdout="", stderr="post unavailable"),
            "exit 2.*post unavailable",
        ),
        (
            SimpleNamespace(returncode=0, stdout="not json", stderr=""),
            "malformed JSON",
        ),
        (
            SimpleNamespace(returncode=0, stdout="[]", stderr=""),
            "no usable metadata",
        ),
        (
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"id": "invalid", "title": "bad"}),
                stderr="",
            ),
            "no usable metadata",
        ),
    ],
)
def test_ytdlp_post_url_source_reports_fetch_failures(result, message):
    url = "https://www.tiktok.com/@owner/video/123"
    source = YtDlpPostUrlSource(
        [url],
        resolver=lambda _: "/bin/yt-dlp",
        runner=lambda _: result,
    )
    ref = list(source.discover("unused"))[0]

    with pytest.raises(DiscoveryError, match=message):
        source.fetch(ref)


def test_ytdlp_post_url_fetch_requires_manifest_discovery():
    source = YtDlpPostUrlSource(
        ["https://www.tiktok.com/@owner/video/123"],
        resolver=lambda _: "/bin/yt-dlp",
    )
    ref = PostRef(
        post_id="123", canonical_url="https://www.tiktok.com/@owner/video/123"
    )

    with pytest.raises(DiscoveryError, match="not in the discovered URL manifest"):
        source.fetch(ref)

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from html import unescape
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .models import PostRef, RawPost, canonical_post_url


_REPLY_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"replying\s+to\b|reply\s+to\b|"
    r"odgovor(?:i)?\s+na\s+komentar\b|"
    r"одговор(?:и)?\s+на\s+коментар\b"
    r")",
    re.IGNORECASE,
)


def is_reply_caption(caption: str) -> bool:
    """Recognise TikTok's English and Serbian reply-caption markers."""

    return _REPLY_CAPTION_RE.match(caption) is not None


class DiscoveryError(RuntimeError):
    """The public source could not be read or did not contain usable posts."""


class DiscoveryBlockedError(DiscoveryError):
    """TikTok asked for authentication, CAPTCHA, or another interactive check."""


class MissingDependencyError(DiscoveryError):
    """An explicitly optional discovery executable is not installed."""


class PostSource(Protocol):
    def discover(self, profile_url: str) -> Iterable[PostRef]: ...

    def fetch(self, ref: PostRef) -> RawPost: ...


class PublicProfileSource:
    """Best-effort reader for post metadata embedded in public TikTok HTML.

    This adapter deliberately does not solve CAPTCHAs, authenticate, execute page
    JavaScript, or imitate a private API. TikTok changes its public HTML often, so
    callers should treat it as an optional live adapter and retain the raw manifest.
    It only sees posts in initial hydration HTML; it does not paginate or scroll a
    JavaScript-rendered profile.
    """

    _SCRIPT_RE = re.compile(
        r"<script\b[^>]*\b(?:id=[\"'](?:__UNIVERSAL_DATA_FOR_REHYDRATION__|SIGI_STATE)[\"'])[^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    _VIDEO_URL_RE = re.compile(
        r"(https?://(?:www\.)?tiktok\.com/@[^/\s\"'<>]+/video/(\d+))"
    )

    def __init__(self, *, timeout: float = 20.0, user_agent: str | None = None):
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; drzavni-posao-eval-collector/1.0; "
            "+https://www.tiktok.com/)"
        )
        self._posts: dict[str, RawPost] = {}

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        parsed = urlparse(profile_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not (
            hostname == "tiktok.com" or hostname.endswith(".tiktok.com")
        ):
            raise DiscoveryError("Expected an http(s) TikTok profile URL")
        html = self._get(profile_url)
        self._guard_interactive_page(html)
        posts = list(self._parse_posts(html))
        if not posts:
            raise DiscoveryError(
                "No public posts were embedded in the profile HTML. The profile may "
                "require JavaScript/login or TikTok may have changed its page format."
            )
        self._posts.update({post.post_id: post for post in posts})
        return [
            PostRef(post_id=p.post_id, canonical_url=p.canonical_url) for p in posts
        ]

    def fetch(self, ref: PostRef) -> RawPost:
        cached = self._posts.get(ref.post_id)
        if cached is not None and cached.source_payload:
            return cached
        html = self._get(ref.canonical_url)
        self._guard_interactive_page(html)
        for post in self._parse_posts(html):
            if post.post_id == ref.post_id and post.source_payload:
                return post
        raise DiscoveryError(
            f"Post {ref.post_id} metadata was not present in public HTML"
        )

    def _get(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if error.code in {401, 403, 429}:
                raise DiscoveryBlockedError(
                    f"TikTok returned HTTP {error.code}; login, CAPTCHA, or rate limiting "
                    "may be required"
                ) from error
            raise DiscoveryError(f"TikTok returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise DiscoveryError(f"Could not fetch {url}: {error}") from error

    @staticmethod
    def _guard_interactive_page(html: str) -> None:
        lowered = html.lower()
        signals = ("captcha", "verify to continue")
        if any(signal in lowered for signal in signals):
            raise DiscoveryBlockedError(
                "TikTok returned an interactive login/CAPTCHA page; this collector will "
                "not bypass it"
            )

    def _parse_posts(self, html: str) -> Iterator[RawPost]:
        seen: set[str] = set()
        for match in self._SCRIPT_RE.finditer(html):
            try:
                payload = json.loads(unescape(match.group(1)))
            except json.JSONDecodeError:
                continue
            for item in _walk_video_items(payload):
                post = _raw_post_from_item(item)
                if post is not None and post.post_id not in seen:
                    seen.add(post.post_id)
                    yield post

        # Useful for discovery if metadata shape changed. Fetch will then inspect
        # the individual post page for its caption and reply marker.
        for matched_url, post_id in self._VIDEO_URL_RE.findall(unescape(html)):
            if post_id not in seen:
                seen.add(post_id)
                yield RawPost(
                    post_id=post_id,
                    canonical_url=matched_url,
                )


class PostUrlSource:
    """Fallback source for a manually/browser-exported manifest of post URLs."""

    def __init__(
        self,
        urls: Iterable[str],
        *,
        fetcher: PublicProfileSource | None = None,
    ):
        self.urls = list(urls)
        self.fetcher = fetcher or PublicProfileSource()

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        del profile_url  # Interface compatibility with profile discovery.
        refs: list[PostRef] = []
        for url in self.urls:
            match = PublicProfileSource._VIDEO_URL_RE.fullmatch(url.strip())
            if match is None:
                raise DiscoveryError(
                    "Manifest entries must be full TikTok /@user/video/<numeric-id> "
                    f"URLs; got {url!r}"
                )
            refs.append(PostRef(post_id=match.group(2), canonical_url=match.group(1)))
        return refs

    def fetch(self, ref: PostRef) -> RawPost:
        return self.fetcher.fetch(ref)


class PostManifestSource:
    """Offline source for browser-exported post metadata.

    The manifest supplies the public URL and caption, so collection does not
    need a second metadata request before attempting media OCR.
    """

    def __init__(self, posts: Iterable[RawPost]):
        self.posts = list(posts)
        self._by_id: dict[str, RawPost] = {}
        for post in self.posts:
            existing = self._by_id.setdefault(post.post_id, post)
            if existing.canonical_url != post.canonical_url:
                raise DiscoveryError(
                    f"Post {post.post_id} occurs with conflicting canonical URLs"
                )

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        del profile_url
        return [
            PostRef(post_id=post.post_id, canonical_url=post.canonical_url)
            for post in self.posts
        ]

    def fetch(self, ref: PostRef) -> RawPost:
        try:
            post = self._by_id[ref.post_id]
        except KeyError as error:
            raise DiscoveryError(
                f"Post {ref.post_id} is not in the manifest"
            ) from error
        if post.canonical_url != ref.canonical_url:
            raise DiscoveryError(f"Post {ref.post_id} URL does not match the manifest")
        return post


class YtDlpPostUrlSource:
    """Fetch metadata for explicit public post URLs with optional ``yt-dlp``.

    Manifest discovery is local and performs no network access. Each first fetch
    invokes yt-dlp for exactly one public post, without cookies, credentials,
    CAPTCHA solving, or private API options, and caches the validated result.
    """

    def __init__(
        self,
        urls: Iterable[str],
        *,
        executable: str = "yt-dlp",
        resolver: Callable[[str], str | None] = shutil.which,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ):
        self.urls = list(urls)
        self.executable = executable
        self.resolver = resolver
        self.runner = runner or self._run
        self._refs: dict[str, PostRef] = {}
        self._posts: dict[str, RawPost] = {}

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        del profile_url  # Interface compatibility; the manifest is authoritative.
        refs: list[PostRef] = []
        for url in self.urls:
            match = PublicProfileSource._VIDEO_URL_RE.fullmatch(url.strip())
            if match is None:
                raise DiscoveryError(
                    "Manifest entries must be full TikTok /@user/video/<numeric-id> "
                    f"URLs; got {url!r}"
                )
            ref = PostRef(post_id=match.group(2), canonical_url=match.group(1))
            refs.append(ref)
            self._refs.setdefault(ref.post_id, ref)
        return refs

    def fetch(self, ref: PostRef) -> RawPost:
        expected_ref = self._refs.get(ref.post_id)
        if expected_ref is None or expected_ref.canonical_url != ref.canonical_url:
            raise DiscoveryError(
                f"Post {ref.post_id} is not in the discovered URL manifest"
            )
        cached = self._posts.get(ref.post_id)
        if cached is not None:
            return cached

        executable_path = self.resolver(self.executable)
        if executable_path is None:
            raise MissingDependencyError(
                "yt-dlp is not installed or not on PATH; install it separately or "
                "use PublicProfileSource/PostUrlSource"
            )
        command = [
            executable_path,
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "--",
            ref.canonical_url,
        ]
        try:
            completed = self.runner(command)
        except FileNotFoundError as error:
            raise MissingDependencyError(
                f"Resolved yt-dlp executable does not exist: {executable_path}"
            ) from error
        except OSError as error:
            raise DiscoveryError(f"Could not execute yt-dlp: {error}") from error
        if completed.returncode != 0:
            detail = _bounded_stderr(completed.stderr)
            suffix = f": {detail}" if detail else ""
            raise DiscoveryError(
                f"yt-dlp failed for public TikTok post {ref.post_id} "
                f"(exit {completed.returncode}){suffix}"
            )
        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise DiscoveryError(
                f"yt-dlp returned malformed JSON for post {ref.post_id}"
            ) from error
        username = _post_url_username(ref.canonical_url)
        post = _raw_post_from_ytdlp(payload, username=username)
        if post is None:
            raise DiscoveryError(
                f"yt-dlp returned no usable metadata for post {ref.post_id}"
            )
        if post.post_id != ref.post_id:
            raise DiscoveryError(
                f"yt-dlp returned post ID {post.post_id}, expected {ref.post_id}"
            )
        # Preserve the exact validated manifest URL as provenance.
        post = post.model_copy(update={"canonical_url": ref.canonical_url})
        self._posts[ref.post_id] = post
        return post


class YtDlpProfileSource:
    """Discover public profile posts through the optional ``yt-dlp`` executable.

    No cookies, credentials, CAPTCHA solver, or private API flags are passed. The
    subprocess and executable resolver are injectable so collection policy can be
    tested without a network or local yt-dlp installation.
    """

    def __init__(
        self,
        *,
        max_posts: int | None = None,
        executable: str = "yt-dlp",
        resolver: Callable[[str], str | None] = shutil.which,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ):
        if max_posts is not None and (
            isinstance(max_posts, bool)
            or not isinstance(max_posts, int)
            or max_posts < 1
        ):
            raise ValueError("max_posts must be a positive integer or None")
        self.max_posts = max_posts
        self.executable = executable
        self.resolver = resolver
        self.runner = runner or self._run
        self._posts: dict[str, RawPost] = {}

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    def discover(self, profile_url: str) -> Iterable[PostRef]:
        username = _profile_username(profile_url)
        executable_path = self.resolver(self.executable)
        if executable_path is None:
            raise MissingDependencyError(
                "yt-dlp is not installed or not on PATH; install it separately or "
                "use PublicProfileSource/PostUrlSource"
            )

        command = [executable_path, "--flat-playlist", "--dump-single-json"]
        if self.max_posts is not None:
            command.extend(["--playlist-end", str(self.max_posts)])
        command.extend(["--", profile_url])
        try:
            completed = self.runner(command)
        except FileNotFoundError as error:
            raise MissingDependencyError(
                f"Resolved yt-dlp executable does not exist: {executable_path}"
            ) from error
        except OSError as error:
            raise DiscoveryError(f"Could not execute yt-dlp: {error}") from error

        if completed.returncode != 0:
            detail = _bounded_stderr(completed.stderr)
            suffix = f": {detail}" if detail else ""
            raise DiscoveryError(
                f"yt-dlp failed for public TikTok profile (exit "
                f"{completed.returncode}){suffix}"
            )
        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise DiscoveryError("yt-dlp returned malformed JSON") from error
        if not isinstance(payload, Mapping):
            raise DiscoveryError("yt-dlp JSON root must be an object")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise DiscoveryError("yt-dlp profile JSON did not contain an entries list")

        posts: list[RawPost] = []
        seen: set[str] = set()
        for entry in entries:
            post = _raw_post_from_ytdlp(entry, username=username)
            if post is not None and post.post_id not in seen:
                seen.add(post.post_id)
                posts.append(post)
        if not posts:
            raise DiscoveryError("yt-dlp returned no usable public TikTok posts")
        self._posts.update({post.post_id: post for post in posts})
        return [
            PostRef(post_id=post.post_id, canonical_url=post.canonical_url)
            for post in posts
        ]

    def fetch(self, ref: PostRef) -> RawPost:
        try:
            return self._posts[ref.post_id]
        except KeyError as error:
            raise DiscoveryError(
                f"Post {ref.post_id} is not cached; call discover() first"
            ) from error


def _walk_video_items(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        identifier = value.get("id") or value.get("itemId")
        if str(identifier).isdigit() and any(
            key in value for key in ("desc", "createTime", "author", "video")
        ):
            yield value
        for child in value.values():
            yield from _walk_video_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_video_items(child)


def _profile_username(profile_url: str) -> str:
    parsed = urlparse(profile_url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not (
        hostname == "tiktok.com" or hostname.endswith(".tiktok.com")
    ):
        raise DiscoveryError("Expected an http(s) TikTok profile URL")
    match = re.fullmatch(r"/@([A-Za-z0-9._-]+)/?", parsed.path)
    if match is None:
        raise DiscoveryError("Expected a public TikTok /@username profile URL")
    return match.group(1)


def _post_url_username(post_url: str) -> str:
    parsed = urlparse(post_url)
    match = re.fullmatch(r"/@([^/]+)/video/\d+", parsed.path)
    if match is None:
        raise DiscoveryError(f"Invalid full TikTok post URL: {post_url!r}")
    return match.group(1)


def _bounded_stderr(stderr: str | None) -> str:
    detail = (stderr or "").strip()
    if len(detail) > 500:
        return detail[:497] + "..."
    return detail


def _raw_post_from_ytdlp(entry: Any, *, username: str) -> RawPost | None:
    if not isinstance(entry, Mapping):
        return None
    identifier = str(entry.get("id") or "").strip()
    if not identifier.isdigit():
        return None
    description = entry.get("description")
    title = entry.get("title")
    caption_value = (
        description if isinstance(description, str) and description.strip() else title
    )
    caption = caption_value.strip() if isinstance(caption_value, str) else ""
    published_at = None
    for timestamp in (entry.get("timestamp"), entry.get("release_timestamp")):
        try:
            if timestamp is None or isinstance(timestamp, bool):
                continue
            published_at = datetime.fromtimestamp(float(timestamp), tz=UTC)
            break
        except (TypeError, ValueError, OSError, OverflowError):
            continue
    return RawPost(
        post_id=identifier,
        canonical_url=(
            f"https://www.tiktok.com/@{quote(username, safe='._-')}/video/{identifier}"
        ),
        caption=caption,
        is_reply=is_reply_caption(caption),
        published_at=published_at,
        source_payload=dict(entry),
    )


def _raw_post_from_item(item: Mapping[str, Any]) -> RawPost | None:
    identifier = str(item.get("id") or item.get("itemId") or "").strip()
    if not identifier.isdigit():
        return None
    caption = str(item.get("desc") or item.get("description") or "").strip()
    author_value = item.get("author")
    if isinstance(author_value, Mapping):
        username = str(author_value.get("uniqueId") or "").strip()
    else:
        username = str(author_value or "").strip()
    canonical_url = canonical_post_url(identifier)
    if username:
        canonical_url = (
            f"https://www.tiktok.com/@{quote(username, safe='._-')}/video/{identifier}"
        )
    create_time = item.get("createTime")
    try:
        published_at = datetime.fromtimestamp(int(str(create_time)), tz=UTC)
    except (TypeError, ValueError, OSError):
        published_at = None
    # TikTok captions for reply videos normally begin with this localized marker.
    is_reply = is_reply_caption(caption)
    return RawPost(
        post_id=identifier,
        canonical_url=canonical_url,
        caption=caption,
        is_reply=is_reply,
        published_at=published_at,
        source_payload=dict(item),
    )

"""Optional, dependency-injected media and OCR adapters for TikTok eval posts."""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from urllib.parse import urlparse

from .extractors import QueryExtractionResult, extract_query


class MediaError(RuntimeError):
    """Base class for expected media pipeline failures."""


class MissingDependencyError(MediaError):
    """An optional executable or Python package is unavailable."""


class MediaDownloadError(MediaError):
    """A public video could not be downloaded."""


class FrameSamplingError(MediaError):
    """A downloaded video could not be decoded into frames."""


class OCRError(MediaError):
    """Tesseract could not extract text from an image."""


Runner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableResolver = Callable[[str], str | None]


class VideoDownloader(Protocol):
    def download(self, url: str, output_dir: str | Path) -> Path: ...


class FrameSampler(Protocol):
    def sample(
        self, video_path: str | Path, output_dir: str | Path
    ) -> tuple[Path, ...]: ...


class ImageOCR(Protocol):
    def recognise(self, image_paths: Sequence[str | Path]) -> tuple[str, ...]: ...


def _is_public_tiktok_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme in {"http", "https"} and (
        host == "tiktok.com" or host.endswith(".tiktok.com")
    )


class YtDlpDownloader:
    """Download one public TikTok URL using an installed ``yt-dlp`` binary.

    This adapter uses no cookies, authentication, CAPTCHA handling, or private
    API flags.  The injected runner/resolver make command construction testable
    without an installed executable.
    """

    def __init__(
        self,
        *,
        binary: str = "yt-dlp",
        runner: Runner = subprocess.run,
        resolver: ExecutableResolver = shutil.which,
        timeout: float = 120.0,
        delay_seconds: float = 0.0,
        max_attempts: int = 4,
        retry_backoff_seconds: float = 5.0,
        retry_backoff_max_seconds: float = 60.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must not be negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        if retry_backoff_max_seconds < 0:
            raise ValueError("retry_backoff_max_seconds must not be negative")
        self.binary = binary
        self.runner = runner
        self.resolver = resolver
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.sleeper = sleeper
        self._last_download: float | None = None

    def _throttle(self) -> None:
        """Space consecutive downloads out to stay under TikTok's rate limit.

        Downloading a long profile at full speed reliably trips a JS challenge
        partway through, which then fails every remaining post in the run.
        """
        if self.delay_seconds <= 0:
            return
        if self._last_download is not None:
            elapsed = time.monotonic() - self._last_download
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                self.sleeper(remaining)
        self._last_download = time.monotonic()

    def download(self, url: str, output_dir: str | Path) -> Path:
        """Download one post, retrying TikTok's intermittent challenge failures.

        TikTok answers a share of requests with a JS challenge page that yt-dlp
        cannot parse. The same post usually downloads on a later attempt, so a
        single failure says nothing about whether the post is reachable.
        """
        if not _is_public_tiktok_url(url):
            raise MediaDownloadError("Expected a public http(s) TikTok URL")
        last_error: MediaDownloadError | None = None
        for attempt in range(self.max_attempts):
            if attempt:
                # Capped exponential backoff: uncapped doubling makes a large
                # attempt budget cost hours per post for no extra benefit.
                self.sleeper(
                    min(
                        self.retry_backoff_seconds * (2 ** (attempt - 1)),
                        self.retry_backoff_max_seconds,
                    )
                )
            self._throttle()
            try:
                return self._download_once(url, output_dir)
            except MediaDownloadError as error:
                last_error = error
        assert last_error is not None
        raise MediaDownloadError(
            f"yt-dlp failed {self.max_attempts} times for the public post: {last_error}"
        ) from last_error

    def _download_once(self, url: str, output_dir: str | Path) -> Path:
        executable = self.resolver(self.binary)
        if executable is None:
            raise MissingDependencyError(
                f"Optional executable '{self.binary}' is not installed or not on PATH"
            )

        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        template = destination / "tiktok-%(id)s.%(ext)s"
        command = [
            executable,
            "--no-playlist",
            "--no-warnings",
            "--print",
            "after_move:filepath",
            "--output",
            str(template),
            url,
        ]
        try:
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except FileNotFoundError as error:
            raise MissingDependencyError(
                f"Optional executable '{self.binary}' could not be started"
            ) from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MediaDownloadError(
                f"yt-dlp could not download the post: {error}"
            ) from error
        if completed.returncode != 0:
            message = (completed.stderr or "").strip()
            raise MediaDownloadError(
                "yt-dlp failed to download the public post"
                + (f": {message}" if message else "")
            )

        output_lines = [
            line.strip()
            for line in (completed.stdout or "").splitlines()
            if line.strip()
        ]
        if not output_lines:
            raise MediaDownloadError("yt-dlp did not report the downloaded file path")
        video_path = Path(output_lines[-1]).expanduser()
        if not video_path.is_absolute():
            video_path = destination / video_path
        try:
            video_path = video_path.resolve(strict=True)
            destination_resolved = destination.resolve(strict=True)
            video_path.relative_to(destination_resolved)
        except (FileNotFoundError, ValueError) as error:
            raise MediaDownloadError(
                "yt-dlp reported a missing file or a path outside the output directory"
            ) from error
        if not video_path.is_file():
            raise MediaDownloadError("yt-dlp output is not a regular file")
        return video_path


@contextmanager
def temporary_download(
    url: str,
    *,
    downloader: VideoDownloader | None = None,
    temp_parent: str | Path | None = None,
) -> Iterator[Path]:
    """Yield a downloaded video and remove its temporary directory afterwards."""

    adapter = downloader or YtDlpDownloader()
    parent = str(Path(temp_parent)) if temp_parent is not None else None
    with TemporaryDirectory(prefix="dp-tiktok-", dir=parent) as directory:
        yield adapter.download(url, directory)


class PyAVFrameSampler:
    """Sample evenly spaced frames from the beginning of a video using PyAV."""

    def __init__(
        self,
        *,
        duration_seconds: float = 4.0,
        frame_count: int = 5,
        av_module: Any | None = None,
    ) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2 for OCR consensus")
        self.duration_seconds = duration_seconds
        self.frame_count = frame_count
        self._av_module = av_module

    def _av(self) -> Any:
        if self._av_module is not None:
            return self._av_module
        try:
            return importlib.import_module("av")
        except ImportError as error:
            raise MissingDependencyError(
                "Frame sampling requires the optional 'av' (PyAV) package"
            ) from error

    def sample(
        self, video_path: str | Path, output_dir: str | Path
    ) -> tuple[Path, ...]:
        source = Path(video_path)
        if not source.is_file():
            raise FrameSamplingError(f"Video file does not exist: {source}")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        step = self.duration_seconds / (self.frame_count - 1)
        targets = [step * index for index in range(self.frame_count)]
        paths: list[Path] = []

        try:
            with self._av().open(str(source)) as container:
                streams = list(container.streams.video)
                if not streams:
                    raise FrameSamplingError("Video has no decodable video stream")
                target_index = 0
                for frame in container.decode(streams[0]):
                    timestamp = frame.time
                    if timestamp is None and frame.pts is not None:
                        timestamp = float(frame.pts * frame.time_base)
                    if timestamp is None:
                        continue
                    while (
                        target_index < len(targets)
                        and float(timestamp) >= targets[target_index]
                    ):
                        output = destination / f"frame-{target_index:03d}.png"
                        frame.to_image().save(output, format="PNG")
                        paths.append(output)
                        target_index += 1
                    if (
                        target_index == len(targets)
                        or float(timestamp) > self.duration_seconds
                    ):
                        break
        except MissingDependencyError:
            raise
        except FrameSamplingError:
            raise
        except ImportError as error:
            raise MissingDependencyError(
                "Converting PyAV frames to images requires the optional Pillow package"
            ) from error
        except Exception as error:
            raise FrameSamplingError(
                f"Could not decode/sample video frames: {error}"
            ) from error
        if not paths:
            raise FrameSamplingError("No timestamped video frames were decoded")
        return tuple(paths)


# Serbian comments appear in both scripts. A combined eng+srp pass lets the
# Latin model dominate and mangles Cyrillic stickers, so recognition runs one
# pass per script group and keeps the higher-confidence result per image.
_CYRILLIC_LANGUAGES = frozenset({"srp"})


class SaturationMasker:
    """Whiten saturated pixels so emoji stop fragmenting the sticker text.

    A reply sticker is black text on white; the emoji in it are the only
    saturated colour. Removing them recovers words that OCR otherwise drops or
    runs together. The same pass slightly blurs Serbian diacritics, which have
    thin colour fringing of their own, so callers OCR both the original and the
    masked frames and keep whichever reads better rather than always masking.
    """

    def __init__(
        self,
        *,
        threshold: int = 60,
        image_module: Any | None = None,
        array_module: Any | None = None,
    ) -> None:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        self.threshold = threshold
        self._image_module = image_module
        self._array_module = array_module

    def _modules(self) -> tuple[Any, Any]:
        image = self._image_module
        array = self._array_module
        if image is None:
            try:
                image = importlib.import_module("PIL.Image")
            except ImportError as error:
                raise MissingDependencyError(
                    "Emoji masking requires the optional Pillow package"
                ) from error
        if array is None:
            try:
                array = importlib.import_module("numpy")
            except ImportError as error:
                raise MissingDependencyError(
                    "Emoji masking requires the optional numpy package"
                ) from error
        return image, array

    def mask(
        self, image_paths: Sequence[str | Path], output_dir: str | Path
    ) -> tuple[Path, ...]:
        image_module, array_module = self._modules()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        masked: list[Path] = []
        for image_path in image_paths:
            source = Path(image_path)
            if not source.is_file():
                raise OCRError(f"Frame to mask does not exist: {source}")
            with image_module.open(source) as handle:
                saturation = array_module.asarray(
                    handle.convert("RGB").convert("HSV")
                )[:, :, 1]
                grey = array_module.asarray(handle.convert("L")).copy()
            grey[saturation > self.threshold] = 255
            output = destination / source.name
            image_module.fromarray(grey).save(output, format="PNG")
            masked.append(output)
        return tuple(masked)


class TesseractOCR:
    """OCR image files with an installed Tesseract executable."""

    def __init__(
        self,
        *,
        binary: str = "tesseract",
        preferred_languages: Sequence[str] = ("eng", "srp_latn", "srp"),
        page_segmentation_mode: int = 6,
        min_word_confidence: float = 70.0,
        runner: Runner = subprocess.run,
        resolver: ExecutableResolver = shutil.which,
        timeout_per_image: float = 30.0,
    ) -> None:
        if not preferred_languages:
            raise ValueError("preferred_languages must not be empty")
        if not 0.0 <= min_word_confidence <= 100.0:
            raise ValueError("min_word_confidence must be between 0 and 100")
        self.binary = binary
        self.preferred_languages = tuple(preferred_languages)
        self.page_segmentation_mode = page_segmentation_mode
        self.min_word_confidence = min_word_confidence
        self.runner = runner
        self.resolver = resolver
        self.timeout_per_image = timeout_per_image

    def _executable(self) -> str:
        executable = self.resolver(self.binary)
        if executable is None:
            raise MissingDependencyError(
                f"Optional executable '{self.binary}' is not installed or not on PATH"
            )
        return executable

    def available_languages(self) -> tuple[str, ...]:
        executable = self._executable()
        try:
            completed = self.runner(
                [executable, "--list-langs"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_per_image,
            )
        except FileNotFoundError as error:
            raise MissingDependencyError("Tesseract could not be started") from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OCRError(f"Could not inspect Tesseract languages: {error}") from error
        if completed.returncode != 0:
            raise OCRError("Tesseract failed while listing installed languages")
        installed = {
            line.strip()
            for line in (completed.stdout or "").splitlines()
            if line.strip() and not line.casefold().startswith("list of available")
        }
        return tuple(
            language for language in self.preferred_languages if language in installed
        )

    def _script_passes(self, languages: Sequence[str]) -> tuple[str, ...]:
        """Group installed languages into one OCR pass per script."""

        latin = [lang for lang in languages if lang not in _CYRILLIC_LANGUAGES]
        cyrillic = [lang for lang in languages if lang in _CYRILLIC_LANGUAGES]
        return tuple(
            "+".join(group) for group in (latin, cyrillic) if group
        )

    def _run(self, executable: str, image: Path, languages: str) -> str:
        command = [
            executable,
            str(image),
            "stdout",
            "-l",
            languages,
            "--psm",
            str(self.page_segmentation_mode),
            "tsv",
        ]
        try:
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_per_image,
            )
        except FileNotFoundError as error:
            raise MissingDependencyError("Tesseract could not be started") from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OCRError(f"Tesseract OCR failed for {image.name}: {error}") from error
        if completed.returncode != 0:
            message = (completed.stderr or "").strip()
            raise OCRError(
                f"Tesseract OCR failed for {image.name}"
                + (f": {message}" if message else "")
            )
        return completed.stdout or ""

    def _parse_tsv(self, payload: str) -> tuple[str, float]:
        """Rebuild line-grouped text and a char-weighted mean word confidence.

        The reply sticker is rendered text and scores far higher than the video
        frame behind it, so dropping low-confidence words removes most of the
        background noise that otherwise defeats multi-frame consensus.
        """

        lines: dict[tuple[str, ...], list[str]] = {}
        weighted_confidence = 0.0
        weight = 0
        for row in payload.splitlines():
            columns = row.split("\t")
            if len(columns) < 12 or columns[0] != "5":
                continue
            word = columns[11].strip()
            if not word:
                continue
            try:
                confidence = float(columns[10])
            except ValueError:
                continue
            if confidence < 0:
                continue
            weighted_confidence += confidence * len(word)
            weight += len(word)
            if confidence < self.min_word_confidence:
                continue
            # Punctuation-only tokens carry no query text and survive at high
            # confidence often enough to distort frame-to-frame agreement.
            if not any(character.isalnum() for character in word):
                continue
            lines.setdefault(tuple(columns[1:5]), []).append(word)
        text = "\n".join(" ".join(words) for words in lines.values())
        return text, (weighted_confidence / weight) if weight else 0.0

    def recognise(self, image_paths: Sequence[str | Path]) -> tuple[str, ...]:
        if not image_paths:
            return ()
        executable = self._executable()
        languages = self.available_languages()
        if not languages:
            expected = ", ".join(self.preferred_languages)
            raise MissingDependencyError(
                f"Tesseract is installed but none of the requested languages are available: {expected}"
            )
        passes = self._script_passes(languages)
        texts: list[str] = []
        for image_path in image_paths:
            image = Path(image_path)
            if not image.is_file():
                raise OCRError(f"OCR image does not exist: {image}")
            best_text = ""
            best_confidence = -1.0
            for pass_languages in passes:
                payload = self._run(executable, image, pass_languages)
                text, confidence = self._parse_tsv(payload)
                # Script choice uses every recognised word, so a pass is not
                # penalised for the words the confidence filter later drops.
                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence
            texts.append(best_text.strip())
        return tuple(texts)


def _reads_the_same_sticker(original: str | None, masked: str | None) -> bool:
    """Check that two OCR readings describe the same sticker.

    Masking only removes emoji from a sticker both passes can see, so the two
    readings should share wording. Without this, a masked pass that erased the
    text and settled on the same artefact in every frame would score a perfect
    consensus and displace a correct reading.
    """
    if not original:
        return True
    if not masked:
        return False

    def tokens(text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[^\W_]{4,}", text.casefold(), re.UNICODE)
        }

    original_tokens = tokens(original)
    return not original_tokens or bool(original_tokens & tokens(masked))


def extract_query_from_video(
    video_path: str | Path,
    *,
    sampler: FrameSampler | None = None,
    ocr: ImageOCR | None = None,
    masker: SaturationMasker | None = None,
    temp_parent: str | Path | None = None,
    **query_options: Any,
) -> QueryExtractionResult:
    """Sample, OCR, and combine query text while cleaning up temporary frames.

    When a masker is supplied the frames are read twice, once as sampled and
    once with emoji removed, and the stronger consensus wins. Neither variant
    is better everywhere: masking rescues stickers whose text emoji have broken
    up, and loses to the original on text with diacritics and no emoji.
    """

    frame_sampler = sampler or PyAVFrameSampler()
    ocr_adapter = ocr or TesseractOCR()
    parent = str(Path(temp_parent)) if temp_parent is not None else None
    with TemporaryDirectory(prefix="dp-tiktok-frames-", dir=parent) as directory:
        frame_paths = frame_sampler.sample(video_path, directory)
        best = extract_query(ocr_adapter.recognise(frame_paths), **query_options)
        if masker is not None:
            masked_paths = masker.mask(frame_paths, Path(directory) / "masked")
            masked = extract_query(
                ocr_adapter.recognise(masked_paths), **query_options
            )
            if masked.confidence > best.confidence and _reads_the_same_sticker(
                best.text, masked.text
            ):
                best = masked
        return best


def extract_query_from_url(
    url: str,
    *,
    downloader: VideoDownloader | None = None,
    sampler: FrameSampler | None = None,
    ocr: ImageOCR | None = None,
    masker: SaturationMasker | None = None,
    temp_parent: str | Path | None = None,
    **query_options: Any,
) -> QueryExtractionResult:
    """Run the complete public URL -> temporary video -> OCR query pipeline."""

    with temporary_download(
        url, downloader=downloader, temp_parent=temp_parent
    ) as video:
        return extract_query_from_video(
            video,
            sampler=sampler,
            ocr=ocr,
            masker=masker,
            temp_parent=temp_parent,
            **query_options,
        )


__all__ = [
    "FrameSampler",
    "FrameSamplingError",
    "ImageOCR",
    "MediaDownloadError",
    "MediaError",
    "MissingDependencyError",
    "OCRError",
    "PyAVFrameSampler",
    "SaturationMasker",
    "TesseractOCR",
    "VideoDownloader",
    "YtDlpDownloader",
    "extract_query_from_url",
    "extract_query_from_video",
    "temporary_download",
]

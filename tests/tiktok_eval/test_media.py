from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from drzavni_posao.tiktok_eval.media import (
    FrameSamplingError,
    MediaDownloadError,
    MissingDependencyError,
    OCRError,
    PyAVFrameSampler,
    SaturationMasker,
    TesseractOCR,
    YtDlpDownloader,
    extract_query_from_url,
    extract_query_from_video,
    temporary_download,
)


def _completed(
    args: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_ytdlp_download_builds_a_public_single_video_command(tmp_path: Path) -> None:
    video = tmp_path / "tiktok-123.mp4"
    video.write_bytes(b"video")
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _completed(command, stdout=f"{video}\n")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
    )
    result = downloader.download("https://www.tiktok.com/@owner/video/123", tmp_path)

    assert result == video.resolve()
    assert calls[0][0] == "/fake/yt-dlp"
    assert "--no-playlist" in calls[0]
    assert not {"--cookies", "--username", "--password"}.intersection(calls[0])


def test_ytdlp_rejects_non_tiktok_urls_without_running(tmp_path: Path) -> None:
    downloader = YtDlpDownloader(
        runner=lambda *_args, **_kwargs: pytest.fail("runner must not be called"),
        resolver=lambda _: "/fake/yt-dlp",
    )

    with pytest.raises(MediaDownloadError, match="TikTok URL"):
        downloader.download("https://example.com/video/123", tmp_path)


def test_ytdlp_reports_missing_optional_binary(tmp_path: Path) -> None:
    downloader = YtDlpDownloader(resolver=lambda _: None)

    with pytest.raises(MissingDependencyError, match="yt-dlp"):
        downloader.download("https://vt.tiktok.com/abc", tmp_path)


def test_ytdlp_nonzero_exit_is_a_clear_download_error(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return _completed(command, returncode=1, stderr="public post unavailable")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
        max_attempts=1,
    )

    with pytest.raises(MediaDownloadError, match="public post unavailable"):
        downloader.download("https://www.tiktok.com/@owner/video/123", tmp_path)


def test_temporary_download_cleans_up_its_directory(tmp_path: Path) -> None:
    class FakeDownloader:
        def download(self, _url: str, output_dir: str | Path) -> Path:
            result = Path(output_dir) / "video.mp4"
            result.write_bytes(b"video")
            return result

    with temporary_download(
        "https://vt.tiktok.com/abc",
        downloader=FakeDownloader(),
        temp_parent=tmp_path,
    ) as video:
        temporary_directory = video.parent
        assert video.exists()
    assert not temporary_directory.exists()


class _FakeImage:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def save(self, path: Path, *, format: str) -> None:  # noqa: A002
        assert format == "PNG"
        path.write_text(self.marker, encoding="utf-8")


class _FakeFrame:
    pts = None
    time_base = None

    def __init__(self, timestamp: float) -> None:
        self.time = timestamp

    def to_image(self) -> _FakeImage:
        return _FakeImage(str(self.time))


class _FakeContainer:
    class Streams:
        video = [object()]

    streams = Streams()

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def decode(self, _stream: object):
        yield from (_FakeFrame(0.0), _FakeFrame(3.0), _FakeFrame(6.0), _FakeFrame(9.0))


class _FakeAV:
    @staticmethod
    def open(_path: str) -> _FakeContainer:
        return _FakeContainer()


def test_pyav_sampler_writes_stably_named_frames_without_real_pyav(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "frames"
    sampler = PyAVFrameSampler(duration_seconds=9, frame_count=4, av_module=_FakeAV())

    frames = sampler.sample(video, output)

    assert [frame.name for frame in frames] == [
        "frame-000.png",
        "frame-001.png",
        "frame-002.png",
        "frame-003.png",
    ]
    assert frames[-1].read_text(encoding="utf-8") == "9.0"


def test_pyav_sampler_requires_an_existing_video(tmp_path: Path) -> None:
    sampler = PyAVFrameSampler(av_module=_FakeAV())

    with pytest.raises(FrameSamplingError, match="does not exist"):
        sampler.sample(tmp_path / "missing.mp4", tmp_path / "frames")


def test_tesseract_uses_only_available_preferred_languages(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "--list-langs" in command:
            return _completed(
                command, stdout="List of available languages (2):\neng\nosd\n"
            )
        return _completed(
            command,
            stdout=_tsv([("ona", 95.0), ("epizoda", 94.0), ("kojoj", 92.0)]),
        )

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    assert ocr.recognise([image]) == ("ona epizoda kojoj",)
    ocr_call = calls[-1]
    assert ocr_call[ocr_call.index("-l") + 1] == "eng"
    assert "--psm" in ocr_call


def _tsv(rows: list[tuple[str, float]]) -> str:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    lines = [header]
    for index, (word, confidence) in enumerate(rows):
        lines.append(
            f"5\t1\t1\t1\t1\t{index + 1}\t0\t0\t10\t10\t{confidence}\t{word}"
        )
    return "\n".join(lines) + "\n"


def test_tesseract_keeps_the_higher_confidence_script_pass(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "--list-langs" in command:
            return _completed(
                command,
                stdout="List of available languages (4):\neng\nosd\nsrp\nsrp_latn\n",
            )
        languages = command[command.index("-l") + 1]
        if languages == "eng+srp_latn":
            return _completed(
                command, stdout=_tsv([("3Ha", 41.0), ("nu", 38.0), ("KO", 45.0)])
            )
        assert languages == "srp"
        return _completed(
            command, stdout=_tsv([("зна", 91.0), ("ли", 89.0), ("ко", 93.0)])
        )

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    assert ocr.recognise([image]) == ("зна ли ко",)
    ocr_calls = [call for call in calls if "--list-langs" not in call]
    assert len(ocr_calls) == 2
    assert all("tsv" in call for call in ocr_calls)


def test_tesseract_single_language_still_filters_low_confidence_words(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "--list-langs" in command:
            return _completed(
                command, stdout="List of available languages (2):\nsrp\nosd\n"
            )
        assert command[command.index("-l") + 1] == "srp"
        return _completed(
            command,
            stdout=_tsv([("која", 95.0), ("епизода", 93.0), ("шум", 12.0)]),
        )

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    assert ocr.recognise([image]) == ("која епизода",)


def test_tesseract_drops_background_noise_below_the_word_threshold(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "--list-langs" in command:
            return _completed(command, stdout="eng\n")
        return _completed(
            command,
            stdout=_tsv(
                [
                    ("koja", 96.0),
                    ("je", 95.0),
                    ("epizoda", 94.0),
                    ("blur", 31.0),
                    ("~", 88.0),
                ]
            ),
        )

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    # The low-confidence word and the high-confidence punctuation both go.
    assert ocr.recognise([image]) == ("koja je epizoda",)


def test_tesseract_word_confidence_threshold_is_configurable(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "--list-langs" in command:
            return _completed(command, stdout="eng\n")
        return _completed(command, stdout=_tsv([("koja", 96.0), ("epizoda", 45.0)]))

    keep_all = TesseractOCR(
        runner=runner, resolver=lambda _: "/fake/tesseract", min_word_confidence=40.0
    )
    strict = TesseractOCR(
        runner=runner, resolver=lambda _: "/fake/tesseract", min_word_confidence=90.0
    )

    assert keep_all.recognise([image]) == ("koja epizoda",)
    assert strict.recognise([image]) == ("koja",)


def test_tesseract_rejects_an_out_of_range_word_confidence() -> None:
    with pytest.raises(ValueError, match="min_word_confidence"):
        TesseractOCR(min_word_confidence=140.0)


def test_tesseract_reports_missing_language_data(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout="List of available languages (1):\nosd\n")

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    with pytest.raises(MissingDependencyError, match="none of the requested languages"):
        ocr.recognise([image])


def test_tesseract_nonzero_ocr_exit_is_clear(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "--list-langs" in command:
            return _completed(command, stdout="eng\n")
        return _completed(command, returncode=1, stderr="bad image")

    ocr = TesseractOCR(runner=runner, resolver=lambda _: "/fake/tesseract")

    with pytest.raises(OCRError, match="bad image"):
        ocr.recognise([image])


def test_complete_url_pipeline_is_injected_and_cleans_up(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class FakeDownloader:
        def download(self, _url: str, output_dir: str | Path) -> Path:
            path = Path(output_dir) / "video.mp4"
            path.write_bytes(b"video")
            observed["download_dir"] = path.parent
            return path

    class FakeSampler:
        def sample(
            self, video_path: str | Path, output_dir: str | Path
        ) -> tuple[Path, ...]:
            assert Path(video_path).exists()
            observed["frame_dir"] = Path(output_dir)
            frames = []
            for index in range(2):
                path = Path(output_dir) / f"frame-{index:03d}.png"
                path.write_bytes(b"png")
                frames.append(path)
            return tuple(frames)

    class FakeOCR:
        def recognise(self, image_paths):
            assert all(Path(path).exists() for path in image_paths)
            return (
                "Replying to @foo ona epizoda u kojoj Boškić slavi?",
                "ona epizoda u kojoj Boškić slavi?",
            )

    result = extract_query_from_url(
        "https://vt.tiktok.com/abc",
        downloader=FakeDownloader(),
        sampler=FakeSampler(),
        ocr=FakeOCR(),
        temp_parent=tmp_path,
    )

    assert result.ok
    assert result.text == "ona epizoda u kojoj Boškić slavi?"
    assert not Path(observed["download_dir"]).exists()
    assert not Path(observed["frame_dir"]).exists()


def test_downloader_spaces_out_consecutive_downloads(tmp_path: Path) -> None:
    video = tmp_path / "tiktok-1.mp4"
    video.write_bytes(b"video")
    slept: list[float] = []

    downloader = YtDlpDownloader(
        runner=lambda command, **_: _completed(command, stdout=f"{video}\n"),
        resolver=lambda _: "/fake/yt-dlp",
        delay_seconds=5.0,
        sleeper=slept.append,
    )
    for _ in range(3):
        downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)

    # The first download starts immediately; later ones wait out the gap.
    assert len(slept) == 2
    assert all(0 < pause <= 5.0 for pause in slept)


def test_downloader_without_a_delay_never_sleeps(tmp_path: Path) -> None:
    video = tmp_path / "tiktok-1.mp4"
    video.write_bytes(b"video")

    downloader = YtDlpDownloader(
        runner=lambda command, **_: _completed(command, stdout=f"{video}\n"),
        resolver=lambda _: "/fake/yt-dlp",
        sleeper=lambda _: pytest.fail("must not sleep when throttling is disabled"),
    )
    downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)


def test_downloader_rejects_a_negative_delay() -> None:
    with pytest.raises(ValueError, match="delay_seconds"):
        YtDlpDownloader(delay_seconds=-1.0)


def test_throttling_does_not_delay_a_rejected_url(tmp_path: Path) -> None:
    downloader = YtDlpDownloader(
        resolver=lambda _: "/fake/yt-dlp",
        delay_seconds=30.0,
        sleeper=lambda _: pytest.fail("must not sleep before validating the URL"),
    )

    with pytest.raises(MediaDownloadError, match="TikTok URL"):
        downloader.download("https://example.com/video/1", tmp_path)


def test_downloader_retries_tiktoks_intermittent_challenge_failure(
    tmp_path: Path,
) -> None:
    video = tmp_path / "tiktok-1.mp4"
    video.write_bytes(b"video")
    attempts: list[int] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        attempts.append(1)
        if len(attempts) < 3:
            return _completed(
                command,
                returncode=1,
                stderr="Unable to extract universal data for rehydration",
            )
        return _completed(command, stdout=f"{video}\n")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
        sleeper=lambda _: None,
    )

    assert downloader.download(
        "https://www.tiktok.com/@owner/video/1", tmp_path
    ) == video.resolve()
    assert len(attempts) == 3


def test_downloader_gives_up_after_max_attempts_and_reports_the_last_error(
    tmp_path: Path,
) -> None:
    attempts: list[int] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        attempts.append(1)
        return _completed(command, returncode=1, stderr="rehydration challenge")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
        max_attempts=3,
        sleeper=lambda _: None,
    )

    with pytest.raises(MediaDownloadError, match="failed 3 times"):
        downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)
    assert len(attempts) == 3


def test_downloader_does_not_retry_a_missing_binary(tmp_path: Path) -> None:
    downloader = YtDlpDownloader(
        resolver=lambda _: None,
        sleeper=lambda _: pytest.fail("a missing binary is not worth retrying"),
    )

    with pytest.raises(MissingDependencyError, match="yt-dlp"):
        downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)


def test_downloader_backs_off_between_retries(tmp_path: Path) -> None:
    video = tmp_path / "tiktok-1.mp4"
    video.write_bytes(b"video")
    attempts: list[int] = []
    slept: list[float] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        attempts.append(1)
        if len(attempts) < 3:
            return _completed(command, returncode=1, stderr="challenge")
        return _completed(command, stdout=f"{video}\n")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
        retry_backoff_seconds=2.0,
        sleeper=slept.append,
    )
    downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)

    assert slept == [2.0, 4.0]


def test_downloader_rejects_a_zero_attempt_budget() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        YtDlpDownloader(max_attempts=0)


def test_downloader_caps_retry_backoff_for_a_large_attempt_budget(
    tmp_path: Path,
) -> None:
    slept: list[float] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return _completed(command, returncode=1, stderr="challenge")

    downloader = YtDlpDownloader(
        runner=runner,
        resolver=lambda _: "/fake/yt-dlp",
        max_attempts=7,
        retry_backoff_seconds=5.0,
        retry_backoff_max_seconds=30.0,
        sleeper=slept.append,
    )

    with pytest.raises(MediaDownloadError):
        downloader.download("https://www.tiktok.com/@owner/video/1", tmp_path)

    assert slept == [5.0, 10.0, 20.0, 30.0, 30.0, 30.0]


class _FakeHSVImage:
    """Minimal stand-in for a PIL image with one saturated pixel."""

    def __init__(self, saturated: bool = True) -> None:
        self.saturated = saturated
        self.saved: Path | None = None

    def __enter__(self) -> "_FakeHSVImage":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def convert(self, mode: str) -> "_FakeHSVImage":
        self.mode = mode
        return self

    def save(self, path: Path, *, format: str) -> None:  # noqa: A002
        assert format == "PNG"
        Path(path).write_bytes(b"masked")


def test_saturation_masker_whitens_saturated_pixels(tmp_path: Path) -> None:
    import numpy as np

    frame = tmp_path / "frame-000.png"
    frame.write_bytes(b"png")
    captured: dict[str, object] = {}

    class FakeImageModule:
        @staticmethod
        def open(_path):  # noqa: ANN001
            return _FakeHSVImage()

        @staticmethod
        def fromarray(array):  # noqa: ANN001
            captured["array"] = array.copy()
            return _FakeHSVImage()

    class FakeArrayModule:
        @staticmethod
        def asarray(image):  # noqa: ANN001
            if image.mode == "HSV":
                # saturation channel: one pixel well above the threshold
                return np.array([[[0, 200, 0], [0, 5, 0]]], dtype=np.uint8)
            return np.array([[10, 20]], dtype=np.uint8)

    masker = SaturationMasker(
        threshold=60, image_module=FakeImageModule(), array_module=FakeArrayModule()
    )
    result = masker.mask([frame], tmp_path / "masked")

    assert [p.name for p in result] == ["frame-000.png"]
    # The saturated pixel is whitened; the unsaturated one is untouched.
    assert list(captured["array"][0]) == [255, 20]


def test_saturation_masker_rejects_an_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SaturationMasker(threshold=300)


def test_masked_pass_wins_only_when_it_reads_better(tmp_path: Path) -> None:
    """Emoji masking helps some stickers and hurts others, so the better wins."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    class FakeSampler:
        def sample(self, _video, output_dir):  # noqa: ANN001
            path = Path(output_dir) / "frame-000.png"
            path.write_bytes(b"png")
            return (path,)

    class FakeMasker:
        def mask(self, _paths, output_dir):  # noqa: ANN001
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            masked = path / "frame-000.png"
            masked.write_bytes(b"png")
            return (masked,)

    class FakeOCR:
        def __init__(self, masked_text: str) -> None:
            self.masked_text = masked_text

        def recognise(self, image_paths):  # noqa: ANN001
            if image_paths[0].parent.name == "masked":
                return (self.masked_text, self.masked_text, self.masked_text)
            return ("zivijedna baba", "zivijedna baba y", "different noise here")

    # Masking recovers the full sentence in every frame, so it wins.
    better = extract_query_from_video(
        video,
        sampler=FakeSampler(),
        ocr=FakeOCR("GARA! Na salasu zivi jedna baba"),
        masker=FakeMasker(),
        temp_parent=tmp_path,
    )
    assert better.text == "GARA! Na salasu zivi jedna baba"

    # A masked pass that agrees no better must not displace the original.
    worse = extract_query_from_video(
        video,
        sampler=FakeSampler(),
        ocr=FakeOCR("nonsense"),
        masker=FakeMasker(),
        temp_parent=tmp_path,
    )
    assert "zivijedna baba" in (worse.text or "")

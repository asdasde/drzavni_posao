# /// script
# requires-python = ">=3.12"
# dependencies = ["yt-dlp[default]"]
# ///

"""Download Državni posao episodes, optionally uploading them to Databricks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

CHANNEL_PLAYLISTS_URL = "https://www.youtube.com/@DrzavniPosaoSerija/playlists"
PLAYLIST_RE = re.compile(
    r"^DRŽAVNI POSAO (?P<season>\d+)(?: - part 2)?$", re.IGNORECASE
)
EPISODE_RE = re.compile(
    r"\bEp\.\s*(?P<number>\d+)\s*[:：]",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\(\d{2}\.\d{2}\.\d{4}\.\)")
PREFIX_RE = re.compile(r"^DRŽAVNI POSAO(?: \[HQ\])?\s*-\s*", re.IGNORECASE)
YOUTUBE_ID_SUFFIX_RE = re.compile(r" \[[^\]]+\]$")
EXCLUDED_TITLES = {
    "DRŽAVNI POSAO [HQ] - Masoni specijal (02.01.2014.)",
    "DRŽAVNI POSAO [HQ] - Najbolji momenti (30.12.2013.)",
    "DRŽAVNI POSAO [HQ] - Specijal (01.01.2014.)",
}

DEFAULT_SEASON = 3
DEFAULT_DOWNLOADS = Path("downloads")
DEFAULT_VOLUME = "dbfs:/Volumes/luka_erceg/drzavni_posao/audios"
DEFAULT_PROFILE = "staging"
RATE_LIMIT_MARKERS = ("REQUEST_LIMIT_EXCEEDED", "RATE_LIMIT", "429")


def volume_name(path: Path) -> str:
    stem = YOUTUBE_ID_SUFFIX_RE.sub("", path.stem)
    return unicodedata.normalize("NFD", f"{stem}{path.suffix}")


def canonical_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name)


def is_episode(entry: dict) -> bool:
    title = entry.get("title") or ""
    return bool(
        title not in EXCLUDED_TITLES
        and DATE_RE.search(title)
        and (EPISODE_RE.search(title) or PREFIX_RE.match(title))
    )


def episode_number(entry: dict) -> int | None:
    match = EPISODE_RE.search(entry.get("title") or "")
    return int(match.group("number")) if match else None


def comma_separated_numbers(value: str) -> set[int]:
    try:
        numbers = {int(item.strip()) for item in value.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not numbers or any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("numbers must be positive")
    return numbers


def is_rate_limited(result: subprocess.CompletedProcess[str]) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return any(marker in combined for marker in RATE_LIMIT_MARKERS)


def run_databricks(
    command: list[str],
    *,
    retries: int = 8,
    base_delay: float = 15.0,
    echo_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    delay = base_delay
    for attempt in range(1, retries + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            if echo_stdout and result.stdout:
                sys.stdout.write(result.stdout)
                sys.stdout.flush()
            return result
        if is_rate_limited(result) and attempt < retries:
            print(
                f"rate limited (attempt {attempt}/{retries}); sleeping {delay:.0f}s...",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
            continue
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )
    raise AssertionError("unreachable")


def remote_names(volume: str, profile: str) -> set[str]:
    result = run_databricks(
        [
            "databricks",
            "fs",
            "ls",
            volume,
            "--profile",
            profile,
            "--output",
            "json",
        ]
    )
    return {
        canonical_name(entry["name"])
        for entry in json.loads(result.stdout or "[]")
        if entry.get("name", "").endswith(".mp3")
    }


def upload_file(local: Path, volume: str, profile: str, *, overwrite: bool) -> None:
    remote = f"{volume.rstrip('/')}/{volume_name(local)}"
    command = [
        "databricks",
        "fs",
        "cp",
        str(local),
        remote,
        "--profile",
        profile,
    ]
    if overwrite:
        command.append("--overwrite")
    run_databricks(command, echo_stdout=True)


def discover_playlists(seasons: set[int] | None) -> list[dict]:
    options = {
        "extract_flat": True,
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        channel = ydl.extract_info(CHANNEL_PLAYLISTS_URL, download=False)

    playlists = []
    for item in channel.get("entries") or []:
        if not item:
            continue
        match = PLAYLIST_RE.fullmatch(item.get("title") or "")
        if match and (seasons is None or int(match.group("season")) in seasons):
            playlists.append(
                {
                    "season": int(match.group("season")),
                    "url": f"https://www.youtube.com/playlist?list={item['id']}",
                }
            )
    return playlists


def playlist_entries(url: str) -> list[dict]:
    options = {
        "extract_flat": True,
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        playlist = ydl.extract_info(url, download=False)
    return [
        entry for entry in playlist.get("entries") or [] if entry and is_episode(entry)
    ]


def upload_existing(
    directory: Path,
    existing_names: set[str],
    args: argparse.Namespace,
) -> tuple[set[str], int]:
    local_names = set()
    uploads = 0
    for local in sorted(directory.glob("*.mp3")):
        name = canonical_name(volume_name(local))
        local_names.add(name)
        if args.local:
            continue
        if name in existing_names and not args.overwrite:
            continue
        if args.dry_run:
            print(f"would upload existing: {local}")
        else:
            print(f"uploading existing: {local.name}", flush=True)
            upload_file(
                local,
                args.volume,
                args.profile,
                overwrite=args.overwrite,
            )
            existing_names.add(name)
        uploads += 1
    return local_names, uploads


def download_and_upload(
    entry: dict,
    directory: Path,
    existing_names: set[str],
    local_names: set[str],
    args: argparse.Namespace,
) -> bool:
    title = entry.get("title") or entry["id"]
    name = canonical_name(f"{title}.mp3")
    if name in local_names:
        return False
    if name in existing_names and not args.overwrite:
        return False

    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
    if args.dry_run:
        action = "download" if args.local else "download and upload"
        print(f"would {action}: {title}")
        return True

    options = {
        "format": "bestaudio/best",
        "js_runtimes": {"node": {}},
        "extractor_args": {"youtube": {"player_client": ["web_embedded"]}},
        "outtmpl": str(directory / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url, download=True)
        local = Path(ydl.prepare_filename(info)).with_suffix(".mp3")

    if not local.is_file():
        raise FileNotFoundError(f"Downloaded MP3 not found: {local}")

    if args.local:
        return True

    print(f"uploading: {local.name}", flush=True)
    upload_file(local, args.volume, args.profile, overwrite=args.overwrite)
    existing_names.add(canonical_name(volume_name(local)))
    if args.sleep > 0:
        time.sleep(args.sleep)
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Download Državni posao episodes and optionally upload each MP3 "
            "to Databricks."
        )
    )
    result.add_argument(
        "--season",
        type=comma_separated_numbers,
        help=f"Comma-separated seasons to process (default: {DEFAULT_SEASON})",
    )
    result.add_argument(
        "--episode",
        type=comma_separated_numbers,
        help="Comma-separated episode numbers to process",
    )
    result.add_argument(
        "--downloads",
        type=Path,
        default=DEFAULT_DOWNLOADS,
        help=f"Local download root (default: {DEFAULT_DOWNLOADS})",
    )
    result.add_argument(
        "--local",
        action="store_true",
        help="Download locally without invoking Databricks",
    )
    result.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help=f"Destination UC volume URI (default: {DEFAULT_VOLUME})",
    )
    result.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"Databricks CLI profile (default: {DEFAULT_PROFILE})",
    )
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite MP3s that already exist on Databricks",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending downloads and uploads without changing anything",
    )
    result.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep after each upload (default: 1)",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    episode_numbers = args.episode or set()
    seasons = (
        args.season if args.season else None if episode_numbers else {DEFAULT_SEASON}
    )
    existing_names = set() if args.local else remote_names(args.volume, args.profile)
    playlists = discover_playlists(seasons)
    if not playlists:
        raise SystemExit(f"No playlists found for seasons: {sorted(seasons or [])}")

    actions = 0
    failures = []
    for playlist in playlists:
        directory = args.downloads / f"season-{playlist['season']}"
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
        local_names, uploads = upload_existing(directory, existing_names, args)
        actions += uploads
        entries = playlist_entries(playlist["url"])
        if episode_numbers:
            entries = [
                entry for entry in entries if episode_number(entry) in episode_numbers
            ]
        for entry in entries:
            try:
                actions += download_and_upload(
                    entry,
                    directory,
                    existing_names,
                    local_names,
                    args,
                )
            except DownloadError:
                failures.append(entry.get("title") or entry["id"])
    if actions == 0:
        print("Nothing to do.")
    elif args.dry_run:
        print(f"{actions} file(s) pending.")
    else:
        print(f"Processed {actions} file(s).")
    if failures:
        print(f"Failed to download {len(failures)} file(s):", file=sys.stderr)
        for title in failures:
            print(f"- {title}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

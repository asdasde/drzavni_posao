# /// script
# requires-python = ">=3.12"
# dependencies = ["yt-dlp"]
# ///

"""Download Državni posao episodes and upload them to Databricks."""

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

CHANNEL_PLAYLISTS_URL = "https://www.youtube.com/@DrzavniPosaoSerija/playlists"
PLAYLIST_RE = re.compile(
    r"^DRŽAVNI POSAO (?P<season>\d+)(?: - part 2)?$", re.IGNORECASE
)
EPISODE_RE = re.compile(r"\bEp\.\s*\d+\s*[:：]", re.IGNORECASE)
DATE_RE = re.compile(r"\(\d{2}\.\d{2}\.\d{4}\.\)")
PREFIX_RE = re.compile(r"^DRŽAVNI POSAO(?: \[HQ\])?\s*-\s*", re.IGNORECASE)
YOUTUBE_ID_SUFFIX_RE = re.compile(r" \[[^\]]+\]$")
YOUTUBE_ID_RE = re.compile(r" \[([^\]]+)\]$")

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


def video_id(path: Path) -> str | None:
    match = YOUTUBE_ID_RE.search(path.stem)
    return match.group(1) if match else None


def is_episode(entry: dict) -> bool:
    title = entry.get("title") or ""
    return bool(
        DATE_RE.search(title)
        and (EPISODE_RE.search(title) or PREFIX_RE.match(title))
    )


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
                f"rate limited (attempt {attempt}/{retries}); "
                f"sleeping {delay:.0f}s...",
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


def upload_file(
    local: Path, volume: str, profile: str, *, overwrite: bool
) -> None:
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


def discover_playlists(seasons: set[int]) -> list[dict]:
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
        if match and int(match.group("season")) in seasons:
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
        entry
        for entry in playlist.get("entries") or []
        if entry and is_episode(entry)
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
        if local_video_id := video_id(local):
            local_names.add(local_video_id)
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
    if name in local_names or entry["id"] in local_names:
        return False
    if name in existing_names and not args.overwrite:
        return False

    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
    if args.dry_run:
        print(f"would download and upload: {title}")
        return True

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(directory / "%(title)s [%(id)s].%(ext)s"),
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

    print(f"uploading: {local.name}", flush=True)
    upload_file(local, args.volume, args.profile, overwrite=args.overwrite)
    existing_names.add(canonical_name(volume_name(local)))
    if args.sleep > 0:
        time.sleep(args.sleep)
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Download Državni posao episodes and upload each MP3 to Databricks."
        )
    )
    result.add_argument(
        "--season",
        type=int,
        action="append",
        help=f"Season to process; repeat for multiple seasons (default: {DEFAULT_SEASON})",
    )
    result.add_argument(
        "--downloads",
        type=Path,
        default=DEFAULT_DOWNLOADS,
        help=f"Local download root (default: {DEFAULT_DOWNLOADS})",
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
    seasons = set(args.season or [DEFAULT_SEASON])
    existing_names = remote_names(args.volume, args.profile)
    playlists = discover_playlists(seasons)
    if not playlists:
        raise SystemExit(f"No playlists found for seasons: {sorted(seasons)}")

    actions = 0
    for playlist in playlists:
        directory = args.downloads / f"season-{playlist['season']}"
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
        local_names, uploads = upload_existing(directory, existing_names, args)
        actions += uploads
        for entry in playlist_entries(playlist["url"]):
            actions += download_and_upload(
                entry,
                directory,
                existing_names,
                local_names,
                args,
            )
    if actions == 0:
        print("Nothing to do.")
    elif args.dry_run:
        print(f"{actions} file(s) pending.")
    else:
        print(f"Processed {actions} file(s).")


if __name__ == "__main__":
    main()

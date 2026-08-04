from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..common.io import read_jsonl, write_jsonl
from ..common.models import EpisodeRecord, SegmentRecord, format_timestamp
from ..text import CHARACTER_DISPLAY_NAMES, normalize_text, semantic_text


EPISODE_PATTERN = re.compile(r"\bEp\.\s*(\d+)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\((\d{2})\.(\d{2})\.(\d{4})\.\)")
GENERIC_SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d+$", re.IGNORECASE)


def parse_episode_filename(filename: str) -> tuple[str, int, str, str | None]:
    """Return episode id, number, title and ISO date parsed from an export name."""
    name = unicodedata.normalize("NFC", Path(filename).stem)
    episode_match = EPISODE_PATTERN.search(name)
    if episode_match is None:
        raise ValueError(f"Cannot parse episode number from filename: {filename!r}")
    episode_number = int(episode_match.group(1))
    episode_id = f"DP-E{episode_number:04d}"

    date_match = DATE_PATTERN.search(name)
    broadcast_date = None
    if date_match is not None:
        day, month, year = date_match.groups()
        broadcast_date = f"{year}-{month}-{day}"

    title_start = episode_match.end()
    title_end = date_match.start() if date_match is not None else len(name)
    title = name[title_start:title_end].strip(" :-：\t")
    if not title:
        title = f"Episode {episode_number}"
    return episode_id, episode_number, title, broadcast_date


def speaker_key(label: object) -> str | None:
    if not isinstance(label, str) or not label.strip():
        return None
    normalized = normalize_text(label)
    tokens = normalized.split()
    for character in CHARACTER_DISPLAY_NAMES:
        if character in tokens:
            return character
    return None


def _load_transcript(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read transcript {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise ValueError(f"Transcript {path} must contain a 'segments' list")
    if not value["segments"]:
        raise ValueError(f"Transcript {path} has no segments")
    return value


def discover_transcripts(directory: Path, *, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Transcript directory does not exist: {directory}")
    paths = sorted(directory.rglob("*.json") if recursive else directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No transcript JSON files found under {directory}")
    return paths


def _candidate_score(
    path: Path, transcript: dict[str, Any], coverage: int
) -> tuple[int, int, int, int, str]:
    labels = [segment.get("speaker") for segment in transcript["segments"]]
    named_count = sum(speaker_key(label) is not None for label in labels)
    is_named = int("named" in {part.casefold() for part in path.parts})
    timed_count = sum(
        isinstance(segment.get("start"), (int, float))
        and isinstance(segment.get("end"), (int, float))
        for segment in transcript["segments"]
    )
    return (is_named, named_count, coverage, timed_count, str(path))


def select_canonical_transcripts(
    paths: list[Path],
) -> list[tuple[Path, dict[str, Any]]]:
    """Deduplicate recursive exports deterministically, preferring named/complete runs."""
    loaded: list[tuple[Path, dict[str, Any], str]] = []
    run_episodes: dict[Path, set[str]] = defaultdict(set)
    for path in paths:
        transcript = _load_transcript(path)
        source_name = str(transcript.get("file") or path.name)
        episode_id, _, _, _ = parse_episode_filename(source_name)
        run_root = (
            path.parent.parent
            if path.parent.name.casefold() == "named"
            else path.parent
        )
        run_episodes[run_root].add(episode_id)
        loaded.append((path, transcript, episode_id))

    candidates: dict[str, list[tuple[Path, dict[str, Any], int]]] = defaultdict(list)
    for path, transcript, episode_id in loaded:
        run_root = (
            path.parent.parent
            if path.parent.name.casefold() == "named"
            else path.parent
        )
        candidates[episode_id].append((path, transcript, len(run_episodes[run_root])))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for episode_id, options in candidates.items():
        winner = max(
            options,
            key=lambda option: _candidate_score(option[0], option[1], option[2]),
        )
        selected.append((winner[0], winner[1]))
    return sorted(
        selected,
        key=lambda item: parse_episode_filename(
            str(item[1].get("file") or item[0].name)
        )[1],
    )


def validate_raw_segments(segments: list[dict[str, Any]], source: Path) -> None:
    previous_start = -math.inf
    for index, segment in enumerate(segments):
        start, end = segment.get("start"), segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"{source}: segment {index} has no numeric start/end")
        if not math.isfinite(float(start)) or not math.isfinite(float(end)):
            raise ValueError(f"{source}: segment {index} has a non-finite timestamp")
        if float(start) < previous_start - 0.05:
            raise ValueError(f"{source}: segment {index} starts out of order")
        if float(end) < float(start):
            raise ValueError(f"{source}: segment {index} ends before it starts")
        if not str(segment.get("text") or "").strip():
            raise ValueError(f"{source}: segment {index} has empty text")
        previous_start = float(start)


def chunk_raw_segments(
    segments: list[dict[str, Any]],
    *,
    max_utterances: int = 8,
    overlap_utterances: int = 2,
    max_seconds: float = 60.0,
    max_gap_seconds: float = 30.0,
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Create overlapping, gap-aware windows from WhisperX microsegments.

    The corpus alignment fragments ordinary sentences into sub-second units, so
    raw utterance count is more reliable than trying to enforce a hard 20-second
    minimum.  Each window is still bounded to 60 seconds and never crosses an
    exceptionally large timestamp gap.
    """
    if max_utterances < 1:
        raise ValueError("max_utterances must be positive")
    if not 0 <= overlap_utterances < max_utterances:
        raise ValueError("overlap_utterances must be between 0 and max_utterances")
    if max_seconds <= 0 or max_gap_seconds < 0:
        raise ValueError("time limits must be positive")

    indexed = list(enumerate(segments))
    blocks: list[list[tuple[int, dict[str, Any]]]] = []
    block: list[tuple[int, dict[str, Any]]] = []
    for item in indexed:
        if block:
            gap = float(item[1]["start"]) - float(block[-1][1]["end"])
            if gap > max_gap_seconds:
                blocks.append(block)
                block = []
        block.append(item)
    if block:
        blocks.append(block)

    chunks: list[list[tuple[int, dict[str, Any]]]] = []
    stride = max_utterances - overlap_utterances
    for items in blocks:
        start_index = 0
        while start_index < len(items):
            end_index = min(len(items), start_index + max_utterances)
            while end_index > start_index + 1:
                span = float(items[end_index - 1][1]["end"]) - float(
                    items[start_index][1]["start"]
                )
                if span <= max_seconds:
                    break
                end_index -= 1

            # A single unusually long source segment is emitted by itself.
            chunks.append(items[start_index:end_index])
            if end_index >= len(items):
                break
            consumed = end_index - start_index
            start_index += min(stride, max(1, consumed - overlap_utterances))
    return chunks


def _run_id(path: Path) -> str:
    if path.parent.name.casefold() == "named":
        return f"{path.parent.parent.name}/named"
    return path.parent.name or "root"


def _load_episode_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "episode_id" in row:
            episode_id = str(row["episode_id"])
        elif "episode_number" in row:
            episode_id = f"DP-E{int(row['episode_number']):04d}"
        else:
            raise ValueError("Episode metadata needs episode_id or episode_number")
        if episode_id in metadata:
            raise ValueError(f"Duplicate episode metadata for {episode_id}")
        metadata[episode_id] = row
    return metadata


def build_records(
    selected: list[tuple[Path, dict[str, Any]]],
    *,
    metadata: dict[str, dict[str, Any]] | None = None,
    source_time_offset_seconds: float = 0.0,
    max_utterances: int = 8,
    overlap_utterances: int = 2,
    max_seconds: float = 60.0,
    max_gap_seconds: float = 30.0,
) -> tuple[list[EpisodeRecord], list[SegmentRecord]]:
    metadata = metadata or {}
    episodes: list[EpisodeRecord] = []
    searchable_segments: list[SegmentRecord] = []
    seen_episode_ids: set[str] = set()

    for path, transcript in selected:
        source_name = unicodedata.normalize(
            "NFC", str(transcript.get("file") or path.name)
        )
        episode_id, episode_number, parsed_title, parsed_date = parse_episode_filename(
            source_name
        )
        if episode_id in seen_episode_ids:
            raise ValueError(f"Duplicate canonical episode id: {episode_id}")
        seen_episode_ids.add(episode_id)
        episode_meta = metadata.get(episode_id, {})
        offset = float(
            episode_meta.get("source_time_offset_seconds", source_time_offset_seconds)
        )
        episode = EpisodeRecord(
            episode_id=episode_id,
            episode_number=episode_number,
            episode_title=str(episode_meta.get("episode_title") or parsed_title),
            broadcast_date=episode_meta.get("broadcast_date") or parsed_date,
            source_file=source_name,
            transcript_run_id=_run_id(path),
            source_time_offset_seconds=offset,
            youtube_url=episode_meta.get("youtube_url"),
        )
        episodes.append(episode)

        raw_segments = transcript["segments"]
        validate_raw_segments(raw_segments, path)
        chunks = chunk_raw_segments(
            raw_segments,
            max_utterances=max_utterances,
            overlap_utterances=overlap_utterances,
            max_seconds=max_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        for chunk_number, chunk in enumerate(chunks):
            start = float(chunk[0][1]["start"])
            end = float(chunk[-1][1]["end"])
            source_indices = [index for index, _ in chunk]
            keys: list[str] = []
            display_lines: list[str] = []
            plain_lines: list[str] = []
            for _, raw in chunk:
                text = unicodedata.normalize("NFC", str(raw["text"])).strip()
                plain_lines.append(text)
                key = speaker_key(raw.get("speaker"))
                raw_label = raw.get("speaker")
                if key:
                    keys.append(key)
                    label = CHARACTER_DISPLAY_NAMES[key]
                    display_lines.append(f"[{label}] {text}")
                elif isinstance(raw_label, str) and GENERIC_SPEAKER_PATTERN.match(
                    raw_label
                ):
                    display_lines.append(f"[{raw_label}] {text}")
                else:
                    display_lines.append(text)

            speaker_keys = list(dict.fromkeys(keys))
            speakers = [CHARACTER_DISPLAY_NAMES[key] for key in speaker_keys]
            text_original = "\n".join(display_lines)
            dense_text = " ".join([episode.episode_title, *speakers, *plain_lines])
            lexical_text = " ".join(
                [episode.episode_title, *speaker_keys, *plain_lines]
            )
            video_start = start + offset
            video_end = end + offset
            searchable_segments.append(
                SegmentRecord(
                    episode_id=episode_id,
                    episode_title=episode.episode_title,
                    segment_id=f"{episode_id}_{chunk_number:04d}",
                    start_seconds=start,
                    end_seconds=end,
                    start_time=format_timestamp(start),
                    end_time=format_timestamp(end),
                    video_start_seconds=video_start,
                    video_end_seconds=video_end,
                    video_start_time=format_timestamp(video_start),
                    video_end_time=format_timestamp(video_end),
                    text_original=text_original,
                    text_semantic=semantic_text(dense_text),
                    text_normalized=normalize_text(lexical_text),
                    speakers=speakers,
                    speaker_keys=speaker_keys,
                    source_segment_indices=source_indices,
                    youtube_url=episode.youtube_url,
                    source_file=source_name,
                    source_time_offset_seconds=offset,
                )
            )
    return episodes, searchable_segments


def validate_records(
    episodes: list[EpisodeRecord], segments: list[SegmentRecord]
) -> None:
    episode_ids = [episode.episode_id for episode in episodes]
    segment_ids = [segment.segment_id for segment in segments]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Episode IDs are not unique")
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Segment IDs are not unique")
    known_episodes = set(episode_ids)
    previous: dict[str, float] = defaultdict(lambda: -math.inf)
    for segment in segments:
        if segment.episode_id not in known_episodes:
            raise ValueError(f"Unknown episode for {segment.segment_id}")
        if segment.end_seconds < segment.start_seconds:
            raise ValueError(f"Invalid timestamps for {segment.segment_id}")
        if segment.start_seconds < previous[segment.episode_id] - 0.05:
            raise ValueError(f"Segments are out of order in {segment.episode_id}")
        if not segment.text_original or not segment.text_normalized:
            raise ValueError(f"Empty searchable text for {segment.segment_id}")
        previous[segment.episode_id] = segment.start_seconds


def build_corpus(
    transcript_dir: Path,
    episodes_output: Path,
    segments_output: Path,
    *,
    metadata_path: Path | None = None,
    recursive: bool = False,
    source_time_offset_seconds: float = 0.0,
    max_utterances: int = 8,
    overlap_utterances: int = 2,
    max_seconds: float = 60.0,
    max_gap_seconds: float = 30.0,
) -> tuple[list[EpisodeRecord], list[SegmentRecord]]:
    paths = discover_transcripts(transcript_dir, recursive=recursive)
    selected = select_canonical_transcripts(paths)
    episodes, segments = build_records(
        selected,
        metadata=_load_episode_metadata(metadata_path),
        source_time_offset_seconds=source_time_offset_seconds,
        max_utterances=max_utterances,
        overlap_utterances=overlap_utterances,
        max_seconds=max_seconds,
        max_gap_seconds=max_gap_seconds,
    )
    validate_records(episodes, segments)
    write_jsonl(episodes_output, (episode.to_dict() for episode in episodes))
    write_jsonl(segments_output, (segment.to_dict() for segment in segments))
    return episodes, segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WhisperX transcript JSON into searchable episode chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing transcript JSON",
    )
    parser.add_argument(
        "--episodes-output", type=Path, default=Path("data/episodes.jsonl")
    )
    parser.add_argument(
        "--segments-output", type=Path, default=Path("data/segments.jsonl")
    )
    parser.add_argument("--metadata", type=Path, help="Optional episode metadata JSONL")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search snapshots recursively and deduplicate",
    )
    parser.add_argument("--source-time-offset", type=float, default=0.0)
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--overlap-utterances", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    args = parser.parse_args()

    episodes, segments = build_corpus(
        args.input,
        args.episodes_output,
        args.segments_output,
        metadata_path=args.metadata,
        recursive=args.recursive,
        source_time_offset_seconds=args.source_time_offset,
        max_utterances=args.max_utterances,
        overlap_utterances=args.overlap_utterances,
        max_seconds=args.max_seconds,
        max_gap_seconds=args.max_gap_seconds,
    )
    run_counts = Counter(episode.transcript_run_id for episode in episodes)
    print(
        f"Wrote {len(episodes)} episodes and {len(segments)} segments "
        f"from {dict(run_counts)}"
    )


if __name__ == "__main__":
    main()

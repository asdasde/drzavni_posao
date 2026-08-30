from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from ...episode import Episode
from ...store import DEFAULT_CATALOG_PATH, MediaStore
from ...tracking import Stage
from ...transcript import Transcript
from ..common.io import write_jsonl
from ..common.models import SegmentRecord
from ..text import CHARACTER_DISPLAY_NAMES, normalize_text, semantic_text


logger = logging.getLogger(__name__)

GENERIC_SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d+$", re.IGNORECASE)


def speaker_key(label: object) -> str | None:
    if not isinstance(label, str) or not label.strip():
        return None
    normalized = normalize_text(label)
    tokens = normalized.split()
    for character in CHARACTER_DISPLAY_NAMES:
        if character in tokens:
            return character
    return None


def _export_transcript_path(episode_dir: Path) -> Path | None:
    for stage in reversed(tuple(Stage)):
        path = episode_dir / stage.value / "transcript.json"
        if path.is_file():
            return path
    return None


def load_episodes_from_mlflow_export(
    media: MediaStore,
    export_dir: Path,
) -> list[Episode]:
    """Load the latest stage transcript from an ``export_labeling_run`` tree."""
    episodes_root = export_dir / "episodes"
    if not episodes_root.is_dir():
        raise ValueError(
            f"{export_dir} is not an MLflow export: expected episodes/<key>/"
            f"<stage>/transcript.json (data/mlflow_exports/<run_id>)"
        )

    episodes = []
    for episode_dir in sorted(episodes_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        path = _export_transcript_path(episode_dir)
        if path is None:
            logger.warning("Skipping %s: no stage transcript.json", episode_dir)
            continue
        episode = media.episode(episode_dir.name)
        episode.transcript = Transcript.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        episodes.append(episode)

    if not episodes:
        raise ValueError(f"No episode transcripts found in {episodes_root}")
    return sorted(episodes)


def source_time_offset_from_export(export_dir: Path) -> float:
    config_path = export_dir / "config.json"
    if not config_path.is_file():
        return 0.0
    audio = json.loads(config_path.read_text(encoding="utf-8")).get("audio") or {}
    return float(audio.get("trim_start") or 0.0)


def validate_transcript(transcript: Transcript, source: Path) -> None:
    previous_start = -math.inf
    for index, segment in enumerate(transcript.segments):
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


def build_records(
    episodes: list[Episode],
    *,
    source_time_offset_seconds: float = 0.0,
    max_utterances: int = 8,
    overlap_utterances: int = 2,
    max_seconds: float = 60.0,
    max_gap_seconds: float = 30.0,
) -> list[SegmentRecord]:
    searchable_segments: list[SegmentRecord] = []
    seen_video_ids: set[str] = set()

    for episode in episodes:
        transcript = episode.transcript
        if transcript is None:
            raise ValueError(f"{episode.audio_path}: episode has no transcript")
        source_name = str(transcript.file or episode.audio_path.name)
        episode_meta = episode.metadata
        video_id = episode_meta.youtube_video_id
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate YouTube video ID: {video_id}")
        seen_video_ids.add(video_id)
        offset = source_time_offset_seconds

        validate_transcript(transcript, episode.audio_path)
        chunks = chunk_raw_segments(
            transcript.segments,
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
                text = str(raw["text"]).strip()
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
            dense_text = " ".join([episode_meta.title, *speakers, *plain_lines])
            lexical_text = " ".join([episode_meta.title, *speaker_keys, *plain_lines])
            video_start = start + offset
            video_end = end + offset
            searchable_segments.append(
                SegmentRecord(
                    youtube_video_id=video_id,
                    episode_number=episode_meta.episode_number,
                    episode_title=episode_meta.title,
                    segment_id=f"{video_id}_{chunk_number:04d}",
                    start_seconds=start,
                    end_seconds=end,
                    video_start_seconds=video_start,
                    video_end_seconds=video_end,
                    text_original=text_original,
                    text_semantic=semantic_text(dense_text),
                    text_normalized=normalize_text(lexical_text),
                    speaker_keys=speaker_keys,
                    source_segment_indices=source_indices,
                    youtube_url=episode_meta.original_url,
                    source_file=source_name,
                    source_time_offset_seconds=offset,
                )
            )
    return searchable_segments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Convert WhisperX transcript JSON into searchable episode chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MLflow export directory (data/mlflow_exports/<run_id>)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Optional directory containing episode MP3 files",
    )
    parser.add_argument(
        "--segments-output", type=Path, default=Path("data/segments.jsonl")
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Episode catalog JSON",
    )
    parser.add_argument(
        "--source-time-offset",
        type=float,
        help="Video offset in seconds (default: audio.trim_start from export config.json)",
    )
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--overlap-utterances", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    args = parser.parse_args()

    media = MediaStore(args.audio_dir, args.input, catalog_path=args.catalog)
    offset = (
        args.source_time_offset
        if args.source_time_offset is not None
        else source_time_offset_from_export(args.input)
    )
    segments = build_records(
        load_episodes_from_mlflow_export(media, args.input),
        source_time_offset_seconds=offset,
        max_utterances=args.max_utterances,
        overlap_utterances=args.overlap_utterances,
        max_seconds=args.max_seconds,
        max_gap_seconds=args.max_gap_seconds,
    )
    write_jsonl(args.segments_output, (segment.model_dump() for segment in segments))
    episode_count = len({segment.youtube_video_id for segment in segments})
    logger.info("Wrote %d segments from %d episodes", len(segments), episode_count)


if __name__ == "__main__":
    main()

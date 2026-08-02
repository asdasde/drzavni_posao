from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format non-negative seconds as HH:MM:SS."""
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    episode_number: int | None
    episode_title: str
    broadcast_date: str | None
    source_file: str
    transcript_run_id: str
    source_time_offset_seconds: float = 0.0
    youtube_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeRecord":
        return cls(
            episode_id=str(value["episode_id"]),
            episode_number=value.get("episode_number"),
            episode_title=str(value.get("episode_title") or "Untitled episode"),
            broadcast_date=value.get("broadcast_date"),
            source_file=str(value.get("source_file") or ""),
            transcript_run_id=str(value.get("transcript_run_id") or "unknown"),
            source_time_offset_seconds=float(
                value.get("source_time_offset_seconds", 0.0)
            ),
            youtube_url=value.get("youtube_url"),
        )


@dataclass(slots=True)
class SegmentRecord:
    episode_id: str
    episode_title: str
    segment_id: str
    start_seconds: float
    end_seconds: float
    start_time: str
    end_time: str
    video_start_seconds: float
    video_end_seconds: float
    video_start_time: str
    video_end_time: str
    text_original: str
    text_semantic: str
    text_normalized: str
    speakers: list[str] = field(default_factory=list)
    speaker_keys: list[str] = field(default_factory=list)
    source_segment_indices: list[int] = field(default_factory=list)
    youtube_url: str | None = None
    source_file: str = ""
    source_time_offset_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SegmentRecord":
        start_seconds = float(value.get("start_seconds", 0.0))
        end_seconds = float(value.get("end_seconds", start_seconds))
        offset = float(value.get("source_time_offset_seconds", 0.0))
        video_start = float(value.get("video_start_seconds", start_seconds + offset))
        video_end = float(value.get("video_end_seconds", end_seconds + offset))
        return cls(
            episode_id=str(value["episode_id"]),
            episode_title=str(value.get("episode_title") or "Untitled episode"),
            segment_id=str(value["segment_id"]),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            start_time=str(value.get("start_time") or format_timestamp(start_seconds)),
            end_time=str(value.get("end_time") or format_timestamp(end_seconds)),
            video_start_seconds=video_start,
            video_end_seconds=video_end,
            video_start_time=str(
                value.get("video_start_time") or format_timestamp(video_start)
            ),
            video_end_time=str(
                value.get("video_end_time") or format_timestamp(video_end)
            ),
            text_original=str(value.get("text_original") or ""),
            text_semantic=str(
                value.get("text_semantic") or value.get("text_original") or ""
            ),
            text_normalized=str(value.get("text_normalized") or ""),
            speakers=[str(item) for item in value.get("speakers", [])],
            speaker_keys=[str(item) for item in value.get("speaker_keys", [])],
            source_segment_indices=[
                int(item) for item in value.get("source_segment_indices", [])
            ],
            youtube_url=value.get("youtube_url"),
            source_file=str(value.get("source_file") or ""),
            source_time_offset_seconds=offset,
        )

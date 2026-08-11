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
class SegmentRecord:
    youtube_video_id: str
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

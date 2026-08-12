from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..text import CHARACTER_DISPLAY_NAMES


def format_timestamp(seconds: float) -> str:
    """Format non-negative seconds as HH:MM:SS."""
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SegmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    youtube_video_id: str
    episode_title: str
    segment_id: str
    start_seconds: float
    end_seconds: float
    video_start_seconds: float
    video_end_seconds: float
    text_original: str
    text_semantic: str
    text_normalized: str
    speaker_keys: list[str] = Field(default_factory=list)
    source_segment_indices: list[int] = Field(default_factory=list)
    youtube_url: str | None = None
    source_file: str = ""
    source_time_offset_seconds: float = 0.0

    @property
    def start_time(self) -> str:
        return format_timestamp(self.start_seconds)

    @property
    def end_time(self) -> str:
        return format_timestamp(self.end_seconds)

    @property
    def video_start_time(self) -> str:
        return format_timestamp(self.video_start_seconds)

    @property
    def video_end_time(self) -> str:
        return format_timestamp(self.video_end_seconds)

    @property
    def speakers(self) -> list[str]:
        return [CHARACTER_DISPLAY_NAMES.get(key, key) for key in self.speaker_keys]

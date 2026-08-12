from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..common.models import SegmentRecord


@dataclass(slots=True)
class SearchResult:
    rank: int
    segment: SegmentRecord
    score: float
    score_components: dict[str, float]
    matched_characters: list[str]
    matched_terms: list[str]
    reason: str
    youtube_url_with_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "youtube_video_id": self.segment.youtube_video_id,
            "episode_title": self.segment.episode_title,
            "segment_id": self.segment.segment_id,
            "start_seconds": self.segment.start_seconds,
            "end_seconds": self.segment.end_seconds,
            "start_time": self.segment.video_start_time,
            "end_time": self.segment.video_end_time,
            "transcript_start_time": self.segment.start_time,
            "transcript_end_time": self.segment.end_time,
            "youtube_url": self.youtube_url_with_timestamp,
            "speakers": self.segment.speakers,
            "snippet": self.segment.text_original,
            "score": round(self.score, 6),
            "score_components": {
                key: round(number, 6) for key, number in self.score_components.items()
            },
            "matched_characters": self.matched_characters,
            "matched_terms": self.matched_terms,
            "reason": self.reason,
        }

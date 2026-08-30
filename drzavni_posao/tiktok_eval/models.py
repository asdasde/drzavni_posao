from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessingStatus(StrEnum):
    FETCH_FAILED = "fetch_failed"
    NOT_A_REPLY = "not_a_reply"
    QUERY_NOT_FOUND = "query_not_found"
    QUERY_LOW_CONFIDENCE = "query_low_confidence"
    EPISODE_NOT_FOUND = "episode_not_found"
    EPISODE_NOT_IN_CATALOG = "episode_not_in_catalog"
    EPISODE_NOT_IN_CORPUS = "episode_not_in_corpus"
    DUPLICATE = "duplicate"
    ACCEPTED = "accepted"


def canonical_post_url(post_id: str) -> str:
    """Return TikTok's ID-addressed fallback when the handle is unavailable."""
    return f"https://www.tiktok.com/@_/video/{post_id}"


class PostRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    post_id: str = Field(min_length=1)
    canonical_url: str = Field(min_length=1)

    @field_validator("post_id")
    @classmethod
    def validate_post_id(cls, value: str) -> str:
        value = value.strip()
        if not value.isdigit():
            raise ValueError("TikTok post_id must contain only digits")
        return value


class RawPost(PostRef):
    caption: str = ""
    is_reply: bool = False
    published_at: datetime | None = None
    source_payload: dict[str, Any] = Field(default_factory=dict)


class ExtractedQuery(BaseModel):
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(min_length=1)


class EpisodeMatch(BaseModel):
    episode_id: str = Field(min_length=1)
    youtube_video_id: str | None = None
    title: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    method: str = Field(min_length=1)
    in_catalog: bool = True
    in_corpus: bool = True


class StageError(BaseModel):
    stage: str
    error_type: str
    message: str


class ProcessingRecord(BaseModel):
    """One durable attempt to turn a public post into an eval example."""

    post_id: str
    canonical_url: str
    caption: str = ""
    published_at: datetime | None = None
    status: ProcessingStatus
    outcome: str = "skipped"
    query: ExtractedQuery | None = None
    episode: EpisodeMatch | None = None
    errors: list[StageError] = Field(default_factory=list)
    duplicate_of: str | None = None
    extractor_version: str = "1"
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("outcome", mode="before")
    @classmethod
    def validate_outcome(cls, value: str) -> str:
        if value not in {"accepted", "skipped"}:
            raise ValueError("outcome must be accepted or skipped")
        return value

    @classmethod
    def from_ref(
        cls,
        ref: PostRef,
        status: ProcessingStatus,
        **values: Any,
    ) -> ProcessingRecord:
        return cls(
            post_id=ref.post_id,
            canonical_url=ref.canonical_url,
            status=status,
            outcome="accepted" if status is ProcessingStatus.ACCEPTED else "skipped",
            **values,
        )

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from .discovery import PostSource
from .models import (
    EpisodeMatch,
    ExtractedQuery,
    PostRef,
    ProcessingRecord,
    ProcessingStatus,
    RawPost,
    StageError,
)
from .state import JsonlStateStore

QueryExtractor = Callable[[RawPost], ExtractedQuery | None]
EpisodeExtractor = Callable[[RawPost], EpisodeMatch | None]


class LowConfidenceQueryError(ValueError):
    pass


class QueryNotFoundError(ValueError):
    pass


class EpisodeNotFoundError(ValueError):
    pass


class EpisodeNotInCatalogError(ValueError):
    pass


class EpisodeNotInCorpusError(ValueError):
    pass


@dataclass
class CollectionSummary:
    discovered: int = 0
    resumed: int = 0
    statuses: Counter[ProcessingStatus] = field(default_factory=Counter)

    @property
    def accepted(self) -> int:
        return self.statuses[ProcessingStatus.ACCEPTED]

    @property
    def skipped(self) -> int:
        return sum(self.statuses.values()) - self.accepted


class TikTokEvalCollector:
    def __init__(
        self,
        *,
        source: PostSource,
        state: JsonlStateStore,
        query_extractor: QueryExtractor,
        episode_extractor: EpisodeExtractor,
        extractor_version: str = "1",
        min_query_confidence: float = 0.7,
        min_episode_confidence: float = 0.7,
    ):
        if not 0.0 <= min_query_confidence <= 1.0:
            raise ValueError("min_query_confidence must be between 0 and 1")
        if not 0.0 <= min_episode_confidence <= 1.0:
            raise ValueError("min_episode_confidence must be between 0 and 1")
        self.source = source
        self.state = state
        self.query_extractor = query_extractor
        self.episode_extractor = episode_extractor
        self.extractor_version = extractor_version
        self.min_query_confidence = min_query_confidence
        self.min_episode_confidence = min_episode_confidence

    def collect(
        self, profile_url: str, *, retry_failed: bool = False
    ) -> CollectionSummary:
        summary = CollectionSummary()
        completed = self.state.latest_by_post()
        seen_this_run: set[str] = set()

        for ref in self.source.discover(profile_url):
            summary.discovered += 1
            if ref.post_id in seen_this_run:
                record = ProcessingRecord.from_ref(
                    ref,
                    ProcessingStatus.DUPLICATE,
                    duplicate_of=ref.post_id,
                    extractor_version=self.extractor_version,
                )
                self._persist(record, summary)
                continue
            seen_this_run.add(ref.post_id)

            previous = completed.get(ref.post_id)
            same_extractor = (
                previous is not None
                and previous.extractor_version == self.extractor_version
            )
            should_retry = (
                retry_failed
                and previous is not None
                and previous.status is not ProcessingStatus.ACCEPTED
            )
            if same_extractor and not should_retry:
                summary.resumed += 1
                continue

            record = self._process(ref)
            self._persist(record, summary)
        return summary

    def _persist(self, record: ProcessingRecord, summary: CollectionSummary) -> None:
        self.state.append(record)
        summary.statuses[record.status] += 1

    def _process(self, ref: PostRef) -> ProcessingRecord:
        try:
            post = self.source.fetch(ref)
        except Exception as error:
            return self._failed(ref, ProcessingStatus.FETCH_FAILED, "fetch", error)

        if not post.is_reply:
            return self._record(post, ProcessingStatus.NOT_A_REPLY)

        errors: list[StageError] = []
        query: ExtractedQuery | None = None
        query_status: ProcessingStatus | None = None
        try:
            query = self.query_extractor(post)
        except LowConfidenceQueryError as error:
            query_status = ProcessingStatus.QUERY_LOW_CONFIDENCE
            errors.append(self._stage_error("query", error))
        except Exception as error:
            query_status = ProcessingStatus.QUERY_NOT_FOUND
            errors.append(self._stage_error("query", error))
        else:
            if query is None:
                query_status = ProcessingStatus.QUERY_NOT_FOUND
                errors.append(
                    self._stage_error(
                        "query",
                        QueryNotFoundError("query extractor returned no result"),
                    )
                )
            elif query.confidence < self.min_query_confidence:
                query_status = ProcessingStatus.QUERY_LOW_CONFIDENCE
                errors.append(
                    self._stage_error(
                        "query",
                        LowConfidenceQueryError(
                            f"confidence {query.confidence:.3f} is below "
                            f"{self.min_query_confidence:.3f}"
                        ),
                    )
                )

        episode: EpisodeMatch | None = None
        episode_status: ProcessingStatus | None = None
        try:
            episode = self.episode_extractor(post)
        except EpisodeNotInCatalogError as error:
            episode_status = ProcessingStatus.EPISODE_NOT_IN_CATALOG
            errors.append(self._stage_error("episode", error))
        except EpisodeNotInCorpusError as error:
            episode_status = ProcessingStatus.EPISODE_NOT_IN_CORPUS
            errors.append(self._stage_error("episode", error))
        except Exception as error:
            episode_status = ProcessingStatus.EPISODE_NOT_FOUND
            errors.append(self._stage_error("episode", error))
        else:
            if episode is None:
                episode_status = ProcessingStatus.EPISODE_NOT_FOUND
                errors.append(
                    self._stage_error(
                        "episode",
                        EpisodeNotFoundError("episode extractor returned no result"),
                    )
                )
            elif episode.confidence < self.min_episode_confidence:
                episode_status = ProcessingStatus.EPISODE_NOT_FOUND
                errors.append(
                    self._stage_error(
                        "episode",
                        ValueError(
                            f"confidence {episode.confidence:.3f} is below "
                            f"{self.min_episode_confidence:.3f}"
                        ),
                    )
                )
            elif not episode.in_catalog:
                episode_status = ProcessingStatus.EPISODE_NOT_IN_CATALOG
            elif not episode.in_corpus:
                episode_status = ProcessingStatus.EPISODE_NOT_IN_CORPUS

        # Query failure wins deterministically, while both stage results/errors
        # remain available for audit and later reprocessing.
        status = query_status or episode_status or ProcessingStatus.ACCEPTED
        return self._record(
            post,
            status,
            query=query,
            episode=episode,
            errors=errors,
        )

    @staticmethod
    def _stage_error(stage: str, error: Exception) -> StageError:
        return StageError(
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
        )

    def _record(
        self,
        post: PostRef,
        status: ProcessingStatus,
        **values: object,
    ) -> ProcessingRecord:
        caption = post.caption if isinstance(post, RawPost) else ""
        published_at = post.published_at if isinstance(post, RawPost) else None
        return ProcessingRecord.from_ref(
            post,
            status,
            caption=caption,
            published_at=published_at,
            extractor_version=self.extractor_version,
            **values,
        )

    def _failed(
        self,
        post: PostRef,
        status: ProcessingStatus,
        stage: str,
        error: Exception,
        **values: object,
    ) -> ProcessingRecord:
        return self._record(
            post,
            status,
            errors=[
                StageError(
                    stage=stage,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            ],
            **values,
        )

from __future__ import annotations

from typing import Protocol

from ..common.models import SegmentRecord
from ..indexing.encoders import resolve_model_path


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker(Protocol):
    name: str

    def score(self, query: str, segments: list[SegmentRecord]) -> dict[str, float]: ...


class CrossEncoderReranker:
    """Optional multilingual cross-encoder reranker for a small candidate pool."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
        batch_size: int = 16,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Reranking needs sentence-transformers. Install the 'search' "
                "optional dependencies first."
            ) from error
        self.name = model_name
        self.batch_size = batch_size
        resolved_model_path = resolve_model_path(
            model_name, local_files_only=local_files_only
        )
        try:
            self.model = CrossEncoder(
                resolved_model_path,
                device=device,
                local_files_only=True,
            )
        except TypeError as error:
            raise RuntimeError(
                "The installed sentence-transformers version does not support "
                "offline cross-encoder loading; upgrade it or pass a local model path."
            ) from error

    def score(self, query: str, segments: list[SegmentRecord]) -> dict[str, float]:
        if not segments:
            return {}
        pairs = [(query, segment.text_semantic) for segment in segments]
        predictions = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return {
            segment.segment_id: float(prediction)
            for segment, prediction in zip(segments, predictions)
        }

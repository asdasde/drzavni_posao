from __future__ import annotations

from typing import Protocol

from ..common.device import resolve_torch_device
from ..common.models import SegmentRecord
from ..indexing.encoders import model_revision_from_path, resolve_model_path


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker(Protocol):
    name: str
    revision: str | None
    requested_revision: str | None

    def score(self, query: str, segments: list[SegmentRecord]) -> dict[str, float]: ...


class CrossEncoderReranker:
    """Optional multilingual cross-encoder reranker for a small candidate pool."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        batch_size: int = 50,
        model_revision: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Reranking needs sentence-transformers. Install the 'search' "
                "optional dependencies first."
            ) from error
        self.name = model_name
        self.requested_revision = model_revision
        self.batch_size = batch_size
        self.device = resolve_torch_device(device)
        resolved_model_path = resolve_model_path(
            model_name,
            local_files_only=local_files_only,
            revision=model_revision,
        )
        self.revision = model_revision_from_path(resolved_model_path)
        try:
            self.model = CrossEncoder(
                str(resolved_model_path),
                device=self.device,
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

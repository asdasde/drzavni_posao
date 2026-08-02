from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HybridWeights:
    bm25: float = 0.45
    dense: float = 0.45
    metadata: float = 0.10

    def validate(self) -> None:
        values = (self.bm25, self.dense, self.metadata)
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("Hybrid weights must be non-negative and not all zero")


def normalize_branch_scores(scores: dict[str, float]) -> dict[str, float]:
    """Per-query min-max calibration for otherwise incomparable score scales."""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {segment_id: 1.0 for segment_id in scores}
    return {
        segment_id: (score - low) / (high - low) for segment_id, score in scores.items()
    }

from __future__ import annotations

import hashlib

from .models import SegmentRecord


def segment_corpus_fingerprint(segments: list[SegmentRecord]) -> str:
    """Stable identity for the ordered, normalized conformed corpus."""
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.segment_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.text_normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def semantic_corpus_fingerprint(segments: list[SegmentRecord]) -> str:
    """Stable identity for the ordered text embedded by a vector index."""
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.segment_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.text_semantic.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

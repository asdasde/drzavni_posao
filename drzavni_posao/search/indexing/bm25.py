from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.fingerprints import segment_corpus_fingerprint
from ..common.io import load_segments, write_json
from ..common.models import SegmentRecord
from ..text import tokenize


INDEX_VERSION = 1


@dataclass(slots=True)
class BM25Hit:
    segment_id: str
    score: float


class BM25Index:
    """Small, serializable Okapi BM25 implementation.

    Keeping the baseline in-project makes its tokenization and artifact version
    explicit and avoids pulling a heavyweight search service into a 1k-document
    university corpus.
    """

    def __init__(
        self,
        *,
        segment_ids: list[str],
        term_frequencies: list[dict[str, int]],
        document_frequencies: dict[str, int],
        document_lengths: list[int],
        average_document_length: float,
        corpus_fingerprint: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.segment_ids = segment_ids
        self.term_frequencies = term_frequencies
        self.document_frequencies = document_frequencies
        self.document_lengths = document_lengths
        self.average_document_length = average_document_length
        self.corpus_fingerprint = corpus_fingerprint
        self.k1 = k1
        self.b = b

    @classmethod
    def build(
        cls, segments: list[SegmentRecord], *, k1: float = 1.5, b: float = 0.75
    ) -> "BM25Index":
        if not segments:
            raise ValueError("Cannot build BM25 over an empty corpus")
        term_frequencies: list[dict[str, int]] = []
        document_frequencies: Counter[str] = Counter()
        document_lengths: list[int] = []
        for segment in segments:
            tokens = tokenize(segment.text_normalized)
            frequencies = Counter(tokens)
            term_frequencies.append(dict(frequencies))
            document_frequencies.update(frequencies.keys())
            document_lengths.append(len(tokens))
        return cls(
            segment_ids=[segment.segment_id for segment in segments],
            term_frequencies=term_frequencies,
            document_frequencies=dict(document_frequencies),
            document_lengths=document_lengths,
            average_document_length=sum(document_lengths) / len(document_lengths),
            corpus_fingerprint=segment_corpus_fingerprint(segments),
            k1=k1,
            b=b,
        )

    def score(self, query_tokens: list[str] | tuple[str, ...]) -> list[float]:
        if not query_tokens:
            return [0.0] * len(self.segment_ids)
        document_count = len(self.segment_ids)
        average_length = self.average_document_length or 1.0
        scores = [0.0] * document_count
        # Expansion terms are sets of alternatives, so repeated query terms do
        # not get an accidental extra weight.
        for term in dict.fromkeys(query_tokens):
            document_frequency = self.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                length_normalization = (
                    1.0
                    - self.b
                    + self.b * (self.document_lengths[index] / average_length)
                )
                scores[index] += (
                    inverse_document_frequency
                    * (frequency * (self.k1 + 1.0))
                    / (frequency + self.k1 * length_normalization)
                )
        return scores

    def search(
        self, query_tokens: list[str] | tuple[str, ...], *, top_k: int = 50
    ) -> list[BM25Hit]:
        if top_k < 1:
            return []
        scores = self.score(query_tokens)
        ranked = sorted(
            (
                BM25Hit(segment_id=segment_id, score=score)
                for segment_id, score in zip(self.segment_ids, scores)
                if score > 0.0
            ),
            key=lambda hit: (-hit.score, hit.segment_id),
        )
        return ranked[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": INDEX_VERSION,
            "algorithm": "okapi_bm25",
            "k1": self.k1,
            "b": self.b,
            "segment_ids": self.segment_ids,
            "term_frequencies": self.term_frequencies,
            "document_frequencies": self.document_frequencies,
            "document_lengths": self.document_lengths,
            "average_document_length": self.average_document_length,
            "corpus_fingerprint": self.corpus_fingerprint,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(
        cls, path: str | Path, *, segments: list[SegmentRecord] | None = None
    ) -> "BM25Index":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("version") != INDEX_VERSION:
            raise ValueError(
                f"Unsupported BM25 artifact version in {source}: {value.get('version')}"
            )
        index = cls(
            segment_ids=[str(item) for item in value["segment_ids"]],
            term_frequencies=[
                {str(term): int(count) for term, count in document.items()}
                for document in value["term_frequencies"]
            ],
            document_frequencies={
                str(term): int(count)
                for term, count in value["document_frequencies"].items()
            },
            document_lengths=[int(item) for item in value["document_lengths"]],
            average_document_length=float(value["average_document_length"]),
            corpus_fingerprint=str(value["corpus_fingerprint"]),
            k1=float(value["k1"]),
            b=float(value["b"]),
        )
        if segments is not None:
            fingerprint = segment_corpus_fingerprint(segments)
            if fingerprint != index.corpus_fingerprint:
                raise ValueError(
                    "BM25 index does not match the segment corpus; rebuild the index"
                )
            if index.segment_ids != [segment.segment_id for segment in segments]:
                raise ValueError("BM25 index segment order does not match the corpus")
        return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local Okapi BM25 index")
    parser.add_argument("--segments", default="data/segments.jsonl")
    parser.add_argument("--output", default="artifacts/bm25_index.json")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    segments = load_segments(args.segments)
    index = BM25Index.build(segments, k1=args.k1, b=args.b)
    index.save(args.output)
    print(
        f"Wrote BM25 index for {len(segments)} segments and "
        f"{len(index.document_frequencies)} terms to {args.output}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..common.io import load_segments
from ..common.models import SegmentRecord
from ..query import ProcessedQuery, display_name, process_query
from ..text import tokenize
from ..indexing.bm25 import BM25Index
from ..indexing.encoders import TextEncoder
from ..indexing.vector import VectorIndex, encoder_for_index
from .fusion import HybridWeights, normalize_branch_scores
from .models import SearchResult
from .rerank import CrossEncoderReranker, DEFAULT_RERANKER_MODEL, Reranker


SearchMethod = Literal["bm25", "dense", "hybrid", "cross_encoder", "hybrid_rerank"]
RERANK_BASE_WEIGHT = 0.30
RERANK_MODEL_WEIGHT = 0.70
OVERLAP_DEDUPLICATION_THRESHOLD = 0.60


def timestamped_youtube_url(url: str | None, seconds: float) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = f"{max(0, int(round(seconds)))}s"
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _temporal_overlap_ratio(first: SegmentRecord, second: SegmentRecord) -> float:
    overlap = max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )
    shorter = min(
        max(0.001, first.end_seconds - first.start_seconds),
        max(0.001, second.end_seconds - second.start_seconds),
    )
    return overlap / shorter


class SearchEngine:
    def __init__(
        self,
        segments: list[SegmentRecord],
        bm25_index: BM25Index,
        *,
        vector_index: VectorIndex | None = None,
        text_encoder: TextEncoder | None = None,
        reranker: Reranker | None = None,
        dense_device: str = "cpu",
        local_models_only: bool = True,
    ) -> None:
        if not segments:
            raise ValueError("Search engine needs at least one segment")
        self.segments = segments
        self.by_id = {segment.segment_id: segment for segment in segments}
        if len(self.by_id) != len(segments):
            raise ValueError("Segment IDs must be unique")
        if set(bm25_index.segment_ids) != set(self.by_id):
            raise ValueError("BM25 index and segment corpus contain different IDs")
        if vector_index is not None and set(vector_index.segment_ids) != set(
            self.by_id
        ):
            raise ValueError("Vector index and segment corpus contain different IDs")
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self._text_encoder = text_encoder
        self.reranker = reranker
        self.dense_device = dense_device
        self.local_models_only = local_models_only

    @classmethod
    def from_paths(
        cls,
        *,
        segments_path: Path = Path("data/segments.jsonl"),
        bm25_path: Path = Path("artifacts/bm25_index.json"),
        vector_path: Path | None = None,
        dense_device: str = "cpu",
        local_models_only: bool = True,
        reranker: Reranker | None = None,
    ) -> "SearchEngine":
        segments = load_segments(segments_path)
        bm25 = BM25Index.load(bm25_path, segments=segments)
        vector = (
            VectorIndex.load(vector_path, segments=segments)
            if vector_path is not None
            else None
        )
        return cls(
            segments,
            bm25,
            vector_index=vector,
            reranker=reranker,
            dense_device=dense_device,
            local_models_only=local_models_only,
        )

    def _encoder(self) -> TextEncoder:
        if self.vector_index is None:
            raise RuntimeError(
                "Dense search is unavailable: build and provide a vector index"
            )
        if self._text_encoder is None:
            self._text_encoder = encoder_for_index(
                self.vector_index,
                device=self.dense_device,
                local_files_only=self.local_models_only,
            )
        if self._text_encoder.dimension != self.vector_index.dimension:
            raise ValueError("Dense encoder dimension does not match vector index")
        return self._text_encoder

    @staticmethod
    def _character_score(
        query: ProcessedQuery, segment: SegmentRecord
    ) -> tuple[float, list[str]]:
        if not query.characters:
            return 0.0, []
        searchable = set(segment.speaker_keys) | set(segment.text_normalized.split())
        matched = [
            character for character in query.characters if character in searchable
        ]
        return len(matched) / len(query.characters), matched

    @staticmethod
    def _matched_terms(query: ProcessedQuery, segment: SegmentRecord) -> list[str]:
        document_terms = set(tokenize(segment.text_normalized))
        matches = [
            term
            for term in query.lexical_tokens
            if not term.startswith("__stem_") and term in document_terms
        ]
        return list(dict.fromkeys(matches))[:10]

    @staticmethod
    def _reason(
        method: SearchMethod,
        matched_characters: list[str],
        matched_terms: list[str],
        dense_score: float,
        rerank_score: float | None,
    ) -> str:
        evidence: list[str] = []
        if matched_characters:
            names = ", ".join(display_name(item) for item in matched_characters)
            evidence.append(f"matches requested character metadata/text ({names})")
        if matched_terms:
            evidence.append(f"shares lexical evidence ({', '.join(matched_terms[:6])})")
        if method in {"dense", "hybrid", "hybrid_rerank"} and dense_score >= 0.5:
            evidence.append("has a strong relative multilingual semantic match")
        if method == "cross_encoder" and rerank_score is not None:
            evidence.append("is a strong exhaustive cross-encoder match")
        elif method == "hybrid_rerank" and rerank_score is not None:
            evidence.append("was promoted by the cross-encoder reranker")
        return (
            "; ".join(evidence)
            if evidence
            else "best relative match in the candidate set"
        )

    def search(
        self,
        query_text: str,
        *,
        method: SearchMethod = "bm25",
        top_k: int = 10,
        candidate_k: int = 50,
        weights: HybridWeights = HybridWeights(),
        group_by_episode: bool = True,
        deduplicate_overlaps: bool = True,
        rerank_k: int = 50,
    ) -> list[SearchResult]:
        if method not in {
            "bm25",
            "dense",
            "hybrid",
            "cross_encoder",
            "hybrid_rerank",
        }:
            raise ValueError(f"Unknown search method: {method}")
        if not query_text.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")
        if method == "hybrid_rerank" and rerank_k < top_k:
            raise ValueError("rerank_k must be at least top_k for hybrid reranking")
        weights.validate()
        query = process_query(query_text)

        raw_bm25: dict[str, float] = {}
        raw_dense: dict[str, float] = {}
        if method in {"bm25", "hybrid", "hybrid_rerank"}:
            raw_bm25 = {
                hit.segment_id: hit.score
                for hit in self.bm25_index.search(
                    query.lexical_tokens, top_k=candidate_k
                )
            }
        if method in {"dense", "hybrid", "hybrid_rerank"}:
            if self.vector_index is None:
                raise RuntimeError(
                    f"Method '{method}' requires a vector index; build it first"
                )
            query_vector = self._encoder().encode_query(query.semantic)
            raw_dense = {
                hit.segment_id: hit.score
                for hit in self.vector_index.search_vector(
                    query_vector, top_k=candidate_k
                )
            }

        candidate_ids = (
            set(self.by_id)
            if method == "cross_encoder"
            else set(raw_bm25) | set(raw_dense)
        )
        if not candidate_ids:
            return []
        bm25_scores = normalize_branch_scores(raw_bm25)
        dense_scores = normalize_branch_scores(raw_dense)
        base_scores: dict[str, float] = {}
        character_scores: dict[str, float] = {}
        character_matches: dict[str, list[str]] = {}
        for segment_id in candidate_ids:
            segment = self.by_id[segment_id]
            metadata_score, matches = self._character_score(query, segment)
            character_scores[segment_id] = metadata_score
            character_matches[segment_id] = matches
            if method == "bm25":
                score = bm25_scores.get(segment_id, 0.0)
            elif method == "dense":
                score = dense_scores.get(segment_id, 0.0)
            elif method == "cross_encoder":
                score = 0.0
            else:
                score = (
                    weights.bm25 * bm25_scores.get(segment_id, 0.0)
                    + weights.dense * dense_scores.get(segment_id, 0.0)
                    + weights.metadata * metadata_score
                )
            base_scores[segment_id] = score

        reranker_scores: dict[str, float] = {}
        final_scores = dict(base_scores)
        rankable_ids: set[str] | list[str] = candidate_ids
        if method == "cross_encoder":
            if self.reranker is None:
                raise RuntimeError(
                    "cross_encoder requires a configured cross-encoder model"
                )
            raw_reranker = self.reranker.score(query.semantic, self.segments)
            reranker_scores = normalize_branch_scores(raw_reranker)
            final_scores = dict(reranker_scores)
        elif method == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError(
                    "hybrid_rerank requires a configured cross-encoder reranker"
                )
            preliminary = sorted(
                candidate_ids, key=lambda item: (-base_scores[item], item)
            )[:rerank_k]
            raw_reranker = self.reranker.score(
                query.semantic, [self.by_id[item] for item in preliminary]
            )
            reranker_scores = normalize_branch_scores(raw_reranker)
            final_scores = {
                segment_id: RERANK_BASE_WEIGHT * base_scores[segment_id]
                + RERANK_MODEL_WEIGHT * reranker_scores.get(segment_id, 0.0)
                for segment_id in preliminary
            }
            rankable_ids = preliminary

        ranked_ids = sorted(rankable_ids, key=lambda item: (-final_scores[item], item))
        selected_ids: list[str] = []
        used_episodes: set[str] = set()
        selected_segments: list[SegmentRecord] = []
        for segment_id in ranked_ids:
            segment = self.by_id[segment_id]
            if group_by_episode and segment.episode_id in used_episodes:
                continue
            if deduplicate_overlaps and any(
                existing.episode_id == segment.episode_id
                and _temporal_overlap_ratio(existing, segment)
                >= OVERLAP_DEDUPLICATION_THRESHOLD
                for existing in selected_segments
            ):
                continue
            selected_ids.append(segment_id)
            selected_segments.append(segment)
            used_episodes.add(segment.episode_id)
            if len(selected_ids) >= top_k:
                break

        results: list[SearchResult] = []
        for rank, segment_id in enumerate(selected_ids, start=1):
            segment = self.by_id[segment_id]
            matched_terms = self._matched_terms(query, segment)
            normalized_dense = dense_scores.get(segment_id, 0.0)
            normalized_reranker = reranker_scores.get(segment_id)
            results.append(
                SearchResult(
                    rank=rank,
                    segment=segment,
                    score=final_scores[segment_id],
                    score_components={
                        "bm25": bm25_scores.get(segment_id, 0.0),
                        "dense": normalized_dense,
                        "metadata": character_scores[segment_id],
                        "bm25_raw": raw_bm25.get(segment_id, 0.0),
                        "dense_raw": raw_dense.get(segment_id, 0.0),
                        **(
                            {
                                (
                                    "cross_encoder"
                                    if method == "cross_encoder"
                                    else "reranker"
                                ): normalized_reranker
                            }
                            if normalized_reranker is not None
                            else {}
                        ),
                    },
                    matched_characters=[
                        display_name(item) for item in character_matches[segment_id]
                    ],
                    matched_terms=matched_terms,
                    reason=self._reason(
                        method,
                        character_matches[segment_id],
                        matched_terms,
                        normalized_dense,
                        normalized_reranker,
                    ),
                    youtube_url_with_timestamp=timestamped_youtube_url(
                        segment.youtube_url, segment.video_start_seconds
                    ),
                )
            )
        return results


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        print("No results.")
        return
    for result in results:
        segment = result.segment
        print(
            f"{result.rank}. {segment.episode_id} — {segment.episode_title}\n"
            f"   Time: {segment.video_start_time}–{segment.video_end_time}\n"
            f"   Characters: {', '.join(segment.speakers) or 'unknown'}\n"
            f"   Relative score: {result.score:.3f}\n"
            f"   Reason: {result.reason}"
        )
        if result.youtube_url_with_timestamp:
            print(f"   YouTube: {result.youtube_url_with_timestamp}")
        print("   Transcript:")
        for line in segment.text_original.splitlines():
            print(f"     {line}")
        print()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Drzavni posao scenes")
    parser.add_argument("query")
    parser.add_argument("--segments", type=Path, default=Path("data/segments.jsonl"))
    parser.add_argument("--bm25", type=Path, default=Path("artifacts/bm25_index.json"))
    parser.add_argument(
        "--vectors", type=Path, help="Vector artifact stem or metadata JSON"
    )
    parser.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid", "cross_encoder", "hybrid_rerank"),
        default="bm25",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--reranker-revision",
        help="Optional Hugging Face commit, tag, or branch to pin",
    )
    parser.add_argument("--all-segments", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    reranker = None
    if args.method in {"cross_encoder", "hybrid_rerank"}:
        reranker = CrossEncoderReranker(
            args.reranker_model,
            device=args.device,
            local_files_only=not args.allow_model_download,
            model_revision=args.reranker_revision,
        )
    engine = SearchEngine.from_paths(
        segments_path=args.segments,
        bm25_path=args.bm25,
        vector_path=args.vectors,
        dense_device=args.device,
        local_models_only=not args.allow_model_download,
        reranker=reranker,
    )
    results = engine.search(
        args.query,
        method=args.method,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        group_by_episode=not args.all_segments,
    )
    if args.json_output:
        print(
            json.dumps(
                [result.to_dict() for result in results], ensure_ascii=False, indent=2
            )
        )
    else:
        _print_results(results)


if __name__ == "__main__":
    main()

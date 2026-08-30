from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from ..common.device import resolve_torch_device
from ..common.fingerprints import segment_corpus_fingerprint
from ..common.io import read_jsonl, write_json
from ..common.models import SegmentRecord
from ..indexing.vector import VectorIndex
from ..retrieval.engine import (
    OVERLAP_DEDUPLICATION_THRESHOLD,
    RERANK_BASE_WEIGHT,
    RERANK_MODEL_WEIGHT,
    SearchEngine,
    SearchMethod,
)
from ..retrieval.fusion import HybridWeights
from ..retrieval.models import SearchResult
from ..retrieval.rerank import CrossEncoderReranker, DEFAULT_RERANKER_MODEL


logger = logging.getLogger(__name__)

TimeReference = Literal["transcript", "video"]
RelevanceLevel = Literal["scene", "episode"]
EVALUATION_TOP_K = 10
EVALUATION_RERANK_K = 50
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEEDS = {
    "recall_at_5": 13,
    "mrr_at_10": 17,
    "ndcg_at_10": 19,
}


def parse_timestamp(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.strip().split(":")
    if not parts or any(not part.replace(".", "", 1).isdigit() for part in parts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    numbers = [float(part) for part in parts]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0]
    raise ValueError(f"Invalid timestamp: {value!r}")


@dataclass(frozen=True, slots=True)
class Judgment:
    youtube_video_id: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    grade: int = 1
    segment_id: str | None = None
    time_reference: TimeReference = "video"


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    judgments: tuple[Judgment, ...]
    language: str = "unspecified"
    category: str = "unspecified"
    split: str = "test"


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    recall_at_5: float
    reciprocal_rank_at_10: float
    ndcg_at_10: float


@dataclass(slots=True)
class EvaluationSummary:
    method: str
    query_count: int
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    mean_latency_ms: float
    p95_latency_ms: float
    confidence_intervals_95: dict[str, list[float]]
    per_query: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_query_fingerprint(queries: list[EvaluationQuery]) -> str:
    """Fingerprint the exact ordered queries and judgments being evaluated."""
    encoded = json.dumps(
        [asdict(query) for query in queries],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _vector_artifact_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    metadata_path, default_vectors_path = VectorIndex.artifact_paths(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors_path = metadata_path.parent / metadata.get(
        "vectors_file", default_vectors_path.name
    )
    return {
        "metadata": _file_manifest(metadata_path),
        "vectors": _file_manifest(vectors_path),
    }


def _package_version() -> str | None:
    try:
        return version("drzavni-posao")
    except PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def build_evaluation_manifest(
    *,
    engine: SearchEngine,
    methods: list[str],
    queries: list[EvaluationQuery],
    query_files: list[Path],
    segments_path: Path,
    bm25_path: Path,
    vector_path: Path | None,
    split: str,
    ranking_unit: Literal["segment", "episode"],
    relevance_level: RelevanceLevel = "scene",
    candidate_k: int,
    tolerance_seconds: float,
    bootstrap_samples: int,
    warmup: bool,
    weights: HybridWeights,
    top_k: int = EVALUATION_TOP_K,
    rerank_k: int = EVALUATION_RERANK_K,
) -> dict[str, Any]:
    """Describe all inputs and settings needed to audit an evaluation run."""
    weights.validate()
    effective_ranking_unit = "episode" if relevance_level == "episode" else ranking_unit
    vector_index = engine.vector_index
    reranker = engine.reranker
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "package": "drzavni-posao",
            "package_version": _package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git": _git_state(),
        },
        "evaluation": {
            "methods": methods,
            "split": split,
            "ranking_unit": effective_ranking_unit,
            "requested_ranking_unit": ranking_unit,
            "relevance_level": relevance_level,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "group_by_episode": (effective_ranking_unit == "episode"),
            "deduplicate_overlaps": (effective_ranking_unit == "episode"),
            "overlap_deduplication_threshold": (OVERLAP_DEDUPLICATION_THRESHOLD),
            "tolerance_seconds": tolerance_seconds,
            "warmup": warmup,
            "hybrid_weights": asdict(weights),
            "rerank_blend_weights": {
                "base": RERANK_BASE_WEIGHT,
                "model": RERANK_MODEL_WEIGHT,
            },
            "metrics": {
                "recall_cutoff": 5,
                "reciprocal_rank_cutoff": 10,
                "ndcg_cutoff": 10,
            },
            "bootstrap": {
                "samples": bootstrap_samples,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seeds": BOOTSTRAP_SEEDS,
            },
        },
        "data": {
            "query_count": len(queries),
            "query_set_fingerprint": evaluation_query_fingerprint(queries),
            "corpus_fingerprint": segment_corpus_fingerprint(engine.segments),
            "semantic_corpus_fingerprint": (
                vector_index.semantic_corpus_fingerprint
                if vector_index is not None
                else None
            ),
        },
        "models": {
            "vector": (
                {
                    "name": vector_index.model_name,
                    "revision": vector_index.model_revision,
                    "requested_revision": getattr(
                        vector_index, "model_requested_revision", None
                    ),
                    "dimension": vector_index.dimension,
                    "uses_e5_prefixes": vector_index.uses_e5_prefixes,
                }
                if vector_index is not None
                else None
            ),
            "reranker": (
                {
                    "name": reranker.name,
                    "revision": getattr(reranker, "revision", None),
                    "requested_revision": getattr(reranker, "requested_revision", None),
                    "batch_size": getattr(reranker, "batch_size", None),
                }
                if reranker is not None
                else None
            ),
            "device": engine.dense_device,
            "local_models_only": engine.local_models_only,
        },
        "artifacts": {
            "queries": [_file_manifest(path) for path in query_files],
            "segments": _file_manifest(segments_path),
            "bm25_index": _file_manifest(bm25_path),
            "vector_index": _vector_artifact_manifest(vector_path),
        },
    }


def _parse_judgment(
    value: dict[str, Any],
    default_reference: str,
    *,
    relevance_level: RelevanceLevel = "scene",
) -> Judgment:
    if "youtube_video_id" not in value:
        raise ValueError("Every relevance judgment needs youtube_video_id")
    grade = int(value.get("grade", value.get("relevance", 1)))
    if grade <= 0:
        raise ValueError("Relevance grades must be positive integers")
    reference = str(value.get("time_reference", default_reference))
    if reference not in {"transcript", "video"}:
        raise ValueError("time_reference must be 'transcript' or 'video'")
    start = parse_timestamp(value.get("start_seconds", value.get("start_time")))
    end = parse_timestamp(value.get("end_seconds", value.get("end_time")))
    if (start is None) != (end is None):
        raise ValueError("A judgment must provide both start and end, or neither")
    if start is not None and end is not None and end < start:
        raise ValueError("A judgment end cannot precede its start")
    segment_id = value.get("segment_id")
    if relevance_level == "scene" and start is None and segment_id is None:
        raise ValueError(
            "A scene judgment needs segment_id or a start/end interval; "
            "episode-only labels are too coarse"
        )
    return Judgment(
        youtube_video_id=str(value["youtube_video_id"]),
        start_seconds=start,
        end_seconds=end,
        grade=grade,
        segment_id=str(segment_id) if segment_id is not None else None,
        time_reference=reference,  # type: ignore[arg-type]
    )


def load_evaluation_queries(
    path: Path, *, relevance_level: RelevanceLevel = "scene"
) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), start=1):
        query_id = str(row.get("query_id") or row.get("id") or f"q{row_number:03d}")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate evaluation query id: {query_id}")
        seen_ids.add(query_id)
        query_text = str(row.get("query") or "").strip()
        if not query_text:
            raise ValueError(f"Evaluation query {query_id} is empty")
        default_reference = str(row.get("time_reference", "video"))
        raw_judgments = row.get("judgments", row.get("relevant"))
        if raw_judgments is None and row.get("relevant_youtube_video_id"):
            raw_judgments = [
                {
                    "youtube_video_id": row["relevant_youtube_video_id"],
                    "start_time": row.get("relevant_segment_start"),
                    "end_time": row.get("relevant_segment_end"),
                    "time_reference": default_reference,
                }
            ]
        if not isinstance(raw_judgments, list) or not raw_judgments:
            raise ValueError(f"Evaluation query {query_id} has no judgments")
        judgments = tuple(
            _parse_judgment(value, default_reference, relevance_level=relevance_level)
            for value in raw_judgments
        )
        queries.append(
            EvaluationQuery(
                query_id=query_id,
                query=query_text,
                judgments=judgments,
                language=str(row.get("language", "unspecified")),
                category=str(row.get("category", "unspecified")),
                split=str(row.get("split", "test")),
            )
        )
    if not queries:
        raise ValueError(f"No evaluation queries found in {path}")
    return queries


def load_evaluation_query_files(
    paths: list[Path], *, relevance_level: RelevanceLevel = "scene"
) -> list[EvaluationQuery]:
    """Load multiple evaluation files while enforcing globally unique query IDs."""
    queries: list[EvaluationQuery] = []
    seen_ids: set[str] = set()
    for path in paths:
        for query in load_evaluation_queries(path, relevance_level=relevance_level):
            if query.query_id in seen_ids:
                raise ValueError(
                    f"Duplicate evaluation query id across files: {query.query_id}"
                )
            seen_ids.add(query.query_id)
            queries.append(query)
    return queries


def judgment_matches_segment(
    judgment: Judgment,
    segment: SegmentRecord,
    *,
    tolerance_seconds: float = 2.0,
    relevance_level: RelevanceLevel = "scene",
) -> bool:
    if judgment.youtube_video_id != segment.youtube_video_id:
        return False
    if relevance_level == "episode":
        return True
    if judgment.segment_id is not None:
        return judgment.segment_id == segment.segment_id
    assert judgment.start_seconds is not None and judgment.end_seconds is not None
    if judgment.time_reference == "video":
        start, end = segment.video_start_seconds, segment.video_end_seconds
    else:
        start, end = segment.start_seconds, segment.end_seconds
    return (
        end >= judgment.start_seconds - tolerance_seconds
        and start <= judgment.end_seconds + tolerance_seconds
    )


def relevance_grades(
    results: list[SearchResult],
    judgments: tuple[Judgment, ...],
    *,
    tolerance_seconds: float = 2.0,
    relevance_level: RelevanceLevel = "scene",
) -> tuple[list[int], set[int]]:
    """Greedily map each retrieved hit to at most one still-unmatched judgment."""
    unmatched = set(range(len(judgments)))
    grades: list[int] = []
    matched: set[int] = set()
    for result in results:
        candidates = [
            index
            for index in unmatched
            if judgment_matches_segment(
                judgments[index],
                result.segment,
                tolerance_seconds=tolerance_seconds,
                relevance_level=relevance_level,
            )
        ]
        if not candidates:
            grades.append(0)
            continue
        best = max(candidates, key=lambda index: (judgments[index].grade, -index))
        grades.append(judgments[best].grade)
        unmatched.remove(best)
        matched.add(best)
    return grades, matched


def dcg(grades: list[int], k: int) -> float:
    return sum(
        ((2**grade) - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], start=1)
        if grade > 0
    )


def calculate_query_metrics(
    results: list[SearchResult],
    judgments: tuple[Judgment, ...],
    *,
    tolerance_seconds: float = 2.0,
    relevance_level: RelevanceLevel = "scene",
) -> QueryMetrics:
    if relevance_level == "episode":
        seen_episodes: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in results:
            episode_id = result.segment.youtube_video_id
            if episode_id in seen_episodes:
                continue
            seen_episodes.add(episode_id)
            unique_results.append(result)
        results = unique_results
        judgments_by_episode: dict[str, Judgment] = {}
        for judgment in judgments:
            existing = judgments_by_episode.get(judgment.youtube_video_id)
            if existing is None or judgment.grade > existing.grade:
                judgments_by_episode[judgment.youtube_video_id] = judgment
        judgments = tuple(judgments_by_episode.values())
    grades, matched = relevance_grades(
        results,
        judgments,
        tolerance_seconds=tolerance_seconds,
        relevance_level=relevance_level,
    )
    top_five_grades, top_five_matched = relevance_grades(
        results[:5],
        judgments,
        tolerance_seconds=tolerance_seconds,
        relevance_level=relevance_level,
    )
    del top_five_grades
    recall = len(top_five_matched) / len(judgments)
    first_relevant = next(
        (rank for rank, grade in enumerate(grades[:10], start=1) if grade > 0),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant if first_relevant is not None else 0.0
    ideal_grades = sorted((judgment.grade for judgment in judgments), reverse=True)
    ideal = dcg(ideal_grades, 10)
    ndcg = dcg(grades, 10) / ideal if ideal > 0 else 0.0
    del matched
    return QueryMetrics(recall, reciprocal_rank, ndcg)


def _bootstrap_interval(
    values: list[float], *, samples: int, seed: int = 13
) -> list[float]:
    if samples <= 0 or len(values) < 2:
        mean = statistics.fmean(values)
        return [mean, mean]
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [lower, upper]


def evaluate_method(
    engine: SearchEngine,
    queries: list[EvaluationQuery],
    method: SearchMethod,
    *,
    candidate_k: int = 50,
    ranking_unit: Literal["segment", "episode"] = "segment",
    relevance_level: RelevanceLevel = "scene",
    tolerance_seconds: float = 2.0,
    bootstrap_samples: int = 1000,
    top_k: int = EVALUATION_TOP_K,
    rerank_k: int = EVALUATION_RERANK_K,
    weights: HybridWeights = HybridWeights(),
) -> EvaluationSummary:
    effective_ranking_unit = "episode" if relevance_level == "episode" else ranking_unit
    per_query: list[dict[str, Any]] = []
    metrics: list[QueryMetrics] = []
    latencies: list[float] = []
    for item in queries:
        started = time.perf_counter()
        results = engine.search(
            item.query,
            method=method,
            top_k=top_k,
            candidate_k=candidate_k,
            weights=weights,
            group_by_episode=effective_ranking_unit == "episode",
            deduplicate_overlaps=effective_ranking_unit == "episode",
            rerank_k=rerank_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        score = calculate_query_metrics(
            results,
            item.judgments,
            tolerance_seconds=tolerance_seconds,
            relevance_level=relevance_level,
        )
        metrics.append(score)
        latencies.append(elapsed_ms)
        per_query.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "language": item.language,
                "category": item.category,
                "recall_at_5": score.recall_at_5,
                "reciprocal_rank_at_10": score.reciprocal_rank_at_10,
                "ndcg_at_10": score.ndcg_at_10,
                "latency_ms": elapsed_ms,
                "result_segment_ids": [result.segment.segment_id for result in results],
            }
        )
    recall_values = [item.recall_at_5 for item in metrics]
    reciprocal_values = [item.reciprocal_rank_at_10 for item in metrics]
    ndcg_values = [item.ndcg_at_10 for item in metrics]
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return EvaluationSummary(
        method=method,
        query_count=len(queries),
        recall_at_5=statistics.fmean(recall_values),
        mrr_at_10=statistics.fmean(reciprocal_values),
        ndcg_at_10=statistics.fmean(ndcg_values),
        mean_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=sorted(latencies)[p95_index],
        confidence_intervals_95={
            "recall_at_5": _bootstrap_interval(
                recall_values,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEEDS["recall_at_5"],
            ),
            "mrr_at_10": _bootstrap_interval(
                reciprocal_values,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEEDS["mrr_at_10"],
            ),
            "ndcg_at_10": _bootstrap_interval(
                ndcg_values,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEEDS["ndcg_at_10"],
            ),
        },
        per_query=per_query,
    )


def _log_table(summaries: list[EvaluationSummary]) -> None:
    lines = ["Method                 Recall@5   MRR@10   nDCG@10   Mean ms   P95 ms"]
    lines.extend(
        f"{summary.method:<22} "
        f"{summary.recall_at_5:>8.3f} "
        f"{summary.mrr_at_10:>8.3f} "
        f"{summary.ndcg_at_10:>9.3f} "
        f"{summary.mean_latency_ms:>9.1f} "
        f"{summary.p95_latency_ms:>8.1f}"
        for summary in summaries
    )
    logger.info("%s", "\n".join(lines))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25, dense, hybrid, and reranked scene retrieval"
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("data/evaluation_queries.jsonl")
    )
    parser.add_argument(
        "--additional-queries",
        action="append",
        type=Path,
        default=[],
        help="Additional JSONL query file; may be supplied more than once",
    )
    parser.add_argument("--segments", type=Path, default=Path("data/segments.jsonl"))
    parser.add_argument("--bm25", type=Path, default=Path("artifacts/bm25_index.json"))
    parser.add_argument(
        "--vectors", type=Path, help="Vector artifact stem or metadata JSON"
    )
    parser.add_argument("--methods", default="bm25", help="Comma-separated methods")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--ranking-unit", choices=("segment", "episode"), default="segment"
    )
    parser.add_argument(
        "--relevance-level",
        choices=("scene", "episode"),
        default="scene",
        help="Match judgments by scene interval/segment, or by YouTube episode ID",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one unmeasured query per method before collecting latency metrics",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto (cuda→mps→cpu), or an explicit device string",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--reranker-revision",
        help="Optional Hugging Face commit, tag, or branch to pin",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation_results.json")
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argument_parser().parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    allowed = {"bm25", "dense", "hybrid", "cross_encoder", "hybrid_rerank"}
    invalid = set(methods) - allowed
    if invalid:
        raise ValueError(f"Unknown evaluation methods: {sorted(invalid)}")
    query_files = [args.queries, *args.additional_queries]
    queries = [
        item
        for item in load_evaluation_query_files(
            query_files, relevance_level=args.relevance_level
        )
        if item.split == args.split
    ]
    if not queries:
        raise ValueError(f"No evaluation queries in split {args.split!r}")
    effective_ranking_unit = (
        "episode" if args.relevance_level == "episode" else args.ranking_unit
    )

    device = resolve_torch_device(args.device)
    reranker = None
    weights = HybridWeights()
    if {"cross_encoder", "hybrid_rerank"} & set(methods):
        reranker = CrossEncoderReranker(
            args.reranker_model,
            device=device,
            local_files_only=not args.allow_model_download,
            model_revision=args.reranker_revision,
        )
    engine = SearchEngine.from_paths(
        segments_path=args.segments,
        bm25_path=args.bm25,
        vector_path=args.vectors,
        dense_device=device,
        local_models_only=not args.allow_model_download,
        reranker=reranker,
    )
    if args.warmup:
        for method in methods:
            engine.search(
                queries[0].query,
                method=method,  # type: ignore[arg-type]
                top_k=EVALUATION_TOP_K,
                candidate_k=args.candidate_k,
                weights=weights,
                group_by_episode=effective_ranking_unit == "episode",
                deduplicate_overlaps=effective_ranking_unit == "episode",
                rerank_k=EVALUATION_RERANK_K,
            )
    summaries = [
        evaluate_method(
            engine,
            queries,
            method,  # type: ignore[arg-type]
            candidate_k=args.candidate_k,
            ranking_unit=args.ranking_unit,
            relevance_level=args.relevance_level,
            tolerance_seconds=args.tolerance_seconds,
            bootstrap_samples=args.bootstrap_samples,
            top_k=EVALUATION_TOP_K,
            rerank_k=EVALUATION_RERANK_K,
            weights=weights,
        )
        for method in methods
    ]
    _log_table(summaries)
    write_json(
        args.output,
        {
            "query_file": str(args.queries),
            "query_files": [str(path) for path in query_files],
            "split": args.split,
            "ranking_unit": effective_ranking_unit,
            "requested_ranking_unit": args.ranking_unit,
            "relevance_level": args.relevance_level,
            "candidate_k": args.candidate_k,
            "tolerance_seconds": args.tolerance_seconds,
            "warmup": args.warmup,
            "manifest": build_evaluation_manifest(
                engine=engine,
                methods=methods,
                queries=queries,
                query_files=query_files,
                segments_path=args.segments,
                bm25_path=args.bm25,
                vector_path=args.vectors,
                split=args.split,
                ranking_unit=args.ranking_unit,
                relevance_level=args.relevance_level,
                candidate_k=args.candidate_k,
                tolerance_seconds=args.tolerance_seconds,
                bootstrap_samples=args.bootstrap_samples,
                warmup=args.warmup,
                weights=weights,
            ),
            "results": [summary.to_dict() for summary in summaries],
        },
    )
    logger.info("\nWrote actual per-query results to %s", args.output)


if __name__ == "__main__":
    main()

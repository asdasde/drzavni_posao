from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..common.device import resolve_torch_device
from ..common.fingerprints import segment_corpus_fingerprint
from ..common.io import read_jsonl, write_json
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

EVALUATION_TOP_K = 10
EVALUATION_RERANK_K = 50
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEEDS = {
    "recall_at_5": 13,
    "mrr_at_10": 17,
    "ndcg_at_10": 19,
}


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_youtube_video_id: str
    tiktok_url: str | None = None


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
    query_file: Path,
    segments_path: Path,
    bm25_path: Path,
    vector_path: Path | None,
    candidate_k: int,
    bootstrap_samples: int,
    warmup: bool,
    weights: HybridWeights,
    top_k: int = EVALUATION_TOP_K,
    rerank_k: int = EVALUATION_RERANK_K,
) -> dict[str, Any]:
    """Describe all inputs and settings needed to audit an evaluation run."""
    weights.validate()
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
            "ranking_unit": "episode",
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "group_by_episode": True,
            "deduplicate_overlaps": True,
            "overlap_deduplication_threshold": (OVERLAP_DEDUPLICATION_THRESHOLD),
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
            "queries": _file_manifest(query_file),
            "segments": _file_manifest(segments_path),
            "bm25_index": _file_manifest(bm25_path),
            "vector_index": _vector_artifact_manifest(vector_path),
        },
    }


def load_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    seen_ids: set[str] = set()
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("Every evaluation query needs query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate evaluation query id: {query_id}")
        seen_ids.add(query_id)
        query_text = str(row.get("query") or "").strip()
        if not query_text:
            raise ValueError(f"Evaluation query {query_id} is empty")
        relevant_video_id = str(row.get("relevant_youtube_video_id") or "").strip()
        if not relevant_video_id:
            raise ValueError(
                f"Evaluation query {query_id} needs relevant_youtube_video_id"
            )
        tiktok_url = str(row.get("tiktok_url") or "").strip() or None
        if tiktok_url is not None and not tiktok_url.startswith(
            "https://www.tiktok.com/"
        ):
            raise ValueError(f"Evaluation query {query_id} has an invalid TikTok URL")
        queries.append(
            EvaluationQuery(
                query_id=query_id,
                query=query_text,
                relevant_youtube_video_id=relevant_video_id,
                tiktok_url=tiktok_url,
            )
        )
    if not queries:
        raise ValueError(f"No evaluation queries found in {path}")
    return queries


def calculate_query_metrics(
    results: list[SearchResult],
    relevant_youtube_video_id: str,
) -> QueryMetrics:
    seen_episodes: set[str] = set()
    ranked_episode_ids: list[str] = []
    for result in results:
        episode_id = result.segment.youtube_video_id
        if episode_id not in seen_episodes:
            seen_episodes.add(episode_id)
            ranked_episode_ids.append(episode_id)
    relevant_rank = next(
        (
            rank
            for rank, episode_id in enumerate(ranked_episode_ids[:10], start=1)
            if episode_id == relevant_youtube_video_id
        ),
        None,
    )
    return QueryMetrics(
        recall_at_5=float(relevant_rank is not None and relevant_rank <= 5),
        reciprocal_rank_at_10=1 / relevant_rank if relevant_rank is not None else 0.0,
        ndcg_at_10=(
            1 / math.log2(relevant_rank + 1) if relevant_rank is not None else 0.0
        ),
    )


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


def _evaluation_checkpoint_fingerprint(
    engine: SearchEngine,
    queries: list[EvaluationQuery],
    method: SearchMethod,
    *,
    candidate_k: int,
    top_k: int,
    rerank_k: int,
    weights: HybridWeights,
) -> str:
    reranker = engine.reranker
    payload = {
        "method": method,
        "queries": evaluation_query_fingerprint(queries),
        "corpus": segment_corpus_fingerprint(engine.segments),
        "candidate_k": candidate_k,
        "ranking_unit": "episode",
        "top_k": top_k,
        "rerank_k": rerank_k,
        "weights": asdict(weights),
        "reranker": (
            {
                "name": reranker.name,
                "revision": reranker.revision,
                "batch_size": getattr(reranker, "batch_size", None),
            }
            if reranker is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_method(
    engine: SearchEngine,
    queries: list[EvaluationQuery],
    method: SearchMethod,
    *,
    candidate_k: int = 50,
    bootstrap_samples: int = 1000,
    top_k: int = EVALUATION_TOP_K,
    rerank_k: int = EVALUATION_RERANK_K,
    weights: HybridWeights = HybridWeights(),
    checkpoint_path: Path | None = None,
) -> EvaluationSummary:
    checkpoint_fingerprint = _evaluation_checkpoint_fingerprint(
        engine,
        queries,
        method,
        candidate_k=candidate_k,
        top_k=top_k,
        rerank_k=rerank_k,
        weights=weights,
    )
    expected_query_ids = {query.query_id for query in queries}
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        for record in read_jsonl(checkpoint_path):
            if record.get("run_fingerprint") != checkpoint_fingerprint:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} belongs to a different evaluation run"
                )
            query_id = str(record.get("query_id") or "")
            if query_id not in expected_query_ids:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} contains unknown query {query_id!r}"
                )
            checkpoint_rows[query_id] = record

    per_query: list[dict[str, Any]] = []
    metrics: list[QueryMetrics] = []
    latencies: list[float] = []
    for index, item in enumerate(queries, start=1):
        checkpoint_row = checkpoint_rows.get(item.query_id)
        if checkpoint_row is None:
            started = time.perf_counter()
            results = engine.search(
                item.query,
                method=method,
                top_k=top_k,
                candidate_k=candidate_k,
                weights=weights,
                group_by_episode=True,
                deduplicate_overlaps=True,
                rerank_k=rerank_k,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            score = calculate_query_metrics(results, item.relevant_youtube_video_id)
            checkpoint_row = {
                "run_fingerprint": checkpoint_fingerprint,
                "query_id": item.query_id,
                "query": item.query,
                "relevant_youtube_video_id": item.relevant_youtube_video_id,
                "tiktok_url": item.tiktok_url,
                "recall_at_5": score.recall_at_5,
                "reciprocal_rank_at_10": score.reciprocal_rank_at_10,
                "ndcg_at_10": score.ndcg_at_10,
                "latency_ms": elapsed_ms,
                "result_segment_ids": [result.segment.segment_id for result in results],
            }
            if checkpoint_path is not None:
                _append_checkpoint(checkpoint_path, checkpoint_row)
                logger.info(
                    "[%s] checkpointed query %d/%d (%s, %.1f s)",
                    method,
                    index,
                    len(queries),
                    item.query_id,
                    elapsed_ms / 1000,
                )
        else:
            if checkpoint_row.get("query") != item.query:
                raise ValueError(f"Checkpoint query text changed for {item.query_id!r}")
            score = QueryMetrics(
                recall_at_5=float(checkpoint_row["recall_at_5"]),
                reciprocal_rank_at_10=float(checkpoint_row["reciprocal_rank_at_10"]),
                ndcg_at_10=float(checkpoint_row["ndcg_at_10"]),
            )
            elapsed_ms = float(checkpoint_row["latency_ms"])

        metrics.append(score)
        latencies.append(elapsed_ms)
        per_query.append(
            {
                "query_id": checkpoint_row["query_id"],
                "query": checkpoint_row["query"],
                "relevant_youtube_video_id": checkpoint_row[
                    "relevant_youtube_video_id"
                ],
                "tiktok_url": checkpoint_row["tiktok_url"],
                "recall_at_5": score.recall_at_5,
                "reciprocal_rank_at_10": score.reciprocal_rank_at_10,
                "ndcg_at_10": score.ndcg_at_10,
                "latency_ms": elapsed_ms,
                "result_segment_ids": checkpoint_row["result_segment_ids"],
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
        description="Evaluate episode retrieval on organic TikTok queries"
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--segments", type=Path, default=Path("data/segments.jsonl"))
    parser.add_argument("--bm25", type=Path, default=Path("artifacts/bm25_index.json"))
    parser.add_argument(
        "--vectors", type=Path, help="Vector artifact stem or metadata JSON"
    )
    parser.add_argument("--methods", default="bm25", help="Comma-separated methods")
    parser.add_argument("--candidate-k", type=int, default=50)
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
    parser.add_argument("--reranker-batch-size", type=int, default=50)
    parser.add_argument(
        "--reranker-revision",
        help="Optional Hugging Face commit, tag, or branch to pin",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation_results.json")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="append one durable per-query result after each completed query; "
        "requires exactly one evaluation method and resumes matching runs",
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
    if args.checkpoint is not None and len(methods) != 1:
        raise ValueError("--checkpoint requires exactly one evaluation method")
    if args.reranker_batch_size < 1:
        raise ValueError("--reranker-batch-size must be positive")
    queries = load_evaluation_queries(args.queries)

    device = resolve_torch_device(args.device)
    reranker = None
    weights = HybridWeights()
    if {"cross_encoder", "hybrid_rerank"} & set(methods):
        reranker = CrossEncoderReranker(
            args.reranker_model,
            device=device,
            local_files_only=not args.allow_model_download,
            batch_size=args.reranker_batch_size,
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
                group_by_episode=True,
                deduplicate_overlaps=True,
                rerank_k=EVALUATION_RERANK_K,
            )
    summaries = [
        evaluate_method(
            engine,
            queries,
            method,  # type: ignore[arg-type]
            candidate_k=args.candidate_k,
            bootstrap_samples=args.bootstrap_samples,
            top_k=EVALUATION_TOP_K,
            rerank_k=EVALUATION_RERANK_K,
            weights=weights,
            checkpoint_path=args.checkpoint,
        )
        for method in methods
    ]
    _log_table(summaries)
    write_json(
        args.output,
        {
            "query_file": str(args.queries),
            "ranking_unit": "episode",
            "candidate_k": args.candidate_k,
            "warmup": args.warmup,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "manifest": build_evaluation_manifest(
                engine=engine,
                methods=methods,
                queries=queries,
                query_file=args.queries,
                segments_path=args.segments,
                bm25_path=args.bm25,
                vector_path=args.vectors,
                candidate_k=args.candidate_k,
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

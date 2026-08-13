from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as error:  # pragma: no cover - exercised only without api extra
    raise RuntimeError(
        "The HTTP API needs FastAPI. Install the 'api' optional dependencies."
    ) from error

from pydantic import BaseModel, Field, field_validator

from drzavni_posao.search.common.device import resolve_torch_device
from drzavni_posao.search.common.io import load_segments
from drzavni_posao.search.retrieval.engine import SearchEngine
from drzavni_posao.search.retrieval.rerank import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)

SEGMENTS_PATH = Path(os.environ.get("DP_SEGMENTS_PATH", "data/segments.jsonl"))
BM25_PATH = Path(os.environ.get("DP_BM25_PATH", "artifacts/bm25_index.json"))
VECTOR_PATH = Path(os.environ.get("DP_VECTOR_PATH", "artifacts/vector_index"))
RERANKER_MODEL = os.environ.get("DP_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
RERANKER_REVISION = os.environ.get("DP_RERANKER_REVISION")
TORCH_DEVICE = resolve_torch_device(os.environ.get("DP_DEVICE", "auto"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DP_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

ExposedMethod = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
EXPOSED_METHODS: tuple[ExposedMethod, ...] = (
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
)


def _vector_metadata_exists(path: Path) -> bool:
    return path.with_suffix(".json").exists()


def _available_methods() -> list[str]:
    if _vector_metadata_exists(VECTOR_PATH):
        return list(EXPOSED_METHODS)
    return ["bm25"]


@functools.lru_cache
def corpus_stats() -> dict[str, int]:
    segments = load_segments(SEGMENTS_PATH)
    return {
        "episodes": len({segment.youtube_video_id for segment in segments}),
        "segments": len(segments),
    }


@functools.lru_cache
def load_engine(use_vectors: bool, use_reranker: bool = False) -> SearchEngine:
    reranker = (
        CrossEncoderReranker(
            RERANKER_MODEL,
            device=TORCH_DEVICE,
            local_files_only=True,
            model_revision=RERANKER_REVISION,
        )
        if use_reranker
        else None
    )
    return SearchEngine.from_paths(
        segments_path=SEGMENTS_PATH,
        bm25_path=BM25_PATH,
        vector_path=VECTOR_PATH if use_vectors else None,
        dense_device=TORCH_DEVICE,
        local_models_only=True,
        reranker=reranker,
    )


class SearchRequest(BaseModel):
    query: str
    method: ExposedMethod = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    candidate_k: int = Field(50, ge=10, le=100)
    group_by_episode: bool = True

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


app = FastAPI(title="Državni posao — search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        stats = corpus_stats()
    except (FileNotFoundError, ValueError, OSError):
        stats = {"episodes": 0, "segments": 0}
    return {"status": "ok", "methods": _available_methods(), **stats}


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    use_vectors = request.method != "bm25"
    if use_vectors and not _vector_metadata_exists(VECTOR_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Method '{request.method}' requires the vector index, which is "
                "not built. Run dp-build-vectors or use method 'bm25'."
            ),
        )
    started = time.perf_counter()
    try:
        engine = load_engine(use_vectors, request.method == "hybrid_rerank")
        results = engine.search(
            request.query,
            method=request.method,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            group_by_episode=request.group_by_episode,
        )
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "query": request.query,
        "method": request.method,
        "took_seconds": time.perf_counter() - started,
        "results": [result.to_dict() for result in results],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "drzavni_posao.search.api.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("DP_API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()

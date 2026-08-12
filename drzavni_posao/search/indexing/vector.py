from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.device import resolve_torch_device
from ..common.fingerprints import (
    segment_corpus_fingerprint,
    semantic_corpus_fingerprint,
)
from ..common.io import load_segments, write_json
from ..common.models import SegmentRecord
from ..text import normalize_text


logger = logging.getLogger(__name__)

VECTOR_INDEX_VERSION = 1
# Version 2 predates the version reset and uses the same binary schema.
SUPPORTED_VECTOR_INDEX_VERSIONS = {VECTOR_INDEX_VERSION, 2}
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
Float32Array = NDArray[np.float32]


class TextEncoder(Protocol):
    name: str
    revision: str | None
    requested_revision: str | None
    dimension: int
    uses_e5_prefixes: bool

    def encode_passages(self, texts: Sequence[str]) -> Float32Array: ...

    def encode_query(self, text: str) -> Float32Array: ...


def model_revision_from_path(path: Path) -> str | None:
    """Return the commit hash from a Hugging Face snapshot path, when present."""
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1]
    return None


def resolve_model_path(
    model_name: str,
    *,
    local_files_only: bool,
    revision: str | None = None,
) -> Path:
    """Resolve a Hub ID to a snapshot directory before model construction.

    Some transformer submodules do not consistently propagate
    ``local_files_only``. Passing an actual snapshot path prevents accidental
    network probes after the model has been downloaded once.
    """
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Resolving a model ID needs huggingface-hub (installed with the "
            "'search' optional dependencies)."
        ) from error
    try:
        return Path(
            snapshot_download(
                repo_id=model_name,
                local_files_only=local_files_only,
                revision=revision,
            )
        )
    except Exception as error:
        mode = "local cache" if local_files_only else "Hugging Face Hub"
        raise RuntimeError(
            f"Could not resolve model {model_name!r} from the {mode}. "
            "Build once with --allow-download or pass a local model path."
        ) from error


def normalize_vector(vector: ArrayLike) -> Float32Array:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Encoder produced an empty or non-finite vector")
    length = np.linalg.norm(values)
    if length == 0.0:
        raise ValueError("Encoder produced a zero vector")
    return values / length


class SentenceTransformerTextEncoder:
    """Lazy adapter around sentence-transformers with E5 prefix handling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        batch_size: int = 32,
        uses_e5_prefixes: bool | None = None,
        model_revision: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval needs sentence-transformers. Install the 'search' "
                "optional dependencies first."
            ) from error
        self.name = model_name
        self.requested_revision = model_revision
        self.batch_size = batch_size
        self.device = resolve_torch_device(device)
        self.uses_e5_prefixes = (
            "e5" in model_name.casefold()
            if uses_e5_prefixes is None
            else uses_e5_prefixes
        )
        resolved_model_path = resolve_model_path(
            model_name,
            local_files_only=local_files_only,
            revision=model_revision,
        )
        self.revision = model_revision_from_path(resolved_model_path)
        try:
            self.model = SentenceTransformer(
                str(resolved_model_path),
                device=self.device,
                local_files_only=True,
            )
        except TypeError as error:
            raise RuntimeError(
                "The installed sentence-transformers version does not support "
                "offline model loading; upgrade it or pass a local model path."
            ) from error
        dimension_method = getattr(self.model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_method())

    def _encode(self, texts: Sequence[str], prefix: str) -> Float32Array:
        prepared = [
            f"{prefix}{text}" if self.uses_e5_prefixes else text for text in texts
        ]
        values = np.asarray(
            self.model.encode(
                prepared,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(prepared) > self.batch_size,
            ),
            dtype=np.float32,
        )
        if values.ndim != 2 or values.shape[0] != len(prepared):
            raise ValueError("Encoder returned an invalid vector matrix")
        return np.stack([normalize_vector(row) for row in values])

    def encode_passages(self, texts: Sequence[str]) -> Float32Array:
        return self._encode(texts, "passage: ")

    def encode_query(self, text: str) -> Float32Array:
        return self._encode([text], "query: ")[0]


class HashingTextEncoder:
    """Deterministic model-free encoder for tests and offline smoke checks.

    This is feature hashing, not a semantic model.  Artifacts built with it are
    labelled accordingly and must not be reported as dense semantic results.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("Hashing encoder dimension must be at least 8")
        self.name = f"hashing-smoke-{dimension}"
        self.revision = None
        self.requested_revision = None
        self.dimension = dimension
        self.uses_e5_prefixes = False

    def _encode_one(self, text: str) -> Float32Array:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = normalize_text(text).split()
        if not tokens:
            tokens = ["__empty__"]
        features = [*tokens, *(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[index] += sign
        return normalize_vector(vector)

    def encode_passages(self, texts: Sequence[str]) -> Float32Array:
        return np.stack([self._encode_one(text) for text in texts])

    def encode_query(self, text: str) -> Float32Array:
        return self._encode_one(text)


@dataclass(slots=True)
class DenseHit:
    segment_id: str
    score: float


class VectorIndex:
    def __init__(
        self,
        *,
        segment_ids: list[str],
        vectors: Float32Array,
        dimension: int,
        model_name: str,
        uses_e5_prefixes: bool,
        corpus_fingerprint: str,
        semantic_fingerprint: str,
        model_revision: str | None = None,
        model_requested_revision: str | None = None,
    ) -> None:
        values = np.asarray(vectors, dtype=np.float32)
        if dimension <= 0 or values.shape != (len(segment_ids), dimension):
            raise ValueError("Vector matrix shape does not match segment IDs")
        self.segment_ids = segment_ids
        self.vectors = values
        self.dimension = dimension
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_requested_revision = model_requested_revision
        self.uses_e5_prefixes = uses_e5_prefixes
        self.corpus_fingerprint = corpus_fingerprint
        self.semantic_corpus_fingerprint = semantic_fingerprint

    @classmethod
    def build(
        cls, segments: list[SegmentRecord], encoder: TextEncoder
    ) -> "VectorIndex":
        if not segments:
            raise ValueError("Cannot build a vector index over an empty corpus")
        rows = np.asarray(
            encoder.encode_passages([segment.text_semantic for segment in segments]),
            dtype=np.float32,
        )
        if rows.ndim != 2 or rows.shape[0] != len(segments):
            raise ValueError("Encoder returned the wrong number of vectors")
        if rows.shape[1] != encoder.dimension:
            raise ValueError("Encoder returned inconsistent dimensions")
        vectors = np.stack([normalize_vector(row) for row in rows])
        return cls(
            segment_ids=[segment.segment_id for segment in segments],
            vectors=vectors,
            dimension=encoder.dimension,
            model_name=encoder.name,
            model_revision=getattr(encoder, "revision", None),
            model_requested_revision=getattr(encoder, "requested_revision", None),
            uses_e5_prefixes=encoder.uses_e5_prefixes,
            corpus_fingerprint=segment_corpus_fingerprint(segments),
            semantic_fingerprint=semantic_corpus_fingerprint(segments),
        )

    def search_vector(
        self, query_vector: ArrayLike, *, top_k: int = 50
    ) -> list[DenseHit]:
        if top_k < 1:
            return []
        query = normalize_vector(query_vector)
        if len(query) != self.dimension:
            raise ValueError(
                f"Query vector has dimension {len(query)}, expected {self.dimension}"
            )
        scores = self.vectors @ query
        hits = [
            DenseHit(segment_id=segment_id, score=float(score))
            for segment_id, score in zip(self.segment_ids, scores)
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.segment_id))
        return hits[:top_k]

    @staticmethod
    def artifact_paths(path: Path) -> tuple[Path, Path]:
        if path.suffix == ".json":
            metadata_path = path
            vectors_path = path.with_suffix(".f32")
        else:
            metadata_path = path.with_suffix(".json")
            vectors_path = path.with_suffix(".f32")
        return metadata_path, vectors_path

    def save(self, path: Path) -> None:
        metadata_path, vectors_path = self.artifact_paths(path)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = vectors_path.with_name(f".{vectors_path.name}.tmp")
        with temporary.open("wb") as handle:
            self.vectors.tofile(handle)
        temporary.replace(vectors_path)
        write_json(
            metadata_path,
            {
                "version": VECTOR_INDEX_VERSION,
                "algorithm": "exact_cosine_float32",
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "model_requested_revision": self.model_requested_revision,
                "uses_e5_prefixes": self.uses_e5_prefixes,
                "dimension": self.dimension,
                "row_count": len(self.segment_ids),
                "segment_ids": self.segment_ids,
                "corpus_fingerprint": self.corpus_fingerprint,
                "semantic_corpus_fingerprint": self.semantic_corpus_fingerprint,
                "byte_order": sys.byteorder,
                "vectors_file": vectors_path.name,
            },
        )

    @classmethod
    def load(
        cls, path: Path, *, segments: list[SegmentRecord] | None = None
    ) -> "VectorIndex":
        metadata_path, default_vectors_path = cls.artifact_paths(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") not in SUPPORTED_VECTOR_INDEX_VERSIONS:
            raise ValueError(
                f"Unsupported vector artifact version: {metadata.get('version')}"
            )
        vectors_path = metadata_path.parent / metadata.get(
            "vectors_file", default_vectors_path.name
        )
        row_count = int(metadata["row_count"])
        dimension = int(metadata["dimension"])
        expected_values = row_count * dimension
        with vectors_path.open("rb") as handle:
            values = np.fromfile(handle, dtype=np.float32, count=expected_values)
        if values.size != expected_values:
            raise ValueError("Vector artifact is truncated")
        if metadata.get("byte_order", sys.byteorder) != sys.byteorder:
            values.byteswap(inplace=True)
        values = values.reshape(row_count, dimension)
        model_revision = metadata.get("model_revision")
        model_requested_revision = metadata.get("model_requested_revision")
        index = cls(
            segment_ids=[str(item) for item in metadata["segment_ids"]],
            vectors=values,
            dimension=dimension,
            model_name=str(metadata["model_name"]),
            model_revision=(
                str(model_revision) if model_revision is not None else None
            ),
            model_requested_revision=(
                str(model_requested_revision)
                if model_requested_revision is not None
                else None
            ),
            uses_e5_prefixes=bool(metadata.get("uses_e5_prefixes", False)),
            corpus_fingerprint=str(metadata["corpus_fingerprint"]),
            semantic_fingerprint=str(metadata["semantic_corpus_fingerprint"]),
        )
        if segments is not None:
            if segment_corpus_fingerprint(segments) != index.corpus_fingerprint:
                raise ValueError(
                    "Vector index does not match the segment corpus; rebuild it"
                )
            if index.segment_ids != [segment.segment_id for segment in segments]:
                raise ValueError("Vector index segment order does not match the corpus")
            if (
                semantic_corpus_fingerprint(segments)
                != index.semantic_corpus_fingerprint
            ):
                raise ValueError(
                    "Vector index does not match the semantic segment corpus; "
                    "rebuild it"
                )
        return index


def encoder_for_index(
    index: VectorIndex,
    *,
    device: str = "auto",
    local_files_only: bool = True,
) -> TextEncoder:
    if index.model_name.startswith("hashing-smoke-"):
        return HashingTextEncoder(index.dimension)
    return SentenceTransformerTextEncoder(
        index.model_name,
        device=device,
        local_files_only=local_files_only,
        uses_e5_prefixes=index.uses_e5_prefixes,
        model_revision=index.model_revision,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build an exact cosine index for multilingual text embeddings"
    )
    parser.add_argument("--segments", type=Path, default=Path("data/segments.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vector_index"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        help="Optional Hugging Face commit, tag, or branch to pin",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto (cuda→mps→cpu), or an explicit device string",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow sentence-transformers to download the requested model",
    )
    parser.add_argument(
        "--hashing-smoke",
        action="store_true",
        help="Build a non-semantic model-free index for smoke testing only",
    )
    parser.add_argument("--hashing-dimension", type=int, default=256)
    args = parser.parse_args()

    segments = load_segments(args.segments)
    if args.hashing_smoke:
        encoder: TextEncoder = HashingTextEncoder(args.hashing_dimension)
    else:
        encoder = SentenceTransformerTextEncoder(
            args.model,
            device=args.device,
            local_files_only=not args.allow_download,
            batch_size=args.batch_size,
            model_revision=args.model_revision,
        )
    index = VectorIndex.build(segments, encoder)
    index.save(args.output)
    metadata_path, _ = index.artifact_paths(args.output)
    logger.info(
        "Wrote %d x %d vector index (%s) to %s",
        len(index.segment_ids),
        index.dimension,
        index.model_name,
        metadata_path,
    )


if __name__ == "__main__":
    main()

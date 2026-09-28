#!/usr/bin/env python3
"""Sweep direct spherical K-means over dialogue-only episode embeddings."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drzavni_posao.search.common.fingerprints import segment_corpus_fingerprint
from drzavni_posao.search.common.io import load_segments, write_json
from drzavni_posao.search.indexing.vector import DEFAULT_MODEL
from drzavni_posao.search.topics import (
    build_dialogue_chunk_corpus,
    load_stop_words,
    pca_transform_episode_embeddings,
)
from drzavni_posao.search.topics.lemmatization import (
    load_or_build_lemma_texts,
)
from drzavni_posao.search.topics.embeddings import load_or_build_embeddings
from drzavni_posao.search.topics.engine import (
    RESULT_VERSION,
    build_clustering_result,
    fit_topic_labels,
)
from drzavni_posao.search.topics.metrics import (
    SELECTION_SCORE_WEIGHTS,
    add_selection_scores,
    cluster_diagnostics,
    stability,
    topic_diversity,
)

logger = logging.getLogger(__name__)


def _parse_int_list(value: str, *, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must contain integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{option} values must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{option} cannot contain duplicates")
    return values


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/segments_luna_corrected_57de2165.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dialogue_topic_sweep_luna_corrected_57de2165"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--utterances-per-chunk", type=int, default=16)
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--topics", default="25,40,50,75,100")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--representative-episodes", type=int, default=3)
    parser.add_argument("--stop-words", type=Path)
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--lemma-cache", type=Path)
    parser.add_argument("--lemma-model-dir", type=Path)
    parser.add_argument("--lemma-device", default="cpu")
    parser.add_argument("--allow-lemma-download", action="store_true")
    args = parser.parse_args()

    topic_counts = _parse_int_list(args.topics, option="--topics")
    seeds = _parse_int_list(args.seeds, option="--seeds")
    if len(seeds) < 2:
        parser.error("--seeds needs at least two values for stability")
    if args.pca_components is not None and args.pca_components < 2:
        parser.error("--pca-components must be at least 2")
    if args.pca_whiten and args.pca_components is None:
        parser.error("--pca-whiten requires --pca-components")
    if args.lemma_cache is not None and not args.lemmatize:
        parser.error("--lemma-cache requires --lemmatize")
    if args.allow_lemma_download and not args.lemmatize:
        parser.error("--allow-lemma-download requires --lemmatize")
    for name in (
        "batch_size",
        "utterances_per_chunk",
        "n_init",
        "min_df",
        "top_n_words",
        "representative_episodes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    segments = load_segments(args.segments)
    source_fingerprint = segment_corpus_fingerprint(segments)
    corpus = build_dialogue_chunk_corpus(
        segments,
        utterances_per_chunk=args.utterances_per_chunk,
    )
    if max(topic_counts) >= len(corpus.documents):
        parser.error("Every topic count must be below the episode count")
    source_embeddings, embedding_metadata = load_or_build_embeddings(
        corpus=corpus,
        source_fingerprint=source_fingerprint,
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        allow_download=args.allow_download,
        utterances_per_chunk=args.utterances_per_chunk,
    )
    try:
        embeddings, representation_metadata = pca_transform_episode_embeddings(
            source_embeddings,
            n_components=args.pca_components,
            whiten=args.pca_whiten,
            random_state=seeds[0],
        )
    except ValueError as error:
        parser.error(str(error))
    stop_words = load_stop_words(args.stop_words)
    topic_texts: Sequence[str] = tuple(document.text for document in corpus.documents)
    label_preprocessing: dict[str, Any] = {"algorithm": "normalized_surface_forms"}
    if args.lemmatize:
        lemma_cache = (
            args.lemma_cache
            if args.lemma_cache is not None
            else args.output_dir / "lemmas.json"
        )
        topic_texts, label_preprocessing = load_or_build_lemma_texts(
            episode_ids=[document.youtube_video_id for document in corpus.documents],
            raw_texts=corpus.raw_episode_texts,
            source_corpus_fingerprint=source_fingerprint,
            cache_path=lemma_cache,
            allow_download=args.allow_lemma_download,
            device=args.lemma_device,
            model_dir=args.lemma_model_dir,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for n_topics in topic_counts:
        logger.info("Fitting direct spherical K-means with K=%d", n_topics)
        model, labels = fit_topic_labels(
            documents=corpus.documents,
            topic_texts=topic_texts,
            embeddings=embeddings,
            n_topics=n_topics,
            seed=seeds[0],
            n_init=args.n_init,
            min_df=args.min_df,
            top_n_words=args.top_n_words,
            stop_words=stop_words,
        )
        metrics, assigned_cosines, margins = cluster_diagnostics(embeddings, labels)
        metrics["topic_diversity"] = round(
            topic_diversity(model, labels, args.top_n_words), 6
        )
        metrics["stability"] = stability(
            embeddings=embeddings,
            canonical_labels=labels,
            n_topics=n_topics,
            seeds=seeds,
            n_init=args.n_init,
        )
        settings = {
            "document_unit": "whole_episode",
            "embedding_source": "dialogue_only_non_overlapping_chunks",
            "dimensionality_reduction": representation_metadata,
            "clusterer": "spherical_kmeans",
            "n_topics": n_topics,
            "n_init": args.n_init,
            "seeds": seeds,
            "min_df": args.min_df,
            "ngram_range": [1, 2],
            "reduce_frequent_words": True,
            "top_n_words": args.top_n_words,
            "representative_episode_count": args.representative_episodes,
            "stop_word_count": len(stop_words),
            "label_preprocessing": label_preprocessing,
        }
        result = build_clustering_result(
            corpus=corpus,
            embeddings=embeddings,
            labels=labels,
            assigned_cosines=assigned_cosines,
            margins=margins,
            topic_model=model,
            embedding_metadata=embedding_metadata,
            representation_metadata=representation_metadata,
            source_fingerprint=source_fingerprint,
            settings=settings,
            representative_episode_count=args.representative_episodes,
        )
        result["label_preprocessing"] = label_preprocessing
        result["evaluation"] = metrics
        output_path = args.output_dir / f"k_{n_topics:03d}.json"
        write_json(output_path, result)
        metrics["n_topics"] = n_topics
        metrics["result_path"] = str(output_path)
        results.append(metrics)
        logger.info(
            "K=%d silhouette=%.4f stability=%.4f margin=%.4f diversity=%.3f",
            n_topics,
            metrics["cosine_silhouette"],
            metrics["stability"]["mean_adjusted_rand_index"],
            metrics["mean_assignment_margin"],
            metrics["topic_diversity"],
        )

    add_selection_scores(results)
    recommended = max(results, key=lambda row: row["selection_score"])
    summary = {
        "version": RESULT_VERSION,
        "algorithm": "dialogue_episode_spherical_kmeans_sweep",
        "segments": str(args.segments),
        "source_corpus_fingerprint": source_fingerprint,
        "episode_count": len(corpus.documents),
        "dialogue_chunk_count": len(corpus.chunk_texts),
        "embedding": {
            key: value
            for key, value in embedding_metadata.items()
            if key not in ("episode_ids", "chunk_count")
        },
        "clustering_representation": representation_metadata,
        "label_preprocessing": label_preprocessing,
        "selection_score_weights": SELECTION_SCORE_WEIGHTS,
        "results": results,
        "recommended_n_topics": recommended["n_topics"],
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    logger.info(
        "Recommended K=%d by composite score %.3f; wrote %s",
        recommended["n_topics"],
        recommended["selection_score"],
        summary_path,
    )


if __name__ == "__main__":
    main()

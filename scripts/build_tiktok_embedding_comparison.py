#!/usr/bin/env python3
"""Build the data displayed by the TikTok embedding-comparison Canvas."""

import json
import random
import statistics
from pathlib import Path

BASE = Path("artifacts/model_comparison")


def latest(pattern, exclude=()):
    candidates = [
        path
        for path in BASE.glob(pattern)
        if not any(word in path.stem for word in exclude)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime)


PATHS = None
OUTPUT = BASE / "tiktok_embedding_comparison_data.json"
MODELS = ("e5", "embedic", "srbedding")
LABELS = {
    "e5": "E5 base",
    "embedic": "Embedić large",
    "srbedding": "SRBedding base",
}


def rows(result, group=None):
    selected = result["per_query"]
    if group:
        organic = group == "organic"
        selected = [row for row in selected if bool(row.get("tiktok_url")) == organic]
    return selected


def summary(result, group=None):
    selected = rows(result, group)
    data = {
        "hit_at_5": sum(row["recall_at_5"] > 0 for row in selected),
        "hit_at_10": sum(row["reciprocal_rank_at_10"] > 0 for row in selected),
        "recall_at_5": statistics.fmean(row["recall_at_5"] for row in selected),
        "mrr_at_10": statistics.fmean(row["reciprocal_rank_at_10"] for row in selected),
        "ndcg_at_10": statistics.fmean(row["ndcg_at_10"] for row in selected),
    }
    if group is None:
        latencies = sorted(row["latency_ms"] for row in selected)
        data["mean_latency_ms"] = statistics.fmean(latencies)
        data["p95_latency_ms"] = latencies[(95 * len(latencies) - 1) // 100]
    return data


def by_query(result):
    return {row["query_id"]: row for row in result["per_query"]}


def paired(left, right):
    left = by_query(left)
    right = by_query(right)
    output = {}
    for name, field in (
        ("recall_at_5", "recall_at_5"),
        ("mrr_at_10", "reciprocal_rank_at_10"),
        ("ndcg_at_10", "ndcg_at_10"),
    ):
        differences = [left[key][field] - right[key][field] for key in left]
        generator = random.Random(91)
        means = sorted(
            statistics.fmean(generator.choice(differences) for _ in differences)
            for _ in range(10_000)
        )
        output[name] = {
            "mean": statistics.fmean(differences),
            "confidence_interval_95": [means[249], means[9749]],
        }
    return output


def rank(row):
    reciprocal_rank = row["reciprocal_rank_at_10"]
    return round(1 / reciprocal_rank) if reciprocal_rank else None


def query_data(runs):
    hybrid = {model: by_query(runs[model]["hybrid"]) for model in MODELS}
    unique = []
    shared_misses = 0
    for query_id, source in hybrid["e5"].items():
        ranks = {model: rank(hybrid[model][query_id]) for model in MODELS}
        if (ranks["embedic"] or 11) <= 5 and all(
            (ranks[model] or 11) > 5 for model in ("e5", "srbedding")
        ):
            unique.append(
                {
                    "query_id": query_id,
                    "query": source["query"],
                    "ranks": ranks,
                }
            )
        shared_misses += all((value or 11) > 5 for value in ranks.values())

    movement = {}
    for model in MODELS:
        before = by_query(runs[model]["hybrid"])
        after = by_query(runs[model]["hybrid_rerank"])
        before_after = [
            (rank(before[query_id]) or 11, rank(after[query_id]) or 11)
            for query_id in before
        ]
        movement[model] = {
            "promoted": sum(new < old for old, new in before_after),
            "harmed": sum(new > old for old, new in before_after),
        }
    return {
        "embedic_only_top_5": unique,
        "shared_top_5_miss_count": shared_misses,
        "reranker_movement": movement,
    }


def build_comparison():
    paths = PATHS or {
        "e5": latest("evaluation_e5*.json"),
        "embedic": latest("evaluation_embedic*.json", exclude=("bge",)),
        "srbedding": latest("evaluation_srbedding*.json"),
        "bge": latest("evaluation_embedic_bge*.json"),
    }
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    runs = {
        name: {result["method"]: result for result in document["results"]}
        for name, document in documents.items()
    }

    reference = documents["e5"]["manifest"]["data"]
    expected = (
        reference["query_set_fingerprint"],
        reference["corpus_fingerprint"],
    )
    if any(
        (
            document["manifest"]["data"]["query_set_fingerprint"],
            document["manifest"]["data"]["corpus_fingerprint"],
        )
        != expected
        for document in documents.values()
    ):
        raise ValueError("Evaluation files use different queries or corpora")

    models = {}
    for model in MODELS:
        model_runs = runs[model]
        models[model] = {
            "label": LABELS[model],
            "embedding": documents[model]["manifest"]["models"]["vector"],
            "dense": {"all": summary(model_runs["dense"])},
            "hybrid": {
                "all": summary(model_runs["hybrid"]),
                "organic": summary(model_runs["hybrid"], "organic"),
                "manual": summary(model_runs["hybrid"], "manual"),
            },
            "hybrid_rerank": {
                "all": summary(model_runs["hybrid_rerank"]),
                "organic": summary(model_runs["hybrid_rerank"], "organic"),
                "manual": summary(model_runs["hybrid_rerank"], "manual"),
            },
        }

    embedding_differences = {}
    for other in ("e5", "srbedding"):
        embedding_differences[f"embedic_vs_{other}"] = {
            method: paired(runs["embedic"][method], runs[other][method])
            for method in ("dense", "hybrid", "hybrid_rerank")
        }

    minilm = runs["embedic"]["hybrid_rerank"]
    bge = runs["bge"]["hybrid_rerank"]
    return {
        "bm25": {
            "all": summary(runs["e5"]["bm25"]),
            "organic": summary(runs["e5"]["bm25"], "organic"),
            "manual": summary(runs["e5"]["bm25"], "manual"),
        },
        "models": models,
        "comparisons": {
            "embedding_differences": embedding_differences,
            "reranker_gain_over_hybrid": {
                model: paired(runs[model]["hybrid_rerank"], runs[model]["hybrid"])
                for model in MODELS
            },
            "bge_minus_minilm": {
                "bge_reranker": documents["bge"]["manifest"]["models"]["reranker"],
                "minilm_reranker": documents["embedic"]["manifest"]["models"][
                    "reranker"
                ],
                "bge": summary(bge),
                "minilm": summary(minilm),
                "difference": paired(bge, minilm),
            },
        },
        "queries": query_data(runs),
    }


def main():
    comparison = build_comparison()
    OUTPUT.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote comparison data to {OUTPUT}")


if __name__ == "__main__":
    main()

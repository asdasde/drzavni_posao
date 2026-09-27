#!/usr/bin/env python3
"""Rank sweep configs by a selection score normalized across all runs, not just Ks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drzavni_posao.search.common.io import write_json
from drzavni_posao.search.topics.metrics import pool_sweep_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        type=Path,
        nargs="+",
        help="summary.json files, or directories containing one",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    loaded = []
    for path in args.summaries:
        summary_path = path / "summary.json" if path.is_dir() else path
        loaded.append(
            (str(summary_path), json.loads(summary_path.read_text(encoding="utf-8")))
        )

    ranked = pool_sweep_results(loaded)
    if args.output is not None:
        write_json(args.output, {"ranked": ranked})
    for row in ranked[: args.top]:
        print(
            f"score={row['selection_score']:.3f}  "
            f"u={row['utterances_per_chunk']} "
            f"{row['reduction']}({row['reduction_dimension']}) K={row['n_topics']}  "
            f"sil={row['cosine_silhouette']:.4f} "
            f"stab={row['stability']['mean_adjusted_rand_index']:.3f} "
            f"div={row['topic_diversity']:.3f}  {row['summary']}"
        )


if __name__ == "__main__":
    main()

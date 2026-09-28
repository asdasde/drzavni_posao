#!/usr/bin/env python3
"""Compare two episode-topic results after optimal cluster alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from drzavni_posao.search.common.io import write_json


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("algorithm") != "bertopic_episode_clustering":
        raise ValueError(f"{path} is not a BERTopic episode result")
    return value


def compare_topic_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_episodes = {row["youtube_video_id"]: row for row in baseline["episodes"]}
    candidate_episodes = {row["youtube_video_id"]: row for row in candidate["episodes"]}
    if baseline_episodes.keys() != candidate_episodes.keys():
        missing = sorted(baseline_episodes.keys() - candidate_episodes.keys())
        added = sorted(candidate_episodes.keys() - baseline_episodes.keys())
        raise ValueError(
            f"Episode sets differ: {len(missing)} missing, {len(added)} added"
        )

    video_ids = sorted(baseline_episodes)
    baseline_assignments = np.asarray(
        [int(baseline_episodes[video_id]["topic_id"]) for video_id in video_ids]
    )
    candidate_assignments = np.asarray(
        [int(candidate_episodes[video_id]["topic_id"]) for video_id in video_ids]
    )
    baseline_ids = sorted(set(baseline_assignments.tolist()))
    candidate_ids = sorted(set(candidate_assignments.tolist()))
    baseline_index = {topic_id: index for index, topic_id in enumerate(baseline_ids)}
    candidate_index = {topic_id: index for index, topic_id in enumerate(candidate_ids)}
    overlap = np.zeros((len(baseline_ids), len(candidate_ids)), dtype=np.int64)
    for baseline_id, candidate_id in zip(baseline_assignments, candidate_assignments):
        overlap[
            baseline_index[int(baseline_id)], candidate_index[int(candidate_id)]
        ] += 1

    baseline_rows, candidate_columns = linear_sum_assignment(-overlap)
    candidate_to_baseline = {
        candidate_ids[candidate_column]: baseline_ids[baseline_row]
        for baseline_row, candidate_column in zip(baseline_rows, candidate_columns)
    }
    aligned_candidate = np.asarray(
        [
            candidate_to_baseline.get(int(topic_id), -10_000)
            for topic_id in candidate_assignments
        ]
    )
    unchanged = aligned_candidate == baseline_assignments

    baseline_topics = {int(row["topic_id"]): row for row in baseline["topics"]}
    candidate_topics = {int(row["topic_id"]): row for row in candidate["topics"]}
    matched_topics = []
    for candidate_id, baseline_id in sorted(
        candidate_to_baseline.items(), key=lambda item: item[1]
    ):
        intersection = int(
            overlap[baseline_index[baseline_id], candidate_index[candidate_id]]
        )
        baseline_size = int(baseline_topics[baseline_id]["episode_count"])
        candidate_size = int(candidate_topics[candidate_id]["episode_count"])
        union = baseline_size + candidate_size - intersection
        matched_topics.append(
            {
                "baseline_topic_id": baseline_id,
                "baseline_label": baseline_topics[baseline_id]["label"],
                "candidate_topic_id": candidate_id,
                "candidate_label": candidate_topics[candidate_id]["label"],
                "overlap_episode_count": intersection,
                "baseline_episode_count": baseline_size,
                "candidate_episode_count": candidate_size,
                "jaccard": round(intersection / union, 6) if union else 1.0,
            }
        )

    changed_episodes = []
    for row, video_id in enumerate(video_ids):
        if unchanged[row]:
            continue
        baseline_episode = baseline_episodes[video_id]
        candidate_episode = candidate_episodes[video_id]
        changed_episodes.append(
            {
                "youtube_video_id": video_id,
                "episode_number": baseline_episode["episode_number"],
                "episode_title": baseline_episode["episode_title"],
                "baseline_topic_id": int(baseline_assignments[row]),
                "baseline_topic_label": baseline_episode["topic_label"],
                "candidate_topic_id": int(candidate_assignments[row]),
                "candidate_topic_label": candidate_episode["topic_label"],
                "candidate_aligned_to_baseline_topic_id": int(aligned_candidate[row]),
            }
        )

    return {
        "episode_count": len(video_ids),
        "baseline_topic_count": len(baseline_ids),
        "candidate_topic_count": len(candidate_ids),
        "adjusted_rand_index": round(
            float(adjusted_rand_score(baseline_assignments, candidate_assignments)),
            6,
        ),
        "normalized_mutual_information": round(
            float(
                normalized_mutual_info_score(
                    baseline_assignments,
                    candidate_assignments,
                )
            ),
            6,
        ),
        "aligned_assignment_agreement": round(float(unchanged.mean()), 6),
        "changed_episode_count": len(changed_episodes),
        "matched_topics": matched_topics,
        "changed_episodes": changed_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_topic_results(_load(args.baseline), _load(args.candidate))
    result["baseline_result_path"] = str(args.baseline)
    result["candidate_result_path"] = str(args.candidate)
    write_json(args.output, result)
    print(
        f"ARI={result['adjusted_rand_index']:.3f}, "
        f"NMI={result['normalized_mutual_information']:.3f}, "
        f"aligned agreement={result['aligned_assignment_agreement']:.1%}; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()

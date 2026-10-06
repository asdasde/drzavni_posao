#!/usr/bin/env python3
"""Create a separate MLflow-style transcript export with correction patches applied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drzavni_posao.search.common.io import write_json
from drzavni_posao.transcription.corrections import (
    CorrectionStats,
    apply_episode_corrections,
)
from drzavni_posao.transcription.transcript import Transcript


def _load_corrections(path: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for artifact in sorted(path.glob("*.json")):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        episode_key = payload.get("episode_key")
        edits = payload.get("edits")
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError(f"{artifact} has no valid episode_key")
        if not isinstance(edits, list):
            raise ValueError(f"{artifact} has no edits list")
        if episode_key in payloads:
            raise ValueError(f"Duplicate correction payload for {episode_key!r}")
        payloads[episode_key] = payload
    if not payloads:
        raise ValueError(f"No correction JSON files found in {path}")
    return payloads


def build_corrected_export(
    *,
    source_export: Path,
    corrections_dir: Path,
    output_dir: Path,
    correction_run_id: str,
    confidences: frozenset[str],
) -> dict[str, Any]:
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed export {output_dir}")

    source_episodes = source_export / "episodes"
    if not source_episodes.is_dir():
        raise ValueError(f"{source_export} has no episodes directory")
    payloads = _load_corrections(corrections_dir)
    source_parent_run_id = source_export.name

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = source_export / "config.json"
    if config_path.is_file():
        write_json(
            output_dir / "config.json",
            json.loads(config_path.read_text(encoding="utf-8")),
        )

    totals = CorrectionStats()
    corrected_episodes = 0
    unchanged_episodes = 0
    episode_stats: dict[str, dict[str, int]] = {}
    source_keys: set[str] = set()

    for episode_dir in sorted(source_episodes.iterdir()):
        if not episode_dir.is_dir():
            continue
        episode_key = episode_dir.name
        source_path = episode_dir / "character_assigner" / "transcript.json"
        if not source_path.is_file():
            continue
        source_keys.add(episode_key)

        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_dict(source_payload)
        correction_payload = payloads.get(episode_key)
        edits = correction_payload["edits"] if correction_payload else []
        if correction_payload:
            payload_parent = correction_payload.get("source_parent_run_id")
            if payload_parent and payload_parent != source_parent_run_id:
                raise ValueError(
                    f"Correction {episode_key!r} targets parent run "
                    f"{payload_parent!r}, not {source_parent_run_id!r}"
                )

        corrected, stats = apply_episode_corrections(
            transcript,
            edits,
            confidences=confidences,
        )
        totals.add(stats)
        if stats.applied:
            corrected_episodes += 1
        else:
            unchanged_episodes += 1
        episode_stats[episode_key] = {
            "applied": stats.applied,
            "skipped": stats.skipped,
            "ignored_confidence": stats.ignored_confidence,
            "changed_segments": stats.changed_segments,
        }

        output_payload = {
            **source_payload,
            "text": corrected.render(),
            "segments": corrected.segments,
            "correction_provenance": {
                "correction_run_id": correction_run_id,
                "confidences": sorted(confidences),
                **episode_stats[episode_key],
            },
        }
        write_json(
            output_dir
            / "episodes"
            / episode_key
            / "character_assigner"
            / "transcript.json",
            output_payload,
        )

    missing_sources = sorted(set(payloads) - source_keys)
    if missing_sources:
        raise ValueError(
            f"{len(missing_sources)} correction payloads have no source transcript: "
            f"{missing_sources[:10]}"
        )

    manifest = {
        "format": "corrected_transcript_export",
        "source_export": str(source_export),
        "source_parent_run_id": source_parent_run_id,
        "correction_run_id": correction_run_id,
        "correction_artifact_count": len(payloads),
        "confidences": sorted(confidences),
        "episode_count": corrected_episodes + unchanged_episodes,
        "corrected_episode_count": corrected_episodes,
        "unchanged_episode_count": unchanged_episodes,
        "edits_applied": totals.applied,
        "edits_skipped": totals.skipped,
        "edits_ignored_confidence": totals.ignored_confidence,
        "changed_source_segments": totals.changed_segments,
        "episode_stats": episode_stats,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correction-run-id", required=True)
    parser.add_argument(
        "--confidence",
        action="append",
        dest="confidences",
        default=[],
        help="Correction confidence to apply; repeat as needed (default: high)",
    )
    args = parser.parse_args()
    confidences = frozenset(args.confidences or ["high"])

    manifest = build_corrected_export(
        source_export=args.source_export,
        corrections_dir=args.corrections,
        output_dir=args.output,
        correction_run_id=args.correction_run_id,
        confidences=confidences,
    )
    print(
        f"Wrote {manifest['episode_count']} corrected transcripts to {args.output}; "
        f"applied {manifest['edits_applied']} edits across "
        f"{manifest['corrected_episode_count']} episodes"
    )


if __name__ == "__main__":
    main()

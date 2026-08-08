#!/usr/bin/env python3
"""Export one configuration MLflow run for offline labeling.

Pulls ``config.json`` from the parent run and every completed stage
(``transcription/``, ``diarization/``, ``character_assigner/``) for each
episode into::

    data/mlflow_exports/<run_id>/
      config.json
      episodes/<episode_key>/transcription/...
      episodes/<episode_key>/character_assigner/...
      episodes/<episode_key>/episode_meta.json

On Databricks, run with the workspace tracking URI already configured.
Locally, point MLflow at Databricks first, e.g.::

    export MLFLOW_TRACKING_URI=databricks
    export DATABRICKS_CONFIG_PROFILE=staging
    uv run python scripts/export_labeling_run.py \\
      --run-id b66150772bc043d091f7b4c9ff606ce0

Pass ``--episode`` one or more times only when a partial export is desired.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlflow.tracking import MlflowClient

from drzavni_posao.tracking import MlflowTracker, Stage


def export_run(
    run_id: str,
    episode_keys: list[str] | None,
    output_root: Path,
) -> Path:
    client = MlflowClient()
    parent = client.get_run(run_id)
    if parent.data.tags.get("run_level") != "configuration":
        raise SystemExit(f"Run {run_id} is not a configuration parent run")

    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=True)

    downloaded_config = Path(
        client.download_artifacts(run_id, "config.json", str(destination))
    )
    config_path = destination / "config.json"
    if downloaded_config.resolve() != config_path.resolve():
        shutil.copy2(downloaded_config, config_path)

    tracker = MlflowTracker(parent.info.experiment_id, client=client)
    states = tracker.episode_run_states(run_id)

    requested = episode_keys or sorted(states)
    missing = [key for key in requested if key not in states]
    if missing:
        raise SystemExit(f"Episodes missing from run {run_id}: {missing}")

    exported = []
    skipped = []
    for key in requested:
        state = states[key]
        if Stage.TRANSCRIPTION not in state.completed_stages:
            skipped.append(key)
            continue
        episode_dir = destination / "episodes" / key
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)
        stages = [stage for stage in Stage if stage in state.completed_stages]
        for stage in stages:
            client.download_artifacts(
                state.run_id,
                stage.value,
                str(episode_dir),
            )
        meta = {
            "episode_key": key,
            "episode_run_id": state.run_id,
            "parent_run_id": run_id,
            "completed_stages": [stage.value for stage in stages],
        }
        (episode_dir / "episode_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )
        stage_names = ", ".join(stage.value for stage in stages)
        print(f"exported episode {key} from {state.run_id}: {stage_names}")
        exported.append(key)

    manifest = {
        "parent_run_id": run_id,
        "exported_episodes": exported,
        "skipped_without_transcription": skipped,
        "stages_by_episode": {
            key: [stage.value for stage in Stage if stage in states[key].completed_stages]
            for key in exported
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {destination}: {len(exported)} episode(s) exported, "
        f"{len(skipped)} skipped without transcription"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Configuration parent run id")
    parser.add_argument(
        "--episode",
        action="append",
        dest="episodes",
        help="Export only this episode key; repeat for multiple (default: all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/mlflow_exports"),
        help="Directory that will contain <run_id>/ (default: data/mlflow_exports)",
    )
    args = parser.parse_args()
    export_run(args.run_id, args.episodes, args.output_root)


if __name__ == "__main__":
    main()

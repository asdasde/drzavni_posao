"""Temporarily migrate MLflow runs to the explicit pipeline stage names.

The script is a dry run unless ``--apply`` is passed. MLflow does not expose
safe deletion for metrics and artifacts, so those are copied to their new
names while the legacy entries remain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from drzavni_posao.tracking import configuration_id


STAGE_NAMES = {
    "asr": "transcription",
    "diar": "diarization",
    "assign": "character_assigner",
}
CONFIG_KEYS = {
    "asr": "transcription",
    "character_assignment": "character_assigner",
}


def renamed_stage(name: str) -> str:
    return STAGE_NAMES.get(name, name)


def renamed_metric(name: str) -> str:
    for old, new in STAGE_NAMES.items():
        if name.startswith(f"{old}."):
            return f"{new}.{name.removeprefix(f'{old}.')}"
        if name.startswith(f"mean.{old}."):
            return f"mean.{new}.{name.removeprefix(f'mean.{old}.')}"
    return name


def renamed_tag_key(name: str) -> str:
    for old, new in STAGE_NAMES.items():
        if name == f"stage.{old}":
            return f"stage.{new}"
        if name == f"source.{old}.run_id":
            return f"source.{new}.run_id"
    return name


def migrated_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(configuration)
    for old, new in CONFIG_KEYS.items():
        if old not in migrated:
            continue
        if new in migrated and migrated[new] != migrated[old]:
            raise ValueError(
                f"Configuration contains conflicting {old!r} and {new!r}"
            )
        migrated[new] = migrated.pop(old)

    character_assigner = migrated.get("character_assigner")
    if isinstance(character_assigner, dict):
        character_assigner = dict(character_assigner)
        character_assigner.pop("anonymous_label", None)
        migrated["character_assigner"] = character_assigner

    return migrated


def all_runs(client: MlflowClient, experiment_id: str) -> list[Run]:
    runs: list[Run] = []
    page_token = None
    while True:
        page = client.search_runs(
            [experiment_id],
            max_results=1000,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if page_token is None:
            return runs


def load_configuration(client: MlflowClient, run_id: str) -> dict[str, Any]:
    with TemporaryDirectory() as directory:
        path = client.download_artifacts(run_id, "config.json", directory)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def planned_tags(run: Run) -> tuple[dict[str, str], set[str]]:
    current = run.data.tags
    updates: dict[str, str] = {}
    deletes: set[str] = set()

    for old_key, value in current.items():
        new_key = renamed_tag_key(old_key)
        if old_key == "stages":
            value = ",".join(
                renamed_stage(stage)
                for stage in value.split(",")
                if stage
            )
        elif old_key == "copy_until":
            value = renamed_stage(value)

        if new_key != old_key:
            existing = current.get(new_key)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Run {run.info.run_id} has conflicting tags "
                    f"{old_key!r} and {new_key!r}"
                )
            deletes.add(old_key)
        if new_key != old_key or value != current[old_key]:
            updates[new_key] = value

    return updates, deletes


def planned_metrics(run: Run) -> dict[str, float]:
    updates = {}
    for old_name, value in run.data.metrics.items():
        new_name = renamed_metric(old_name)
        if new_name == old_name:
            continue
        existing = run.data.metrics.get(new_name)
        if existing is not None and existing != value:
            raise ValueError(
                f"Run {run.info.run_id} has conflicting metrics "
                f"{old_name!r} and {new_name!r}"
            )
        if existing is None:
            updates[new_name] = value
    return updates


def planned_artifacts(
    client: MlflowClient,
    run_id: str,
) -> list[tuple[str, str]]:
    roots = {artifact.path for artifact in client.list_artifacts(run_id)}
    return [
        (old, new)
        for old, new in STAGE_NAMES.items()
        if old in roots and new not in roots
    ]


def migrate(experiment_name: str, *, apply: bool) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Unknown experiment {experiment_name!r}")

    runs = all_runs(client, experiment.experiment_id)
    plans: dict[str, dict[str, Any]] = {}
    config_owners: dict[str, str] = {}

    for i, run in enumerate(runs):
        run_id = run.info.run_id
        print(f"Migrate run {i}/{len(runs)}", run_id)
        plan: dict[str, Any] = {}

        if run.data.tags.get("run_level") == "configuration":
            current = load_configuration(client, run_id)
            configuration = migrated_configuration(current)
            config_id = configuration_id(configuration)
            owner = config_owners.setdefault(config_id, run_id)
            if owner != run_id:
                raise RuntimeError(
                    "Migration would create duplicate configuration runs "
                    f"{owner} and {run_id} with config_id {config_id}"
                )
            plan["configuration"] = configuration
            plan["configuration_changed"] = configuration != current
            if run.data.tags.get("config_id") != config_id:
                plan["config_id"] = config_id

        plan["tags"], plan["delete_tags"] = planned_tags(run)
        plan["metrics"] = planned_metrics(run)
        plan["artifacts"] = planned_artifacts(client, run_id)
        plans[run_id] = plan

    changed_runs = sum(
        bool(
            plan.get("configuration_changed")
            or plan.get("config_id")
            or plan["tags"]
            or plan["delete_tags"]
            or plan["metrics"]
            or plan["artifacts"]
        )
        for plan in plans.values()
    )
    mode = "Applying" if apply else "Dry run"
    print(f"{mode}: {changed_runs} of {len(runs)} runs need updates")

    if not apply:
        print("Re-run with --apply to perform the migration.")
        return

    for run_id, plan in plans.items():
        if plan.get("configuration_changed"):
            client.log_dict(
                run_id,
                plan["configuration"],
                "config.json",
            )
        if config_id := plan.get("config_id"):
            client.set_tag(run_id, "config_id", config_id)
        for name, value in plan["tags"].items():
            client.set_tag(run_id, name, value)
        for name in plan["delete_tags"]:
            client.delete_tag(run_id, name)
        for name, value in plan["metrics"].items():
            client.log_metric(run_id, name, value)
        for old, new in plan["artifacts"]:
            with TemporaryDirectory() as directory:
                source = client.download_artifacts(run_id, old, directory)
                client.log_artifacts(run_id, source, artifact_path=new)

    print(f"Migrated {changed_runs} runs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        help="Override the active MLflow tracking URI",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; without this flag only report the plan",
    )
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    migrate(args.experiment, apply=args.apply)


if __name__ == "__main__":
    main()

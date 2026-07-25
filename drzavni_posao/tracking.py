from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from contextlib import contextmanager
from statistics import fmean
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _flatten_params(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened = {}

    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_params(item, name))
    elif isinstance(value, list):
        flattened[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        flattened[prefix] = "null" if value is None else value

    return flattened


def _configuration_params(configuration: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in _flatten_params(configuration).items()
    }


_RESUMABLE_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED", "RUNNING"})


def _find_latest_matching_parent_run(
    configuration: Mapping[str, Any],
    experiment_name: str,
):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    expected_params = _configuration_params(configuration)
    runs = MlflowClient().search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_level = 'configuration'",
        max_results=1000,
        order_by=["attributes.start_time DESC"],
    )
    return next(
        (
            run
            for run in runs
            if run.info.status in _RESUMABLE_STATUSES
            and run.data.params == expected_params
        ),
        None,
    )


def _finished_child_runs(parent_run_id: str):
    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    return client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run_id}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1000,
        order_by=["attributes.start_time DESC"],
    )


def finished_episode_splits(parent_run_id: str) -> set[tuple[str | None, str]]:
    return {
        (
            run.data.tags.get("episode_key"),
            run.data.tags.get("split", "tune"),
        )
        for run in _finished_child_runs(parent_run_id)
    }


def finished_child_metric_sets(parent_run_id: str) -> list[dict[str, float]]:
    latest_metrics = {}
    for run in _finished_child_runs(parent_run_id):
        key = (
            run.data.tags.get("episode_key"),
            run.data.tags.get("split", "tune"),
        )
        latest_metrics.setdefault(key, dict(run.data.metrics))
    return list(latest_metrics.values())


@contextmanager
def pipeline_run(
    name: str,
    configuration: Mapping[str, Any],
    *,
    experiment_name: str,
    tags: Mapping[str, Any] | None = None,
    resume_matching_run: bool = False,
):
    """Start a configuration run, or resume the latest exact resumable match."""
    if mlflow.active_run() is not None:
        raise RuntimeError("An MLflow run is already active; call mlflow.end_run()")

    mlflow.set_experiment(experiment_name)
    matching_run = (
        _find_latest_matching_parent_run(configuration, experiment_name)
        if resume_matching_run
        else None
    )

    if matching_run is not None:
        logger.info("Resuming matching parent run: %s", matching_run.info.run_id)
        with mlflow.start_run(run_id=matching_run.info.run_id) as run:
            yield run.info.run_id
        return

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "run_level": "configuration",
                **(tags or {}),
            }
        )
        mlflow.log_params(_flatten_params(configuration))
        mlflow.log_dict(configuration, "config.json")
        yield run.info.run_id


@contextmanager
def episode_run(episode_key: str, split: str = "tune"):
    """Start one episode child run under the active configuration run."""
    with mlflow.start_run(
        run_name=f"ep-{episode_key}",
        nested=True,
        tags={
            "run_level": "episode",
            "episode_key": str(episode_key),
            "split": split,
        },
    ) as run:
        yield run.info.run_id


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    }


def log_stage(episode, stage: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    """Log metrics and transcript artifacts for one episode stage."""
    prefixed = {
        f"{stage}.{key}": value
        for key, value in _finite_metrics(metrics).items()
    }
    mlflow.log_metrics(prefixed)

    mlflow.log_dict(
        episode.transcript.to_dict(),
        f"{stage}/transcript.json",
    )
    mlflow.log_text(
        episode.transcript.render(),
        f"{stage}/transcript.txt",
    )

    if stage == "diar" and episode.diarization is not None:
        mlflow.log_dict(
            episode.diarization,
            "diar/turns.json",
        )

    return prefixed


def log_reference(reference_text: str) -> None:
    mlflow.log_text(reference_text, "reference/reference.txt")


def log_parent_averages(
    episode_metric_sets: list[Mapping[str, Any]],
) -> None:
    """Log averages after every episode child run has closed."""
    metric_names = {
        name
        for metrics in episode_metric_sets
        for name in metrics
    }

    averages = {}
    for name in metric_names:
        values = [
            metrics[name]
            for metrics in episode_metric_sets
            if name in metrics and math.isfinite(float(metrics[name]))
        ]
        if values:
            averages[f"mean.{name}"] = fmean(values)

    mlflow.log_metrics(averages)
    mlflow.log_dict(
        episode_metric_sets,
        "metrics-by-episode.json",
    )

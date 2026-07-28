from __future__ import annotations

import json
import hashlib
import logging
import math
from collections.abc import Collection, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from .transcript import Transcript

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    def __new__(cls, value: str, config_key: str):
        member = str.__new__(cls, value)
        member._value_ = value
        member._config_key = config_key
        return member

    ASR = ("asr", "asr")
    DIAR = ("diar", "diarization")
    ASSIGN = ("assign", "character_assignment")

    @property
    def next(self) -> Stage | None:
        stages = tuple(type(self))
        index = stages.index(self) + 1
        return stages[index] if index < len(stages) else None

    @property
    def config_keys(self) -> tuple[str, ...]:
        stages = tuple(type(self))
        prefix = stages[: stages.index(self) + 1]
        return ("audio", *(stage._config_key for stage in prefix))

    @property
    def required_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                f"{self}/transcript.json",
                f"{self}/transcript.txt",
            }
        )


@dataclass(frozen=True)
class EpisodeRunState:
    run_id: str
    completed_stages: frozenset[Stage]


class MlflowTracker:
    def __init__(
        self,
        experiment_name: str,
        client: MlflowClient | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.client = client or MlflowClient()

    @classmethod
    def _validate_stages(
        cls,
        stages: Iterable[Stage],
    ) -> tuple[Stage, ...]:
        stages = tuple(stages)
        if not all(isinstance(stage, Stage) for stage in stages):
            raise TypeError(f"Stages must be Stage members: {stages}")
        if len(set(stages)) != len(stages):
            raise ValueError(f"Duplicate stages are not allowed: {stages}")
        pipeline = tuple(Stage)
        positions = [pipeline.index(stage) for stage in stages]
        if positions != sorted(positions):
            raise ValueError(f"Stages must follow pipeline order: {pipeline}")
        return stages

    def _configuration(self, run_id: str) -> dict[str, Any]:
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(
                run_id,
                "config.json",
                directory,
            )
            return json.loads(Path(path).read_text(encoding="utf-8"))

    def _validate_configuration(
        self,
        run_id: str,
        configuration: Mapping[str, Any],
    ) -> None:
        if self._configuration(run_id) != configuration:
            raise ValueError(
                f"configuration does not match MLflow run {run_id}"
            )

    def _matching_parent(self, configuration: Mapping[str, Any]):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        expected = _configuration_params(configuration)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.run_level = 'configuration'",
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )
        return next(
            (run for run in runs if run.data.params == expected),
            None,
        )

    def _stage_tag(self, run_id: str, requested: tuple[Stage, ...]) -> str:
        current = self.client.get_run(run_id).data.tags.get("stages", "")
        included = {
            Stage(stage) for stage in current.split(",") if stage
        } | set(requested)
        return ",".join(stage for stage in Stage if stage in included)

    @contextmanager
    def pipeline_run(
        self,
        name: str,
        configuration: Mapping[str, Any],
        *,
        stages: Iterable[Stage],
        tags: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        resume_matching_run: bool = False,
    ):
        """Start or resume one exact configuration-level run."""
        if mlflow.active_run() is not None:
            raise RuntimeError(
                "An MLflow run is already active; call mlflow.end_run()"
            )

        requested = self._validate_stages(stages)
        mlflow.set_experiment(self.experiment_name)
        selected = None
        if run_id is not None:
            selected = self.client.get_run(run_id)
            if selected.data.tags.get("run_level") != "configuration":
                raise ValueError(f"Run {run_id} is not a configuration run")
            self._validate_configuration(run_id, configuration)
        elif resume_matching_run:
            selected = self._matching_parent(configuration)

        if selected is not None:
            run_id = selected.info.run_id
            logger.info("Resuming parent run: %s", run_id)
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tags(
                    {
                        **(tags or {}),
                        "stages": self._stage_tag(run_id, requested),
                    }
                )
                yield run_id
            return

        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "run_level": "configuration",
                    "stages": ",".join(requested),
                    **(tags or {}),
                }
            )
            mlflow.log_params(_flatten_params(configuration))
            mlflow.log_dict(dict(configuration), "config.json")
            yield run.info.run_id

    def _child_runs(self, parent_run_id: str):
        parent = self.client.get_run(parent_run_id)
        return self.client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=(
                f"tags.mlflow.parentRunId = '{parent_run_id}'"
            ),
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )

    def episode_states(
        self,
        parent_run_id: str,
    ) -> dict[str, EpisodeRunState]:
        states = {}
        for run in self._child_runs(parent_run_id):
            episode_key = run.data.tags.get("episode_key")
            if episode_key is None or episode_key in states:
                continue
            completed = {
                Stage(key.removeprefix("stage."))
                for key, value in run.data.tags.items()
                if key.startswith("stage.") and value == "complete"
            }
            states[episode_key] = EpisodeRunState(
                run_id=run.info.run_id,
                completed_stages=frozenset(completed),
            )
        return states

    @contextmanager
    def episode_run(
        self,
        episode_key: str,
        run_id: str | None = None,
    ):
        tags = {
            "run_level": "episode",
            "episode_key": str(episode_key),
        }
        if run_id is not None:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.set_tags(tags)
                yield
            return

        with mlflow.start_run(
            run_name=f"ep-{episode_key}",
            nested=True,
            tags=tags,
        ):
            yield

    @classmethod
    def pending_stages(
        cls,
        *,
        completed: Collection[Stage],
        requested: Iterable[Stage],
    ) -> tuple[Stage, ...]:
        requested = cls._validate_stages(requested)
        available = set(completed)
        pipeline = tuple(Stage)
        pending = []
        for stage in requested:
            prerequisites = set(pipeline[: pipeline.index(stage)])
            missing = prerequisites - available
            if missing:
                raise ValueError(
                    f"Stage {stage!r} requires completed stages: "
                    f"{', '.join(sorted(missing))}"
                )
            if stage not in available:
                pending.append(stage)
                available.add(stage)
        return tuple(pending)

    def log_stage(
        self,
        episode,
        stage: Stage,
        metrics: Mapping[str, Any],
    ) -> None:
        stage = self._validate_stages((stage,))[0]
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
        mlflow.set_tag(f"stage.{stage}", "complete")

    def restore_stage(
        self,
        episode,
        run_id: str,
        stage: Stage,
    ) -> None:
        stage = self._validate_stages((stage,))[0]
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(
                run_id,
                f"{stage}/transcript.json",
                directory,
            )
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            episode.transcript = Transcript.from_dict(data)

    def log_parent_averages(self, parent_run_id: str) -> None:
        metric_sets = [
            dict(self.client.get_run(state.run_id).data.metrics)
            for state in self.episode_states(parent_run_id).values()
        ]
        metric_names = {
            name for metrics in metric_sets for name in metrics
        }
        averages = {}
        for name in metric_names:
            values = [
                metrics[name]
                for metrics in metric_sets
                if name in metrics
                and math.isfinite(float(metrics[name]))
            ]
            if values:
                averages[f"mean.{name}"] = fmean(values)
        mlflow.log_metrics(averages)
        mlflow.log_dict(metric_sets, "metrics-by-episode.json")

    def episode_metrics(
        self,
        run_name: str,
        metric: str,
    ) -> dict[str, float]:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return {}
        parents = self.client.search_runs(
            [experiment.experiment_id],
            (
                "tags.run_level = 'configuration' "
                f"AND tags.mlflow.runName = '{run_name}'"
            ),
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if not parents:
            return {}
        return {
            episode_key: run.data.metrics[metric]
            for episode_key, state in self.episode_states(
                parents[0].info.run_id
            ).items()
            if metric
            in (run := self.client.get_run(state.run_id)).data.metrics
        }

    def _fork_key(
        self,
        source_run_id: str,
        name: str,
        configuration: Mapping[str, Any],
        copy_until: Stage,
    ) -> str:
        identity = json.dumps(
            {
                "source_run_id": source_run_id,
                "name": name,
                "configuration": configuration,
                "copy_until": copy_until,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    def _existing_fork(self, fork_key: str):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        runs = self.client.search_runs(
            [experiment.experiment_id],
            f"tags.fork.key = '{fork_key}'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        return runs[0] if runs else None

    def _validate_fork_configuration(
        self,
        source_run_id: str,
        configuration: Mapping[str, Any],
        copy_until: Stage,
    ) -> tuple[Stage, ...]:
        pipeline = tuple(Stage)
        prefix = pipeline[: pipeline.index(copy_until) + 1]
        source = self._configuration(source_run_id)
        for key in copy_until.config_keys:
            if source.get(key) != configuration.get(key):
                raise ValueError(
                    f"{key} configuration must match to copy "
                    f"through {copy_until}"
                )
        return prefix

    def _copy_stage(
        self,
        source_episode_run_id: str,
        stage: Stage,
    ) -> None:
        artifacts = {
            artifact.path
            for artifact in self.client.list_artifacts(
                source_episode_run_id,
                stage,
            )
        }
        missing = stage.required_artifacts - artifacts
        if missing:
            raise ValueError(
                f"Completed {stage} stage is missing artifacts: "
                f"{sorted(missing)}"
            )

        source = self.client.get_run(source_episode_run_id)
        metrics = {
            key: value
            for key, value in source.data.metrics.items()
            if key.startswith(f"{stage}.")
        }
        with TemporaryDirectory() as directory:
            stage_directory = self.client.download_artifacts(
                source_episode_run_id,
                stage,
                directory,
            )
            mlflow.log_artifacts(stage_directory, artifact_path=stage)
        mlflow.log_metrics(metrics)
        mlflow.set_tag(
            f"source.{stage}.run_id",
            source_episode_run_id,
        )
        mlflow.set_tag(f"stage.{stage}", "complete")

    def fork_run(
        self,
        source_run_id: str,
        name: str,
        configuration: Mapping[str, Any],
        *,
        copy_until: Stage,
    ) -> str:
        source = self.client.get_run(source_run_id)
        if source.data.tags.get("run_level") != "configuration":
            raise ValueError(
                f"Run {source_run_id} is not a configuration run"
            )
        prefix = self._validate_fork_configuration(
            source_run_id,
            configuration,
            copy_until,
        )
        fork_key = self._fork_key(
            source_run_id,
            name,
            configuration,
            copy_until,
        )
        existing = self._existing_fork(fork_key)
        target_run_id = existing.info.run_id if existing else None
        source_states = self.episode_states(source_run_id)
        fork_tags = {
            "fork.key": fork_key,
            "source.run_id": source_run_id,
            "copy_until": copy_until,
        }

        with self.pipeline_run(
            name,
            configuration,
            stages=prefix,
            tags=fork_tags,
            run_id=target_run_id,
        ) as target_run_id:
            target_states = self.episode_states(target_run_id)
            for episode_key, source_state in source_states.items():
                target_state = target_states.get(episode_key)
                target_completed = set(
                    target_state.completed_stages if target_state else ()
                )
                blocked = False
                with self.episode_run(
                    episode_key,
                    target_state.run_id if target_state else None,
                ):
                    mlflow.set_tag(
                        "source.episode_run_id",
                        source_state.run_id,
                    )
                    for stage in prefix:
                        if blocked or stage not in source_state.completed_stages:
                            blocked = True
                            continue
                        if stage not in target_completed:
                            self._copy_stage(source_state.run_id, stage)
                            target_completed.add(stage)

        return target_run_id


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


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    }

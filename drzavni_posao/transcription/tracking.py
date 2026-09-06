from __future__ import annotations

import json
import hashlib
import logging
import math
from collections.abc import Collection, Mapping
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


class DuplicateConfigurationError(RuntimeError):
    pass


class DuplicateEpisodeRunError(RuntimeError):
    pass


class Stage(StrEnum):
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    CHARACTER_ASSIGNER = "character_assigner"

    @property
    def next(self) -> Stage | None:
        stages = tuple(type(self))
        index = stages.index(self) + 1
        return stages[index] if index < len(stages) else None

    @property
    def prerequisites(self) -> frozenset[Stage]:
        if self is Stage.TRANSCRIPTION:
            return frozenset()
        return frozenset({Stage.TRANSCRIPTION})

    @property
    def config_keys(self) -> tuple[str, ...]:
        stages = self.prerequisites | {self}
        ordered = tuple(stage for stage in type(self) if stage in stages)
        return ("audio", *(stage.value for stage in ordered))

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

    def _configuration(self, run_id: str) -> dict[str, Any]:
        with TemporaryDirectory() as directory:
            path = Path(
                self.client.download_artifacts(
                    run_id,
                    "config.json",
                    directory,
                )
            )
            return json.loads(path.read_text(encoding="utf-8"))

    def _get_or_create_run(self, name: str, config: Any) -> str:
        configuration = config.to_dict()
        config_id = configuration_id(config.to_dict(exclude_defaults=True))
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Unknown experiment {self.experiment_name}")

        runs = self.client.search_runs(
            [experiment.experiment_id],
            (f"tags.run_level = 'configuration' AND tags.config_id = '{config_id}'"),
            max_results=2,
            order_by=["attributes.start_time DESC"],
        )
        if len(runs) > 1:
            run_ids = [run.info.run_id for run in runs]
            raise DuplicateConfigurationError(
                f"Multiple parent runs have config_id {config_id}: {run_ids}"
            )
        if runs:
            return runs[0].info.run_id

        run = self.client.create_run(
            experiment.experiment_id,
            run_name=name,
            tags={
                "run_level": "configuration",
                "config_id": config_id,
            },
        )
        for key, value in _flatten_params(configuration).items():
            self.client.log_param(run.info.run_id, key, value)
        self.client.log_dict(
            run.info.run_id,
            configuration,
            "config.json",
        )
        return run.info.run_id

    def _stage_tag(self, run_id: str, requested: set[Stage]) -> str:
        current = self.client.get_run(run_id).data.tags.get("stages", "")
        included = {Stage(stage) for stage in current.split(",") if stage} | set(
            requested
        )
        return ",".join(stage for stage in Stage if stage in included)

    @contextmanager
    def pipeline_run(
        self,
        name: str,
        config: Any,
        *,
        stages: set[Stage],
        tags: Mapping[str, Any] | None = None,
    ):
        """Start or resume a configuration-level run for ``config``.

        ``config`` is a ``PipelineConfig`` (or anything with ``to_dict``).
        The full dump is logged; ``config_id`` uses ``exclude_defaults=True``.
        """
        if mlflow.active_run() is not None:
            raise RuntimeError("An MLflow run is already active; call mlflow.end_run()")

        mlflow.set_experiment(self.experiment_name)
        run_id = self._get_or_create_run(name, config)

        logger.info("Using parent run: %s", run_id)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags(
                {
                    **(tags or {}),
                    "stages": self._stage_tag(run_id, stages),
                }
            )
            yield run_id

    def _child_runs(self, parent_run_id: str):
        parent = self.client.get_run(parent_run_id)
        runs = []
        page_token = None
        while True:
            page = self.client.search_runs(
                experiment_ids=[parent.info.experiment_id],
                filter_string=(
                    f"tags.`mlflow.parentRunId` = '{parent_run_id}'"
                ),
                max_results=100,
                order_by=["attributes.start_time DESC"],
                page_token=page_token,
            )
            runs.extend(
                run
                for run in page
                if run.data.tags.get("run_level") == "episode"
            )
            page_token = page.token
            if not page_token:
                return runs

    def _episode_run_id(
        self,
        parent_run_id: str,
        episode_key: str,
    ) -> str | None:
        parent = self.client.get_run(parent_run_id)
        runs = self.client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=(
                f"tags.mlflow.parentRunId = '{parent_run_id}' "
                "AND tags.run_level = 'episode' "
                f"AND tags.episode_key = '{episode_key}'"
            ),
            max_results=2,
        )
        if len(runs) > 1:
            run_ids = [run.info.run_id for run in runs]
            raise DuplicateEpisodeRunError(
                f"Multiple child runs exist for episode {episode_key!r}: {run_ids}"
            )
        return runs[0].info.run_id if runs else None

    def episode_run_states(
        self,
        parent_run_id: str,
    ) -> dict[str, EpisodeRunState]:
        states = {}
        for run in self._child_runs(parent_run_id):
            episode_key = run.data.tags.get("episode_key")
            if episode_key is None:
                continue
            if episode_key in states:
                run_ids = [states[episode_key].run_id, run.info.run_id]
                raise DuplicateEpisodeRunError(
                    f"Multiple child runs exist for episode {episode_key!r}: {run_ids}"
                )
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
    def episode_run(self, episode_key: str):
        parent = mlflow.active_run()
        if parent is None:
            raise RuntimeError("Episode runs require an active parent run")

        run_id = self._episode_run_id(parent.info.run_id, episode_key)
        if run_id is not None:
            with mlflow.start_run(run_id=run_id, nested=True):
                yield
            return

        tags = {
            "run_level": "episode",
            "episode_key": str(episode_key),
        }
        with mlflow.start_run(
            run_name=f"ep-{episode_key}",
            nested=True,
            tags=tags,
        ):
            yield

    @staticmethod
    def pending_stages(
        *,
        completed: Collection[Stage],
        requested: set[Stage],
    ) -> tuple[Stage, ...]:
        available = set(completed)
        pending = []
        for stage in Stage:
            if stage not in requested:
                continue
            missing = stage.prerequisites - available
            if missing:
                raise ValueError(
                    f"Stage {stage!r} requires completed stages: {', '.join(missing)}"
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
        prefixed = {
            f"{stage}.{key}": value for key, value in _finite_metrics(metrics).items()
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

    def stage_transcript(self, run_id: str, stage: Stage) -> Transcript:
        with TemporaryDirectory() as directory:
            path = Path(
                self.client.download_artifacts(
                    run_id,
                    f"{stage}/transcript.json",
                    directory,
                )
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            return Transcript.from_dict(data)

    def restore_stage(
        self,
        episode,
        run_id: str,
        stage: Stage,
    ) -> None:
        episode.transcript = self.stage_transcript(run_id, stage)

    def log_parent_averages(self, parent_run_id: str) -> None:
        metric_sets = [
            dict(self.client.get_run(state.run_id).data.metrics)
            for state in self.episode_run_states(parent_run_id).values()
        ]
        metric_names = {name for metrics in metric_sets for name in metrics}
        averages = {}
        for name in metric_names:
            values = [
                metrics[name]
                for metrics in metric_sets
                if name in metrics and math.isfinite(float(metrics[name]))
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
            for episode_key, state in self.episode_run_states(
                parents[0].info.run_id
            ).items()
            if metric in (run := self.client.get_run(state.run_id)).data.metrics
        }

    def _validate_fork_configuration(
        self,
        source: Any,
        target: Any,
        copy_until: Stage,
    ) -> tuple[Stage, ...]:
        pipeline = tuple(Stage)
        prefix = pipeline[: pipeline.index(copy_until) + 1]
        source_dump = source.to_dict(exclude_defaults=True)
        target_dump = target.to_dict(exclude_defaults=True)
        for key in copy_until.config_keys:
            if source_dump.get(key) != target_dump.get(key):
                raise ValueError(
                    f"{key} configuration must match to copy through {copy_until}"
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
                f"Completed {stage} stage is missing artifacts: {sorted(missing)}"
            )

        source = self.client.get_run(source_episode_run_id)
        metrics = {
            key: value
            for key, value in source.data.metrics.items()
            if key.startswith(f"{stage}.")
        }
        with TemporaryDirectory() as directory:
            stage_directory = Path(
                self.client.download_artifacts(
                    source_episode_run_id,
                    stage,
                    directory,
                )
            )
            mlflow.log_artifacts(str(stage_directory), artifact_path=stage)
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
        config: Any,
        *,
        copy_until: Stage,
    ) -> str:
        source = self.client.get_run(source_run_id)
        if source.data.tags.get("run_level") != "configuration":
            raise ValueError(f"Run {source_run_id} is not a configuration run")
        source_config = type(config).from_dict(self._configuration(source_run_id))
        prefix = self._validate_fork_configuration(
            source_config,
            config,
            copy_until,
        )
        source_states = self.episode_run_states(source_run_id)
        fork_tags = {
            "source.run_id": source_run_id,
            "copy_until": copy_until,
        }

        with self.pipeline_run(
            name,
            config,
            stages=set(prefix),
            tags=fork_tags,
        ) as target_run_id:
            target_states = self.episode_run_states(target_run_id)
            for episode_key, source_state in source_states.items():
                target_state = target_states.get(episode_key)
                target_completed = set(
                    target_state.completed_stages if target_state else ()
                )
                blocked = False
                with self.episode_run(episode_key):
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


def configuration_id(configuration: Mapping[str, Any]) -> str:
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    }

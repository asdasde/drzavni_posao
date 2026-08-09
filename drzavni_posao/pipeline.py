from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .character_assigner import CharacterAssigner, CharacterAssignerConfig
from .diarizer import Diarizer, DiarizerConfig
from .metrics import (
    alignment_metrics,
    assignment_metrics,
    diarization_metrics,
    repetition_metrics,
)
from .store import MediaStore
from .tracking import MlflowTracker, Stage, configuration_id
from .transcriber import Transcriber, TranscriberConfig

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    trim_start: float
    trim_end: float
    transcription: TranscriberConfig
    diarization: DiarizerConfig | None = None
    character_assigner: CharacterAssignerConfig | None = None

    @classmethod
    def from_media(
        cls,
        media: MediaStore,
        *,
        transcription: TranscriberConfig,
        diarization: DiarizerConfig | None = None,
        character_assigner: CharacterAssignerConfig | None = None,
    ) -> PipelineConfig:
        return cls(
            trim_start=media.trim_start,
            trim_end=media.trim_end,
            transcription=transcription,
            diarization=diarization,
            character_assigner=character_assigner,
        )

    def to_dict(self, *, exclude_defaults: bool = False) -> dict:
        configuration = {
            "audio": {
                "trim_start": self.trim_start,
                "trim_end": self.trim_end,
            },
            "transcription": self.transcription.model_dump(
                exclude_defaults=exclude_defaults
            ),
        }
        if self.diarization is not None:
            configuration["diarization"] = self.diarization.model_dump(
                exclude_defaults=exclude_defaults,
                exclude={"hf_token"},
            )
        if self.character_assigner is not None:
            configuration["character_assigner"] = self.character_assigner.model_dump(
                exclude_defaults=exclude_defaults,
                exclude={"embedder": {"hf_token"}},
            )
        return configuration

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineConfig:
        return cls(
            trim_start=data["audio"]["trim_start"],
            trim_end=data["audio"]["trim_end"],
            transcription=TranscriberConfig.model_validate(data["transcription"]),
            diarization=(
                DiarizerConfig.model_validate(data["diarization"])
                if data.get("diarization") is not None
                else None
            ),
            character_assigner=(
                CharacterAssignerConfig.model_validate(data["character_assigner"])
                if data.get("character_assigner") is not None
                else None
            ),
        )

    @property
    def stages(self) -> tuple[Stage, ...]:
        stages = [Stage.TRANSCRIPTION]
        if self.diarization is not None:
            stages.append(Stage.DIARIZATION)
        if self.character_assigner is not None:
            stages.append(Stage.CHARACTER_ASSIGNER)
        return tuple(stages)

    def generated_name(self) -> str:
        config_id = configuration_id(self.to_dict(exclude_defaults=True))
        model = Path(self.transcription.model_name).name
        parts = ["transcription", _slug(model)]
        parts.extend(stage.value.replace("_", "-") for stage in self.stages[1:])
        return f"{'-'.join(parts)}-{config_id[:8]}"

    def runner_for(self, stage: Stage, media: MediaStore):
        match stage:
            case Stage.TRANSCRIPTION:
                return Transcriber(self.transcription)
            case Stage.DIARIZATION:
                return Diarizer(self.diarization)
            case Stage.CHARACTER_ASSIGNER:
                return CharacterAssigner(self.character_assigner, media)


def run_pipeline(
    tracker: MlflowTracker,
    media: MediaStore,
    config: PipelineConfig,
    episode_keys: set[str],
    *,
    through: Stage | None = None,
    name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> str:
    if config.trim_start != media.trim_start or config.trim_end != media.trim_end:
        raise ValueError("PipelineConfig trimming must match MediaStore")

    stages = config.stages
    if through is None:
        requested = stages
    elif through not in stages:
        raise ValueError(f"Stage {through} is not configured")
    else:
        requested = stages[: stages.index(through) + 1]

    run_name = name or config.generated_name()

    with tracker.pipeline_run(
        run_name,
        config,
        stages=set(requested),
        tags=tags,
    ) as parent_run_id:
        states = tracker.episode_run_states(parent_run_id)
        todo = []

        for episode_key in episode_keys:
            state = states.get(episode_key)
            completed = state.completed_stages if state else frozenset()
            pending = tracker.pending_stages(
                completed=completed,
                requested=set(requested),
            )
            if pending:
                todo.append((episode_key, state, pending))

        if not todo:
            tracker.log_parent_averages(parent_run_id)
            return parent_run_id

        needed = {stage for _, _, pending in todo for stage in pending}
        runners = {}

        try:
            for stage in config.stages:
                if stage in needed:
                    runners[stage] = config.runner_for(stage, media)

            for episode_key, state, pending in todo:
                episode = None
                try:
                    episode = media.episode(episode_key)
                    with tracker.episode_run(episode_key):
                        if state is not None:
                            first_pending_index = config.stages.index(pending[0])
                            if first_pending_index > 0:
                                tracker.restore_stage(
                                    episode,
                                    state.run_id,
                                    config.stages[first_pending_index - 1],
                                )

                        for stage in pending:
                            episode = runners[stage].run(episode)
                            tracker.log_stage(
                                episode,
                                stage,
                                metrics_for(stage, episode),
                            )

                        media.save(episode, subdir=run_name)
                except Exception:
                    logger.exception(
                        "Episode %s failed; continuing",
                        episode_key,
                    )
                finally:
                    if episode is not None:
                        episode.clear_audio()
        finally:
            for runner in reversed(tuple(runners.values())):
                runner.free()
            tracker.log_parent_averages(parent_run_id)

    return parent_run_id


def metrics_for(stage: Stage, episode) -> dict[str, Any]:
    match stage:
        case Stage.TRANSCRIPTION:
            return alignment_metrics(episode) | repetition_metrics(episode)
        case Stage.DIARIZATION:
            return diarization_metrics(episode)
        case Stage.CHARACTER_ASSIGNER:
            return assignment_metrics(episode)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")

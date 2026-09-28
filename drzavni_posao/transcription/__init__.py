"""Public transcription API.

Exports are loaded lazily so search and generation can use transcript models
without initializing the audio transcription stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "SR": "constants",
    "Episode": "episode",
    "EpisodeMetadata": "episode",
    "Transcript": "transcript",
    "MediaStore": "store",
    "Transcriber": "transcriber",
    "TranscriberConfig": "transcriber",
    "Diarizer": "diarizer",
    "DiarizerConfig": "diarizer",
    "Embedder": "embedder",
    "EmbedderConfig": "embedder",
    "Gallery": "gallery",
    "PipelineConfig": "pipeline",
    "run_pipeline": "pipeline",
    "CharacterAssigner": "character_assigner",
    "CharacterAssignerConfig": "character_assigner",
    "MlflowTracker": "tracking",
    "Stage": "tracking",
    "evaluate_run": "evaluation",
}

__all__ = [
    "SR",
    "Episode",
    "EpisodeMetadata",
    "Transcript",
    "MediaStore",
    "Transcriber",
    "TranscriberConfig",
    "Diarizer",
    "DiarizerConfig",
    "Embedder",
    "EmbedderConfig",
    "Gallery",
    "PipelineConfig",
    "run_pipeline",
    "CharacterAssigner",
    "CharacterAssignerConfig",
    "MlflowTracker",
    "Stage",
    "evaluate_run",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value

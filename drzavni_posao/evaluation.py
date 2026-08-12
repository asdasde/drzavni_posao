import json
from pathlib import Path

import mlflow
import pandas as pd

from .metrics import (
    assignment_scores,
    character_table,
    confusion_table,
    labeled_segments,
)
from .tracking import MlflowTracker, Stage
from .transcript import Transcript

ASSIGNMENT_PREFIX = "assignment"


def _cached_json(cache_dir: Path | None, relative_path: Path, download) -> dict:
    """Run artifacts never change once logged, so a hit needs no freshness check."""
    if cache_dir is None:
        return download()

    path = cache_dir / relative_path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    data = download()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def run_labeled_segments(
    tracker: MlflowTracker,
    run_id: str,
    label_dir: Path,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Labeled segments of every episode of a run that has saved labels."""
    config = _cached_json(
        cache_dir,
        Path(run_id) / "config.json",
        lambda: tracker._configuration(run_id),
    )
    trim_start = config["audio"]["trim_start"]
    episodes = []
    for episode_key, state in tracker.episode_run_states(run_id).items():
        path = label_dir / f"{episode_key}.json"
        if Stage.CHARACTER_ASSIGNER not in state.completed_stages or not path.exists():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))["segments"]
        if not labels:
            continue

        transcript = Transcript.from_dict(
            _cached_json(
                cache_dir,
                Path(run_id) / f"{episode_key}.json",
                lambda: tracker.stage_transcript(
                    state.run_id,
                    Stage.CHARACTER_ASSIGNER,
                ).to_dict(),
            )
        )
        segments = labeled_segments(labels, transcript.segments, trim_start)
        segments.insert(0, "episode_key", episode_key)
        segments.insert(1, "episode_run_id", state.run_id)
        episodes.append(segments)

    if not episodes:
        raise ValueError(f"Run {run_id} has no labeled episodes in {label_dir}")
    return pd.concat(episodes, ignore_index=True)


def _log_assignment(segments: pd.DataFrame) -> None:
    scores = assignment_scores(segments)
    mlflow.log_metrics(
        {f"{ASSIGNMENT_PREFIX}.{name}": value for name, value in scores.items()}
    )
    tables = {
        "segments": segments.to_csv(index=False),
        "confusion": confusion_table(segments).to_csv(),
        "characters": character_table(segments).to_csv(),
    }
    for name, csv in tables.items():
        mlflow.log_text(csv, f"{ASSIGNMENT_PREFIX}/{name}.csv")


def evaluate_run(
    tracker: MlflowTracker,
    run_id: str,
    label_dir: Path,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Score a character-assignment run against saved labels, back onto its runs.

    Every episode run is scored on its own labels; the parent is scored on all of
    them pooled, so a heavily labeled episode weighs more there than a thin one.

    Metrics and CSV tables land under the ``assignment/`` prefix
    (e.g. ``assignment.accuracy``, ``assignment/confusion.csv``).

    A ``cache_dir`` keeps the downloaded configuration and transcripts on disk,
    which turns re-scoring from ~20 seconds per episode into milliseconds.
    """
    segments = run_labeled_segments(tracker, run_id, label_dir, cache_dir)
    with mlflow.start_run(run_id=run_id):
        for episode_run_id, episode in segments.groupby("episode_run_id"):
            with mlflow.start_run(run_id=episode_run_id, nested=True):
                _log_assignment(episode)
        _log_assignment(segments)
    return segments

"""One training run on Databricks, tracked in MLflow with a checkpoint on the volume."""

import gc
import traceback
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import mlflow
import numpy as np
import torch

from .prepare import read_jsonl
from .train import (
    TrainConfig,
    default_device,
    format_metrics,
    sample_turns,
    save_checkpoint,
    train,
)


VOLUME = Path("/Volumes/luka_erceg/drzavni_posao/lstm")
EXPERIMENT_NAME = "/Users/luka.erceg@databricks.com/drzavni_posao/lstm"


def tracked_run(
    name: str, config: TrainConfig, samples_per_speaker: int = 3, temperature: float = 0.8
) -> dict[str, float]:
    """Train on the volume's episodes and log params, metrics, checkpoint and samples."""
    checkpoint_dir = VOLUME / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(VOLUME / "episodes.jsonl")
    device = default_device()
    print(f"training {name} on {device}", flush=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(asdict(config))
        model, results = train(
            config,
            episodes,
            device,
            on_epoch=_logger("epoch"),
            on_step=_logger("  step"),
        )
        mlflow.log_metrics({f"best_{k}": v for k, v in results.items()})
        checkpoint = checkpoint_dir / f"{name}-{run.info.run_id}.pt"
        save_checkpoint(model, config, checkpoint)
        mlflow.log_artifact(str(checkpoint))
        rng = np.random.default_rng(config.seed)
        samples = sample_turns(model, rng, samples_per_speaker, temperature)
        mlflow.log_text("\n".join(samples), "samples.txt")
    print(results)
    print("\n".join(samples))
    return results


def run_queue(runs: list[tuple[str, TrainConfig]]) -> dict[str, dict[str, float] | str]:
    """Run each (name, config) in turn; a failure is reported and the queue goes on.

    GPU memory is released between runs, so one run's OOM does not starve the next.
    Batches load in the main process: on serverless, after a few runs in one process
    loader workers can no longer hand batches over ("Pin memory thread exited").
    Cancelling the cell still stops the whole queue: KeyboardInterrupt is not caught.
    """
    outcomes: dict[str, dict[str, float] | str] = {}
    for name, config in runs:
        try:
            outcomes[name] = tracked_run(name, replace(config, num_workers=0))
        except Exception as error:
            traceback.print_exc()
            outcomes[name] = f"failed: {type(error).__name__}: {error}"
        gc.collect()
        torch.cuda.empty_cache()
    for name, outcome in outcomes.items():
        print(f"{name}: {outcome}")
    return outcomes


def _logger(label: str) -> Callable[[int, dict[str, float]], None]:
    def log(step: int, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics, step=step)
        print(f"{label} {step}: {format_metrics(metrics)}", flush=True)

    return log

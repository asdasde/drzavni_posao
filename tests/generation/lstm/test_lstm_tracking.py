import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow

from drzavni_posao.generation.lstm import tracking
from drzavni_posao.generation.lstm.train import TrainConfig
from drzavni_posao.transcription.constants import CORE_CHARACTERS


EPISODE = {
    "premise": "Kafa.",
    "turns": [
        {"speaker": "Boskic", "text": "Ima kafe?"},
        {"speaker": "Djordje Cvarkov", "text": "Nema."},
        {"speaker": "Dragan Torbica", "text": "Ajde."},
    ],
}


class TrackedRunTests(unittest.TestCase):
    def test_logs_metrics_checkpoint_and_samples(self):
        config = TrainConfig(
            embedding_dim=8,
            role_dim=4,
            hidden_size=16,
            num_layers=1,
            context_turns=1,
            epochs=1,
            num_workers=0,
            log_every=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            (volume / "episodes.jsonl").write_text(
                "\n".join(
                    json.dumps({**EPISODE, "split": split})
                    for split in ("train", "validation", "test")
                )
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "MLFLOW_TRACKING_URI": (volume / "mlruns").as_uri(),
                        "MLFLOW_ALLOW_FILE_STORE": "true",
                    },
                ),
                mock.patch.object(tracking, "VOLUME", volume),
                mock.patch.object(tracking, "EXPERIMENT_NAME", "test"),
            ):
                results = tracking.tracked_run("tiny", config, samples_per_speaker=1)
                run = mlflow.last_active_run()
                artifacts = mlflow.artifacts.download_artifacts(run_id=run.info.run_id)
                samples = (Path(artifacts) / "samples.txt").read_text().splitlines()

            self.assertEqual(run.info.run_name, "tiny")
            self.assertEqual(run.data.metrics["best_test_bpc"], results["test_bpc"])
            self.assertIn("train_bits_per_token", run.data.metrics)
            self.assertEqual(len(list((volume / "checkpoints").glob("tiny-*.pt"))), 1)
            self.assertEqual(len(samples), len(CORE_CHARACTERS))
            self.assertTrue(samples[0].startswith(f"[{CORE_CHARACTERS[0]}]"))

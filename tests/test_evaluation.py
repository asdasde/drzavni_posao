import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd

from drzavni_posao import evaluation
from drzavni_posao.tracking import MlflowTracker, Stage

CONFIGURATION = {
    "audio": {"path": "/audio", "trim_start": 25, "trim_end": 7},
    "transcription": {"model": "sr-cmb"},
    "character_assigner": {"embedder": {"model": "wespeaker"}},
}


class StubTranscript:
    def __init__(self, segments):
        self.segments = segments
        self.file = "episode.wav"

    def render(self):
        return "zdravo"

    def to_dict(self):
        return {"file": self.file, "text": self.render(), "segments": self.segments}


class EvaluateRunTests(unittest.TestCase):
    def setUp(self):
        tracking_uri = mlflow.get_tracking_uri()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(mlflow.set_tracking_uri, tracking_uri)

        mlflow.set_tracking_uri(f"sqlite:///{Path(directory.name, 'mlflow.db')}")
        mlflow.MlflowClient().create_experiment(
            "evaluation-tests",
            artifact_location=Path(directory.name, "artifacts").as_uri(),
        )
        self.tracker = MlflowTracker("evaluation-tests")
        self.label_dir = Path(directory.name, "labels")
        self.label_dir.mkdir()

    def tearDown(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def write_labels(self, episode_key, segments):
        (self.label_dir / f"{episode_key}.json").write_text(
            json.dumps({"episode_key": episode_key, "segments": segments}),
            encoding="utf-8",
        )

    def assignment_run(self, episodes):
        with self.tracker.pipeline_run(
            "assign",
            CONFIGURATION,
            stages={Stage.CHARACTER_ASSIGNER},
        ) as run_id:
            for episode_key, segments in episodes.items():
                with self.tracker.episode_run(episode_key):
                    self.tracker.log_stage(
                        SimpleNamespace(transcript=StubTranscript(segments)),
                        Stage.CHARACTER_ASSIGNER,
                        {},
                    )
        return run_id

    def test_scores_labels_against_the_trimmed_prediction_timeline(self):
        run_id = self.assignment_run(
            {
                "111": [
                    {"start": 0, "end": 5, "speaker": "Djordje"},
                    {"start": 5, "end": 10, "speaker": "Djordje"},
                ]
            }
        )
        self.write_labels(
            "111",
            [
                {"start": 25, "end": 30, "speaker": "Djordje"},
                {"start": 30, "end": 35, "speaker": "Boskic"},
            ],
        )

        segments = evaluation.evaluate_run(self.tracker, run_id, self.label_dir)

        self.assertEqual(set(segments["episode_key"]), {"111"})
        self.assertEqual(segments["seconds"].sum(), 10)

        episode_run_id = self.tracker.episode_run_states(run_id)["111"].run_id
        for scored in (run_id, episode_run_id):
            metrics = self.tracker.client.get_run(scored).data.metrics
            self.assertEqual(metrics["assignment.accuracy"], 0.5)
            self.assertEqual(metrics["assignment.coverage"], 1)
        artifacts = {
            artifact.path
            for artifact in self.tracker.client.list_artifacts(
                episode_run_id,
                "assignment",
            )
        }
        self.assertEqual(
            artifacts,
            {
                "assignment/segments.csv",
                "assignment/confusion.csv",
                "assignment/characters.csv",
            },
        )

    def test_parent_pools_episodes_by_labeled_duration(self):
        run_id = self.assignment_run(
            {
                "111": [{"start": 0, "end": 30, "speaker": "Djordje"}],
                "112": [{"start": 0, "end": 10, "speaker": "Boskic"}],
            }
        )
        self.write_labels("111", [{"start": 25, "end": 55, "speaker": "Djordje"}])
        self.write_labels("112", [{"start": 25, "end": 35, "speaker": "Djordje"}])

        evaluation.evaluate_run(self.tracker, run_id, self.label_dir)

        metrics = self.tracker.client.get_run(run_id).data.metrics
        self.assertEqual(metrics["assignment.labeled_seconds"], 40)
        self.assertEqual(metrics["assignment.accuracy"], 0.75)

    def test_episodes_without_labels_are_skipped(self):
        run_id = self.assignment_run(
            {
                "111": [{"start": 0, "end": 10, "speaker": "Djordje"}],
                "112": [{"start": 0, "end": 10, "speaker": "Boskic"}],
            }
        )
        self.write_labels("112", [{"start": 25, "end": 35, "speaker": "Boskic"}])

        segments = evaluation.evaluate_run(self.tracker, run_id, self.label_dir)

        self.assertEqual(set(segments["episode_key"]), {"112"})

    def test_cache_dir_serves_a_second_evaluation_without_downloads(self):
        run_id = self.assignment_run(
            {"111": [{"start": 0, "end": 10, "speaker": "Djordje"}]}
        )
        self.write_labels("111", [{"start": 25, "end": 35, "speaker": "Djordje"}])
        cache_dir = self.label_dir.parent / "cache"

        first = evaluation.evaluate_run(
            self.tracker,
            run_id,
            self.label_dir,
            cache_dir,
        )

        def fail(*args, **kwargs):
            raise AssertionError("artifact was downloaded twice")

        self.tracker.stage_transcript = fail
        self.tracker._configuration = fail
        second = evaluation.evaluate_run(
            self.tracker,
            run_id,
            self.label_dir,
            cache_dir,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((cache_dir / run_id / "config.json").exists())
        self.assertTrue((cache_dir / run_id / "111.json").exists())

    def test_run_without_any_labels_raises(self):
        run_id = self.assignment_run(
            {"111": [{"start": 0, "end": 10, "speaker": "Djordje"}]}
        )

        with self.assertRaises(ValueError):
            evaluation.evaluate_run(self.tracker, run_id, self.label_dir)


if __name__ == "__main__":
    unittest.main()

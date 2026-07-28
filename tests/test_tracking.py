import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlflow

import drzavni_posao
from drzavni_posao import tracking

Stage = tracking.Stage


class StubTranscript:
    def __init__(self, segments=None, file="episode.wav"):
        self.segments = segments or [{"text": "zdravo", "words": []}]
        self.file = file

    def render(self):
        return "zdravo"

    def to_dict(self):
        return {
            "file": self.file,
            "text": self.render(),
            "segments": self.segments,
        }


class MlflowTrackerTests(unittest.TestCase):
    def setUp(self):
        self._tracking_uri = mlflow.get_tracking_uri()
        self._tempdir = tempfile.TemporaryDirectory()
        database = Path(self._tempdir.name, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{database}")
        mlflow.MlflowClient().create_experiment(
            "tracking-tests",
            artifact_location=Path(self._tempdir.name, "artifacts").as_uri(),
        )
        self.tracker = tracking.MlflowTracker("tracking-tests")

    def tearDown(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(self._tracking_uri)
        self._tempdir.cleanup()

    @staticmethod
    def configuration(
        *,
        asr_model="asr-a",
        diar_model="diar-a",
        assignment_threshold=0.5,
    ):
        return {
            "audio": {
                "path": "/audio",
                "trim_start": 25,
                "trim_end": 7,
            },
            "asr": {
                "model": asr_model,
                "options": ["beam"],
            },
            "diarization": {"model": diar_model},
            "character_assignment": {
                "threshold": assignment_threshold,
                "embedder": {"model": "embed-a"},
                "gallery": {"Đorđe": ["sample-a"]},
            },
        }

    @staticmethod
    def episode(*, diarization=None):
        return SimpleNamespace(
            transcript=StubTranscript(),
            diarization=diarization,
        )

    def test_tracker_is_exported_from_package(self):
        self.assertIs(drzavni_posao.MlflowTracker, tracking.MlflowTracker)

    def test_stage_enum_follows_pipeline_order(self):
        self.assertTrue(hasattr(tracking, "Stage"))
        self.assertIs(drzavni_posao.Stage, tracking.Stage)
        self.assertIs(tracking.Stage.ASR.next, tracking.Stage.DIAR)
        self.assertIs(tracking.Stage.DIAR.next, tracking.Stage.ASSIGN)
        self.assertIsNone(tracking.Stage.ASSIGN.next)
        self.assertEqual(
            tracking.Stage.DIAR.config_keys,
            ("audio", "asr", "diarization"),
        )
        self.assertEqual(
            tracking.Stage.ASSIGN.config_keys,
            ("audio", "asr", "character_assignment"),
        )
        self.assertEqual(tracking.Stage.ASSIGN.prerequisites, frozenset({Stage.ASR}))
        for stage in tracking.Stage:
            self.assertEqual(
                stage.required_artifacts,
                frozenset(
                    {
                        f"{stage}/transcript.json",
                        f"{stage}/transcript.txt",
                    }
                ),
            )

    def test_pipeline_run_resumes_only_when_requested_or_explicit(self):
        configuration = self.configuration()

        with self.tracker.pipeline_run(
            "first",
            configuration,
            stages=(Stage.ASR,),
        ) as first:
            pass
        with self.tracker.pipeline_run(
            "resumed",
            configuration,
            stages=(Stage.DIAR,),
            resume_matching_run=True,
        ) as resumed:
            pass
        with self.tracker.pipeline_run(
            "independent",
            configuration,
            stages=(Stage.ASR,),
        ) as independent:
            pass

        self.assertEqual(resumed, first)
        self.assertNotEqual(independent, first)

        with self.assertRaisesRegex(ValueError, "configuration"):
            with self.tracker.pipeline_run(
                "invalid",
                self.configuration(asr_model="other"),
                stages=(Stage.ASR,),
                run_id=first,
            ):
                pass

    def test_episode_state_uses_key_and_completed_stage_not_split_or_status(self):
        configuration = self.configuration()
        with self.tracker.pipeline_run(
            "source",
            configuration,
            stages=(Stage.ASR, Stage.DIAR),
        ) as parent_run_id:
            with mlflow.start_run(
                run_name="stale",
                nested=True,
                tags={"episode_key": "1", "split": "scribe"},
            ):
                mlflow.set_tag("stage.asr", "complete")
                mlflow.log_metric("asr.word_count", 2)
            with self.tracker.episode_run("1"):
                self.tracker.log_stage(
                    self.episode(diarization=[{"speaker": "SPEAKER_00"}]),
                    Stage.DIAR,
                    {"turn_count": 1},
                )

        states = self.tracker.episode_states(parent_run_id)

        self.assertEqual(set(states), {"1"})
        self.assertEqual(states["1"].completed_stages, frozenset({Stage.DIAR}))

    def test_log_and_restore_stage_round_trip(self):
        configuration = self.configuration()
        with self.tracker.pipeline_run(
            "source",
            configuration,
            stages=(Stage.ASR,),
        ) as parent_run_id:
            with self.tracker.episode_run("1"):
                result = self.tracker.log_stage(
                    self.episode(),
                    Stage.ASR,
                    {"word_count": 3, "bad": math.nan},
                )

        state = self.tracker.episode_states(parent_run_id)["1"]
        restored = SimpleNamespace(transcript=None, diarization=None)
        self.tracker.restore_stage(restored, state.run_id, Stage.ASR)
        run = self.tracker.client.get_run(state.run_id)

        self.assertIsNone(result)
        self.assertEqual(restored.transcript.render(), "zdravo")
        self.assertEqual(run.data.metrics, {"asr.word_count": 3.0})
        self.assertEqual(run.data.tags["stage.asr"], "complete")

    def test_pending_stages_requires_completed_or_requested_prerequisites(self):
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.ASR},
                requested=(Stage.DIAR, Stage.ASSIGN),
            ),
            ("diar", "assign"),
        )
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.ASR},
                requested=(Stage.ASSIGN,),
            ),
            ("assign",),
        )
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.ASR, Stage.DIAR},
                requested=(Stage.DIAR, Stage.ASSIGN),
            ),
            ("assign",),
        )
        with self.assertRaisesRegex(ValueError, "asr"):
            self.tracker.pending_stages(
                completed=set(),
                requested=(Stage.DIAR, Stage.ASSIGN),
            )

    def test_string_stage_names_are_rejected(self):
        with self.assertRaises(TypeError):
            self.tracker.pending_stages(
                completed=set(),
                requested=("asr",),
            )

    def test_fork_run_copies_only_validated_completed_prefix_and_is_idempotent(self):
        source_configuration = self.configuration()
        target_configuration = self.configuration(
            diar_model="diar-b",
            assignment_threshold=0.7,
        )
        with self.tracker.pipeline_run(
            "source",
            source_configuration,
            stages=(Stage.ASR,),
        ) as source_run_id:
            with self.tracker.episode_run("1"):
                self.tracker.log_stage(
                    self.episode(),
                    Stage.ASR,
                    {"word_count": 3},
                )
            with self.tracker.episode_run("2"):
                pass

        target_run_id = self.tracker.fork_run(
            source_run_id,
            "derived",
            target_configuration,
            copy_until=Stage.ASR,
        )
        repeated_run_id = self.tracker.fork_run(
            source_run_id,
            "derived",
            target_configuration,
            copy_until=Stage.ASR,
        )

        self.assertEqual(repeated_run_id, target_run_id)
        states = self.tracker.episode_states(target_run_id)
        self.assertEqual(states["1"].completed_stages, frozenset({Stage.ASR}))
        self.assertEqual(states["2"].completed_stages, frozenset())
        copied_run = self.tracker.client.get_run(states["1"].run_id)
        self.assertEqual(copied_run.data.metrics["asr.word_count"], 3)
        self.assertEqual(
            copied_run.data.tags["source.episode_run_id"],
            self.tracker.episode_states(source_run_id)["1"].run_id,
        )

        restored = SimpleNamespace(transcript=None, diarization=None)
        self.tracker.restore_stage(restored, states["1"].run_id, Stage.ASR)
        self.assertEqual(restored.transcript.render(), "zdravo")

        with self.tracker.pipeline_run(
            "derived",
            target_configuration,
            stages=(Stage.DIAR, Stage.ASSIGN),
            run_id=target_run_id,
        ) as resumed:
            self.assertEqual(resumed, target_run_id)

    def test_fork_run_rejects_changed_copied_configuration(self):
        source_configuration = self.configuration()
        with self.tracker.pipeline_run(
            "source",
            source_configuration,
            stages=(Stage.ASR,),
        ) as source_run_id:
            pass

        with self.assertRaisesRegex(ValueError, "asr"):
            self.tracker.fork_run(
                source_run_id,
                "invalid",
                self.configuration(asr_model="asr-b"),
                copy_until=Stage.ASR,
            )

    def test_parent_averages_and_episode_metric_query_use_latest_children(self):
        configuration = self.configuration()
        with self.tracker.pipeline_run(
            "source",
            configuration,
            stages=(Stage.ASR,),
        ) as parent_run_id:
            for key, count in (("1", 2), ("2", 4)):
                with self.tracker.episode_run(key):
                    self.tracker.log_stage(
                        self.episode(),
                        Stage.ASR,
                        {"word_count": count},
                    )
            self.tracker.log_parent_averages(parent_run_id)

        parent = self.tracker.client.get_run(parent_run_id)
        self.assertEqual(parent.data.metrics["mean.asr.word_count"], 3)
        self.assertEqual(
            self.tracker.episode_metrics("source", "asr.word_count"),
            {"1": 2, "2": 4},
        )


if __name__ == "__main__":
    unittest.main()

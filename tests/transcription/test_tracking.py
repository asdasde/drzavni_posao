import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow

from drzavni_posao import transcription
from drzavni_posao.transcription import tracking
from drzavni_posao.transcription.pipeline import PipelineConfig

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

    def start_run(self, name, configuration, **kwargs):
        return self.tracker.pipeline_run(
            name,
            PipelineConfig.from_dict(configuration),
            **kwargs,
        )

    def start_fork(self, source_run_id, name, configuration, **kwargs):
        return self.tracker.fork_run(
            source_run_id,
            name,
            PipelineConfig.from_dict(configuration),
            **kwargs,
        )

    @staticmethod
    def configuration(
        *,
        transcription_model="transcription-a",
        diar_model="diar-a",
        character_assigner_threshold=0.5,
    ):
        from drzavni_posao.transcription.character_assigner import CharacterAssignerConfig
        from drzavni_posao.transcription.diarizer import DiarizerConfig
        from drzavni_posao.transcription.embedder import EmbedderConfig
        from drzavni_posao.transcription.transcriber import TranscriberConfig

        return {
            "audio": {
                "path": "/audio",
                "trim_start": 25,
                "trim_end": 7,
            },
            "transcription": TranscriberConfig(
                model_name=transcription_model,
            ).model_dump(),
            "diarization": DiarizerConfig(
                diarize_model=diar_model,
            ).model_dump(exclude={"hf_token"}),
            "character_assigner": CharacterAssignerConfig(
                min_similarity=character_assigner_threshold,
                embedder=EmbedderConfig(embed_model="embed-a"),
                speaker_samples={
                    "Đorđe": [{"episode_key": "1", "start": 0.0, "end": 1.0}]
                },
            ).model_dump(exclude={"embedder": {"hf_token"}}),
        }

    @staticmethod
    def episode(*, diarization=None):
        return SimpleNamespace(
            transcript=StubTranscript(),
            diarization=diarization,
        )

    def test_tracker_is_exported_from_package(self):
        self.assertIs(transcription.MlflowTracker, tracking.MlflowTracker)

    def test_stage_enum_follows_pipeline_order(self):
        self.assertTrue(hasattr(tracking, "Stage"))
        self.assertIs(transcription.Stage, tracking.Stage)
        self.assertIs(
            tracking.Stage.TRANSCRIPTION.next,
            tracking.Stage.DIARIZATION,
        )
        self.assertIs(
            tracking.Stage.DIARIZATION.next,
            tracking.Stage.CHARACTER_ASSIGNER,
        )
        self.assertIsNone(tracking.Stage.CHARACTER_ASSIGNER.next)
        self.assertEqual(
            tracking.Stage.DIARIZATION.config_keys,
            ("audio", "transcription", "diarization"),
        )
        self.assertEqual(
            tracking.Stage.CHARACTER_ASSIGNER.config_keys,
            ("audio", "transcription", "character_assigner"),
        )
        self.assertEqual(
            tracking.Stage.CHARACTER_ASSIGNER.prerequisites,
            frozenset({Stage.TRANSCRIPTION}),
        )
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

    def test_pipeline_run_reuses_exact_configuration(self):
        configuration = self.configuration()
        reordered = {
            "character_assigner": configuration["character_assigner"],
            "diarization": configuration["diarization"],
            "transcription": configuration["transcription"],
            "audio": configuration["audio"],
        }

        with self.start_run(
            "first",
            configuration,
            stages={Stage.TRANSCRIPTION},
        ) as first:
            pass
        with self.start_run(
            "different-display-name",
            reordered,
            stages={Stage.TRANSCRIPTION},
        ) as resumed:
            pass

        self.assertEqual(resumed, first)
        run = self.tracker.client.get_run(first)
        self.assertRegex(run.data.tags["config_id"], r"^[0-9a-f]{64}$")

    def test_pipeline_run_separates_changed_configuration(self):
        with self.start_run(
            "first",
            self.configuration(),
            stages={Stage.TRANSCRIPTION},
        ) as first:
            pass
        with self.start_run(
            "changed",
            self.configuration(transcription_model="other"),
            stages={Stage.TRANSCRIPTION},
        ) as changed:
            pass

        self.assertNotEqual(changed, first)

    def test_episode_run_states_use_key_and_completed_stage_not_split_or_status(self):
        configuration = self.configuration()
        with self.start_run(
            "source",
            configuration,
            stages={Stage.TRANSCRIPTION, Stage.DIARIZATION},
        ) as parent_run_id:
            with mlflow.start_run(
                run_name="stale",
                nested=True,
                tags={"episode_key": "1", "split": "scribe"},
            ):
                mlflow.set_tag("stage.transcription", "complete")
                mlflow.log_metric("transcription.word_count", 2)
            with self.tracker.episode_run("1"):
                self.tracker.log_stage(
                    self.episode(diarization=[{"speaker": "SPEAKER_00"}]),
                    Stage.DIARIZATION,
                    {"turn_count": 1},
                )

        states = self.tracker.episode_run_states(parent_run_id)

        self.assertEqual(set(states), {"1"})
        self.assertEqual(
            states["1"].completed_stages,
            frozenset({Stage.DIARIZATION}),
        )

    def test_child_runs_follow_all_search_pages(self):
        class Page(list):
            def __init__(self, runs, token):
                super().__init__(runs)
                self.token = token

        first = SimpleNamespace(
            info=SimpleNamespace(run_id="first"),
            data=SimpleNamespace(tags={"run_level": "episode"}),
        )
        stale = SimpleNamespace(
            info=SimpleNamespace(run_id="stale"),
            data=SimpleNamespace(tags={}),
        )
        second = SimpleNamespace(
            info=SimpleNamespace(run_id="second"),
            data=SimpleNamespace(tags={"run_level": "episode"}),
        )
        self.tracker.client = MagicMock()
        self.tracker.client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(experiment_id="experiment")
        )
        self.tracker.client.search_runs.side_effect = [
            Page([first, stale], "next-page"),
            Page([second], None),
        ]

        runs = self.tracker._child_runs("parent")

        self.assertEqual(runs, [first, second])
        self.assertEqual(
            [
                call.kwargs["page_token"]
                for call in self.tracker.client.search_runs.call_args_list
            ],
            [None, "next-page"],
        )
        self.assertEqual(
            [
                call.kwargs["max_results"]
                for call in self.tracker.client.search_runs.call_args_list
            ],
            [100, 100],
        )
        self.assertEqual(
            {
                call.kwargs["filter_string"]
                for call in self.tracker.client.search_runs.call_args_list
            },
            {"tags.`mlflow.parentRunId` = 'parent'"},
        )

    def test_episode_run_reuses_unique_child(self):
        with self.start_run(
            "parent",
            self.configuration(),
            stages={Stage.TRANSCRIPTION},
        ) as parent_run_id:
            with self.tracker.episode_run("1"):
                first_run_id = mlflow.active_run().info.run_id
            with self.tracker.episode_run("1"):
                resumed_run_id = mlflow.active_run().info.run_id

        self.assertEqual(resumed_run_id, first_run_id)
        self.assertEqual(
            len(self.tracker._child_runs(parent_run_id)),
            1,
        )

    def test_log_and_restore_stage_round_trip(self):
        configuration = self.configuration()
        with self.start_run(
            "source",
            configuration,
            stages={Stage.TRANSCRIPTION},
        ) as parent_run_id:
            with self.tracker.episode_run("1"):
                result = self.tracker.log_stage(
                    self.episode(),
                    Stage.TRANSCRIPTION,
                    {"word_count": 3, "bad": math.nan},
                )

        state = self.tracker.episode_run_states(parent_run_id)["1"]
        restored = SimpleNamespace(transcript=None, diarization=None)
        self.tracker.restore_stage(
            restored,
            state.run_id,
            Stage.TRANSCRIPTION,
        )
        run = self.tracker.client.get_run(state.run_id)

        self.assertIsNone(result)
        self.assertEqual(restored.transcript.render(), "zdravo")
        self.assertEqual(run.data.metrics, {"transcription.word_count": 3.0})
        self.assertEqual(run.data.tags["stage.transcription"], "complete")

    def test_pending_stages_requires_completed_or_requested_prerequisites(self):
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.TRANSCRIPTION},
                requested={
                    Stage.DIARIZATION,
                    Stage.CHARACTER_ASSIGNER,
                },
            ),
            ("diarization", "character_assigner"),
        )
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.TRANSCRIPTION},
                requested={Stage.CHARACTER_ASSIGNER},
            ),
            ("character_assigner",),
        )
        self.assertEqual(
            self.tracker.pending_stages(
                completed={Stage.TRANSCRIPTION, Stage.DIARIZATION},
                requested={
                    Stage.DIARIZATION,
                    Stage.CHARACTER_ASSIGNER,
                },
            ),
            ("character_assigner",),
        )
        with self.assertRaisesRegex(ValueError, "transcription"):
            self.tracker.pending_stages(
                completed=set(),
                requested={
                    Stage.DIARIZATION,
                    Stage.CHARACTER_ASSIGNER,
                },
            )

    def test_fork_run_copies_only_validated_completed_prefix_and_is_idempotent(self):
        source_configuration = self.configuration()
        target_configuration = self.configuration(
            diar_model="diar-b",
            character_assigner_threshold=0.7,
        )
        with self.start_run(
            "source",
            source_configuration,
            stages={Stage.TRANSCRIPTION},
        ) as source_run_id:
            with self.tracker.episode_run("1"):
                self.tracker.log_stage(
                    self.episode(),
                    Stage.TRANSCRIPTION,
                    {"word_count": 3},
                )
            with self.tracker.episode_run("2"):
                pass

        target_run_id = self.start_fork(
            source_run_id,
            "derived",
            target_configuration,
            copy_until=Stage.TRANSCRIPTION,
        )
        repeated_run_id = self.start_fork(
            source_run_id,
            "different-display-name",
            target_configuration,
            copy_until=Stage.TRANSCRIPTION,
        )

        self.assertEqual(repeated_run_id, target_run_id)
        states = self.tracker.episode_run_states(target_run_id)
        self.assertEqual(
            states["1"].completed_stages,
            frozenset({Stage.TRANSCRIPTION}),
        )
        self.assertEqual(states["2"].completed_stages, frozenset())
        copied_run = self.tracker.client.get_run(states["1"].run_id)
        self.assertEqual(
            copied_run.data.metrics["transcription.word_count"],
            3,
        )
        self.assertEqual(
            copied_run.data.tags["source.episode_run_id"],
            self.tracker.episode_run_states(source_run_id)["1"].run_id,
        )

        restored = SimpleNamespace(transcript=None, diarization=None)
        self.tracker.restore_stage(
            restored,
            states["1"].run_id,
            Stage.TRANSCRIPTION,
        )
        self.assertEqual(restored.transcript.render(), "zdravo")

        with self.start_run(
            "derived",
            target_configuration,
            stages={Stage.DIARIZATION, Stage.CHARACTER_ASSIGNER},
        ) as resumed:
            self.assertEqual(resumed, target_run_id)

    def test_empty_options_match_a_configuration_recorded_without_them(self):
        recorded = self.configuration()
        with_empty_options = self.configuration()
        with_empty_options["transcription"]["vad_options"] = {}

        # Empty option dicts are defaults, so they match an absent field.
        self.assertEqual(
            PipelineConfig.from_dict(recorded).to_dict(exclude_defaults=True),
            PipelineConfig.from_dict(with_empty_options).to_dict(exclude_defaults=True),
        )

        with self.start_run(
            "source",
            recorded,
            stages={Stage.TRANSCRIPTION},
        ) as source_run_id:
            with self.tracker.episode_run("1"):
                self.tracker.log_stage(
                    self.episode(),
                    Stage.TRANSCRIPTION,
                    {"word_count": 2},
                )

        derived = self.configuration(character_assigner_threshold=0.9)
        derived["transcription"]["vad_options"] = {}
        forked = self.start_fork(
            source_run_id,
            "forked",
            derived,
            copy_until=Stage.TRANSCRIPTION,
        )
        self.assertNotEqual(forked, source_run_id)

    def test_fork_run_rejects_changed_copied_configuration(self):
        source_configuration = self.configuration()
        with self.start_run(
            "source",
            source_configuration,
            stages={Stage.TRANSCRIPTION},
        ) as source_run_id:
            pass

        with self.assertRaisesRegex(ValueError, "transcription"):
            self.start_fork(
                source_run_id,
                "invalid",
                self.configuration(transcription_model="transcription-b"),
                copy_until=Stage.TRANSCRIPTION,
            )

    def test_parent_averages_and_episode_metric_query_use_latest_children(self):
        configuration = self.configuration()
        with self.start_run(
            "source",
            configuration,
            stages={Stage.TRANSCRIPTION},
        ) as parent_run_id:
            for key, count in (("1", 2), ("2", 4)):
                with self.tracker.episode_run(key):
                    self.tracker.log_stage(
                        self.episode(),
                        Stage.TRANSCRIPTION,
                        {"word_count": count},
                    )
            self.tracker.log_parent_averages(parent_run_id)

        parent = self.tracker.client.get_run(parent_run_id)
        self.assertEqual(
            parent.data.metrics["mean.transcription.word_count"],
            3,
        )
        self.assertEqual(
            self.tracker.episode_metrics(
                "source",
                "transcription.word_count",
            ),
            {"1": 2, "2": 4},
        )


if __name__ == "__main__":
    unittest.main()

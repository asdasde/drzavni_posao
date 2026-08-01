import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drzavni_posao.character_assigner import CharacterAssignerConfig
from drzavni_posao.diarizer import DiarizerConfig
from drzavni_posao.embedder import EmbedderConfig
from drzavni_posao.pipeline import PipelineConfig, run_pipeline
from drzavni_posao.tracking import Stage, configuration_id
from drzavni_posao.transcriber import TranscriberConfig


class PipelineConfigTests(unittest.TestCase):
    @staticmethod
    def config(*, diarization=True, assignment=True):
        media = SimpleNamespace(trim_start=25, trim_end=7)
        return PipelineConfig.from_media(
            media,
            transcription=TranscriberConfig(
                model_name="/models/whisper-sr-ct2",
                language="sr",
                align_model_name="classla/align",
                asr_options={"no_repeat_ngram_size": 5},
            ),
            diarization=(
                DiarizerConfig(hf_token="diar-secret")
                if diarization
                else None
            ),
            character_assigner=(
                CharacterAssignerConfig(
                    min_similarity=0.1,
                    embedder=EmbedderConfig(hf_token="embed-secret"),
                    speaker_samples={
                        "Djordje": [
                            {
                                "episode_key": "15",
                                "start": 99,
                                "end": 108,
                            }
                        ]
                    },
                )
                if assignment
                else None
            ),
        )

    def test_tracking_dict_contains_behavior_but_omits_secrets(self):
        configuration = self.config().to_tracking_dict()
        serialized = json.dumps(configuration)

        self.assertEqual(
            configuration["audio"],
            {"trim_start": 25, "trim_end": 7},
        )
        self.assertEqual(
            configuration["character_assigner"]["speaker_samples"],
            {
                "Djordje": [
                    {
                        "episode_key": "15",
                        "start": 99,
                        "end": 108,
                    }
                ]
            },
        )
        self.assertNotIn("diar-secret", serialized)
        self.assertNotIn("embed-secret", serialized)

    def test_stage_order_is_inferred_from_present_configs(self):
        full = self.config()
        direct_assignment = self.config(diarization=False)

        self.assertEqual(
            full.stages,
            (
                Stage.TRANSCRIPTION,
                Stage.DIARIZATION,
                Stage.CHARACTER_ASSIGNER,
            ),
        )
        self.assertEqual(
            direct_assignment.stages,
            (Stage.TRANSCRIPTION, Stage.CHARACTER_ASSIGNER),
        )

    def test_generated_name_uses_model_stages_and_config_hash(self):
        config = self.config()
        config_id = configuration_id(config.to_tracking_dict())

        self.assertEqual(
            config.generated_name(),
            "transcription-whisper-sr-ct2-diarization-character-assigner-"
            f"{config_id[:8]}",
        )


class PipelineExecutionTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def active_run(run_id=None):
        yield run_id

    @staticmethod
    def media():
        episode = SimpleNamespace(clear_audio=Mock())
        media = SimpleNamespace(
            trim_start=25,
            trim_end=7,
            episode=Mock(return_value=episode),
            save=Mock(),
        )
        return media, episode

    def tracker(self, pending, state=None):
        tracker = Mock()
        tracker.pipeline_run.return_value = self.active_run("parent")
        tracker.episode_run.side_effect = lambda _: self.active_run()
        tracker.episode_run_states.return_value = (
            {"1": state} if state else {}
        )
        tracker.pending_stages.return_value = pending
        return tracker

    def test_runs_only_through_requested_stage(self):
        config = PipelineConfigTests.config()
        media, episode = self.media()
        tracker = self.tracker((Stage.TRANSCRIPTION,))

        with (
            patch("drzavni_posao.pipeline.Transcriber") as transcriber_type,
            patch("drzavni_posao.pipeline.Diarizer") as diarizer_type,
            patch(
                "drzavni_posao.pipeline.CharacterAssigner"
            ) as assigner_type,
            patch(
                "drzavni_posao.pipeline.alignment_metrics",
                return_value={"aligned": 1},
            ),
            patch(
                "drzavni_posao.pipeline.repetition_metrics",
                return_value={"repeated": 2},
            ),
        ):
            transcriber_type.return_value.run.return_value = episode
            run_id = run_pipeline(
                tracker,
                media,
                config,
                {"1"},
                through=Stage.TRANSCRIPTION,
                tags={"kind": "test"},
            )

        config_id = configuration_id(config.to_tracking_dict())
        name = config.generated_name()
        self.assertEqual(run_id, "parent")
        tracker.pipeline_run.assert_called_once_with(
            name,
            config.to_tracking_dict(),
            stages={Stage.TRANSCRIPTION},
            tags={"kind": "test"},
        )
        transcriber_type.assert_called_once_with(config.transcription)
        diarizer_type.assert_not_called()
        assigner_type.assert_not_called()
        tracker.log_stage.assert_called_once_with(
            episode,
            Stage.TRANSCRIPTION,
            {"aligned": 1, "repeated": 2},
        )
        media.save.assert_called_once_with(episode, subdir=name)
        episode.clear_audio.assert_called_once()
        transcriber_type.return_value.free.assert_called_once()
        tracker.log_parent_averages.assert_called_once_with("parent")

    def test_restores_transcription_before_direct_character_assignment(self):
        config = PipelineConfigTests.config(diarization=False)
        media, episode = self.media()
        state = SimpleNamespace(
            run_id="episode-run",
            completed_stages=frozenset({Stage.TRANSCRIPTION}),
        )
        tracker = self.tracker((Stage.CHARACTER_ASSIGNER,), state)

        with (
            patch("drzavni_posao.pipeline.Transcriber") as transcriber_type,
            patch(
                "drzavni_posao.pipeline.CharacterAssigner"
            ) as assigner_type,
            patch(
                "drzavni_posao.pipeline.assignment_metrics",
                return_value={"named": 1},
            ),
        ):
            assigner_type.return_value.run.return_value = episode
            run_pipeline(
                tracker,
                media,
                config,
                {"1"},
                through=Stage.CHARACTER_ASSIGNER,
            )

        transcriber_type.assert_not_called()
        assigner_type.assert_called_once_with(
            config.character_assigner,
            media,
        )
        tracker.restore_stage.assert_called_once_with(
            episode,
            "episode-run",
            Stage.TRANSCRIPTION,
        )
        tracker.log_stage.assert_called_once_with(
            episode,
            Stage.CHARACTER_ASSIGNER,
            {"named": 1},
        )
        assigner_type.return_value.free.assert_called_once()


if __name__ == "__main__":
    unittest.main()

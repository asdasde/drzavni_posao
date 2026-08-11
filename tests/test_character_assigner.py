import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from drzavni_posao.character_assigner import CharacterAssigner


class CharacterAssignerTests(unittest.TestCase):
    def test_speaker_labels_select_cluster_assignment_after_restore(self):
        assigner = CharacterAssigner.__new__(CharacterAssigner)
        assigner._assign_per_cluster = Mock()
        assigner._assign_per_segment = Mock()
        episode = SimpleNamespace(
            metadata=SimpleNamespace(episode_number=1),
            transcript=SimpleNamespace(
                segments=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                    }
                ]
            ),
            diarization=None,
            audio=Mock(return_value=object()),
        )

        assigner.run(episode)

        assigner._assign_per_cluster.assert_called_once()
        assigner._assign_per_segment.assert_not_called()


if __name__ == "__main__":
    unittest.main()

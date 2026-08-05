import unittest

from drzavni_posao.metrics import (
    UNCOVERED_LABEL,
    assignment_scores,
    character_table,
    confusion_table,
    labeled_segments,
)


class AssignmentScoresTests(unittest.TestCase):
    def test_split_predictions_are_duration_weighted(self):
        segments = labeled_segments(
            [{"start": 25, "end": 35, "speaker": "Djordje"}],
            [
                {"start": 0, "end": 4, "speaker": "Djordje"},
                {"start": 4, "end": 10, "speaker": "Djordje"},
            ],
            trim_start=25,
        )
        scores = assignment_scores(segments)

        self.assertEqual(scores["labeled_seconds"], 10)
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(scores["coverage"], 1)

    def test_wrong_and_uncovered_seconds_both_count_against_accuracy(self):
        segments = labeled_segments(
            [{"start": 0, "end": 10, "speaker": "Djordje"}],
            [{"start": 0, "end": 5, "speaker": "Boskic"}],
        )
        scores = assignment_scores(segments)

        self.assertEqual(scores["accuracy"], 0)
        self.assertEqual(scores["coverage"], 0.5)
        self.assertEqual(scores["covered_accuracy"], 0)
        self.assertEqual(
            confusion_table(segments).loc["Djordje"].to_dict(),
            {"Boskic": 5, "Djordje": 0, UNCOVERED_LABEL: 5},
        )

    def test_covered_accuracy_omitted_when_nothing_is_covered(self):
        segments = labeled_segments(
            [{"start": 0, "end": 10, "speaker": "Djordje"}],
            [],
        )
        scores = assignment_scores(segments)

        self.assertEqual(scores["coverage"], 0)
        self.assertNotIn("covered_accuracy", scores)

    def test_per_character_precision_recall_and_f1(self):
        segments = labeled_segments(
            [
                {"start": 0, "end": 10, "speaker": "Djordje"},
                {"start": 10, "end": 20, "speaker": "Boskic"},
            ],
            [
                {"start": 0, "end": 15, "speaker": "Djordje"},
                {"start": 15, "end": 20, "speaker": "Boskic"},
            ],
        )
        characters = character_table(segments)
        scores = assignment_scores(segments)

        self.assertAlmostEqual(characters.at["Djordje", "precision"], 10 / 15)
        self.assertAlmostEqual(characters.at["Djordje", "recall"], 1.0)
        self.assertAlmostEqual(characters.at["Boskic", "precision"], 1.0)
        self.assertAlmostEqual(characters.at["Boskic", "recall"], 0.5)
        self.assertAlmostEqual(scores["accuracy"], 0.75)
        support = characters["labeled_seconds"]
        self.assertAlmostEqual(
            scores["weighted_f1"],
            (characters["f1"] * support).sum() / support.sum(),
        )

    def test_overlapping_predictions_cannot_exceed_labeled_duration(self):
        segments = labeled_segments(
            [{"start": 0, "end": 10, "speaker": "Djordje"}],
            [
                {"start": 0, "end": 8, "speaker": "Djordje"},
                {"start": 5, "end": 12, "speaker": "Boskic"},
            ],
        )
        scores = assignment_scores(segments)

        self.assertEqual(
            confusion_table(segments).loc["Djordje"].to_dict(),
            {"Boskic": 2, "Djordje": 8},
        )
        self.assertEqual(scores["accuracy"], 0.8)
        self.assertEqual(scores["coverage"], 1)


if __name__ == "__main__":
    unittest.main()

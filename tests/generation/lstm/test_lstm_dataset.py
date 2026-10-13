import unittest

import torch

from drzavni_posao.generation.lstm.dataset import (
    ROLES,
    CharVocabulary,
    TurnWindowDataset,
    collate_windows,
    roles_for,
)


EPISODE = {
    "premise": "Kafa.",
    "turns": [
        {"speaker": "Boskic", "text": "Ima?"},
        {"speaker": "Djordje Cvarkov", "text": "Nema."},
        {"speaker": "Dragan Torbica", "text": "Ajde."},
    ],
}


def rendered(dataset, index):
    example = dataset.example(index)
    return dataset.vocab.decode(example["input_ids"] + example["target_ids"][-1:])


def target_text(dataset, index):
    example = dataset.example(index)
    ids = [i for i, m in zip(example["target_ids"], example["loss_mask"]) if m]
    return dataset.vocab.decode(ids)


class TurnWindowDatasetTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocabulary()

    def test_one_example_per_turn(self):
        dataset = TurnWindowDataset([EPISODE, EPISODE], self.vocab, context_turns=1)

        self.assertEqual(len(dataset), 6)

    def test_window_holds_previous_turns_and_start_token(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=3)

        self.assertEqual(
            rendered(dataset, 2),
            "<start>[Boskic] Ima?\n[Djordje Cvarkov] Nema.\n[Dragan Torbica] Ajde.\n",
        )

    def test_start_token_only_when_window_reaches_episode_start(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=1)

        self.assertEqual(rendered(dataset, 0), "<start>[Boskic] Ima?\n")
        self.assertEqual(
            rendered(dataset, 2), "[Djordje Cvarkov] Nema.\n[Dragan Torbica] Ajde.\n"
        )

    def test_loss_covers_only_target_text_and_newline(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=2)

        self.assertEqual(target_text(dataset, 2), "Ajde.\n")

    def test_target_tag_is_input_not_prediction(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=0)
        example = dataset.example(1)

        self.assertEqual(example["input_ids"][0], self.vocab.ids["[Djordje Cvarkov]"])
        self.assertTrue(example["loss_mask"][0])

    def test_inputs_and_targets_are_shifted_by_one(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=2)
        example = dataset.example(2)

        self.assertEqual(example["input_ids"][1:], example["target_ids"][:-1])
        self.assertEqual(len(example["roles"]), len(example["input_ids"]))

    def test_context_is_truncated_from_the_left_at_turn_boundaries(self):
        dataset = TurnWindowDataset(
            [EPISODE], self.vocab, context_turns=2, max_context_chars=8
        )

        self.assertEqual(
            rendered(dataset, 2), "[Djordje Cvarkov] Nema.\n[Dragan Torbica] Ajde.\n"
        )

    def test_prefix_premise_comes_first(self):
        dataset = TurnWindowDataset(
            [EPISODE], self.vocab, context_turns=0, premise_mode="prefix"
        )

        self.assertEqual(rendered(dataset, 0), "<premise>Kafa.\n<start>[Boskic] Ima?\n")
        self.assertEqual(target_text(dataset, 0), "Ima?\n")

    def test_encoder_premise_is_returned_separately(self):
        dataset = TurnWindowDataset(
            [EPISODE], self.vocab, context_turns=0, premise_mode="encoder"
        )

        self.assertEqual(self.vocab.decode(dataset.example(0)["premise_ids"]), "Kafa.")
        self.assertEqual(rendered(dataset, 0), "<start>[Boskic] Ima?\n")

    def test_roles_follow_the_speaker_of_each_token(self):
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=1)
        roles = dataset.example(1)["roles"]

        self.assertEqual(
            roles,
            [ROLES.index("<start>")]
            + [ROLES.index("Boskic")] * len("TIma?\n")
            + [ROLES.index("Djordje Cvarkov")] * len("TNema."),
        )

    def test_roles_can_be_recovered_from_token_ids(self):
        for premise_mode in ("none", "prefix"):
            dataset = TurnWindowDataset(
                [EPISODE], self.vocab, context_turns=2, premise_mode=premise_mode
            )
            for index in range(len(dataset)):
                with self.subTest(premise_mode=premise_mode, index=index):
                    example = dataset.example(index)

                    self.assertEqual(
                        roles_for(example["input_ids"], self.vocab), example["roles"]
                    )

    def test_rejects_unknown_premise_mode(self):
        with self.assertRaisesRegex(ValueError, "premise_mode"):
            TurnWindowDataset([EPISODE], self.vocab, context_turns=0, premise_mode="x")


class CollateTests(unittest.TestCase):
    def test_pads_batch_and_reports_lengths(self):
        vocab = CharVocabulary()
        dataset = TurnWindowDataset(
            [EPISODE], vocab, context_turns=2, premise_mode="encoder"
        )
        batch = collate_windows([dataset[0], dataset[2]])

        lengths = batch["lengths"].tolist()
        self.assertEqual(batch["input_ids"].shape, (2, max(lengths)))
        self.assertFalse(batch["loss_mask"][0, lengths[0] :].any())
        self.assertTrue(
            torch.equal(batch["premise_lengths"], torch.tensor([5, 5]))
        )


if __name__ == "__main__":
    unittest.main()

import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from drzavni_posao.generation.lstm.dataset import (
    ROLES,
    CharVocabulary,
    TurnWindowDataset,
    collate_windows,
)
from drzavni_posao.generation.lstm.model import NeuralLM, build_network
from drzavni_posao.generation.lstm.train import (
    TrainConfig,
    length_batches,
    load_checkpoint,
    save_checkpoint,
    train,
)


EPISODE = {
    "premise": "Kafa.",
    "turns": [
        {"speaker": "Boskic", "text": "Ima kafe?"},
        {"speaker": "Djordje Cvarkov", "text": "Nema."},
        {"speaker": "Dragan Torbica", "text": "Ajde."},
    ],
}

VARIANTS = {
    "lstm": {},
    "gru": {"cell": "gru"},
    "rnn": {"cell": "rnn"},
    "residual": {"residual": True},
    "attention": {"attention_heads": 2},
    "tied": {"tie_weights": True},
    "locked": {"locked_dropout": True},
    "transformer": {"cell": "transformer", "attention_heads": 2},
}


def tiny_network(vocab, **overrides):
    torch.manual_seed(0)
    options = dict(
        cell="lstm",
        embedding_dim=8,
        role_dim=4,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        residual=False,
        attention_heads=0,
        tie_weights=False,
        locked_dropout=False,
    )
    return build_network(len(vocab), len(ROLES), **(options | overrides))


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocabulary()
        dataset = TurnWindowDataset([EPISODE], self.vocab, context_turns=2)
        self.batch = collate_windows([dataset[i] for i in range(len(dataset))])

    def test_logits_cover_every_position(self):
        for name, overrides in VARIANTS.items():
            with self.subTest(variant=name):
                network = tiny_network(self.vocab, **overrides)
                logits, _ = network(self.batch["input_ids"], self.batch["roles"])

                self.assertEqual(
                    logits.shape, (*self.batch["input_ids"].shape, len(self.vocab))
                )

    def test_right_padding_does_not_change_real_positions(self):
        for name, overrides in VARIANTS.items():
            with self.subTest(variant=name):
                network = tiny_network(self.vocab, **overrides).eval()
                padded, _ = network(self.batch["input_ids"], self.batch["roles"])
                for row, length in enumerate(self.batch["lengths"].tolist()):
                    alone, _ = network(
                        self.batch["input_ids"][row : row + 1, :length],
                        self.batch["roles"][row : row + 1, :length],
                    )
                    torch.testing.assert_close(
                        padded[row, :length], alone[0], rtol=1e-4, atol=1e-5
                    )

    def test_tied_output_shares_the_token_embedding(self):
        network = tiny_network(self.vocab, tie_weights=True)

        self.assertIs(network.output.weight, network.token_embedding.weight)

    def test_locked_dropout_drops_the_same_features_at_every_step(self):
        network = tiny_network(self.vocab, locked_dropout=True, dropout=0.5).train()
        dropped = network.dropout(torch.ones(2, 7, 16)) == 0

        self.assertTrue((dropped == dropped[:, :1]).all())


class NeuralLMTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocabulary()
        self.history = [self.vocab.ids["[Boskic]"], *self.vocab.encode("Ima")]

    def test_stepping_matches_a_fresh_pass(self):
        for name in ("lstm", "attention", "transformer"):
            with self.subTest(variant=name):
                network = tiny_network(self.vocab, **VARIANTS[name])
                stepping = NeuralLM(network, self.vocab)
                for end in range(1, len(self.history)):
                    stepping.next_token_probability(self.history[:end])

                np.testing.assert_allclose(
                    stepping.next_token_probability(self.history),
                    NeuralLM(network, self.vocab).next_token_probability(self.history),
                    rtol=1e-5,
                )

    def test_a_different_history_starts_over(self):
        network = tiny_network(self.vocab)
        model = NeuralLM(network, self.vocab)
        model.next_token_probability(self.history)
        other = [self.vocab.ids["[Dragan Torbica]"], *self.vocab.encode("Ne")]

        np.testing.assert_allclose(
            model.next_token_probability(other),
            NeuralLM(network, self.vocab).next_token_probability(other),
            rtol=1e-6,
        )

    def test_distribution_sums_to_one(self):
        model = NeuralLM(tiny_network(self.vocab), self.vocab)

        self.assertAlmostEqual(model.next_token_probability(self.history).sum(), 1.0)


class LengthBatchesTests(unittest.TestCase):
    def test_every_example_once_in_batches_of_similar_length(self):
        lengths = [random.Random(i).randint(1, 1000) for i in range(1000)]

        batches = length_batches(lengths, 10, random.Random(0), pool=100)

        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(1000)))
        self.assertTrue(all(len(batch) == 10 for batch in batches))
        spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
        self.assertLess(spread, 50)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_keeps_predictions(self):
        episodes = [{**EPISODE, "split": split} for split in ("train", "validation", "test")]
        for bpe_vocab_size in (None, 120):
            with self.subTest(bpe_vocab_size=bpe_vocab_size):
                config = TrainConfig(
                    embedding_dim=8,
                    role_dim=4,
                    hidden_size=16,
                    num_layers=1,
                    context_turns=1,
                    epochs=1,
                    bpe_vocab_size=bpe_vocab_size,
                )
                model, _ = train(config, episodes)
                history = [model.vocab.ids["[Boskic]"], *model.vocab.encode("Ima")]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    save_checkpoint(model, config, path)
                    loaded, loaded_config = load_checkpoint(path)

                self.assertEqual(loaded_config, config)
                self.assertEqual(loaded.vocab.tokens, model.vocab.tokens)
                np.testing.assert_allclose(
                    loaded.next_token_probability(history),
                    model.next_token_probability(history),
                    rtol=1e-6,
                )


class TrainTests(unittest.TestCase):
    def test_learns_a_repeated_episode(self):
        episodes = [{**EPISODE, "split": split} for split in ("train", "validation", "test")]
        config = TrainConfig(
            embedding_dim=8,
            role_dim=4,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            context_turns=2,
            batch_size=3,
            learning_rate=0.02,
            epochs=60,
            patience=60,
        )
        epochs = []

        model, results = train(config, episodes * 4, on_epoch=lambda e, m: epochs.append(m))

        self.assertLess(results["validation_bpc"], 0.5 * math.log2(len(model.vocab)))
        self.assertLess(epochs[-1]["train_bpc"], epochs[0]["train_bpc"])
        self.assertEqual(set(results), {"validation_bpc", "test_bpc", "epochs", "parameters"})


if __name__ == "__main__":
    unittest.main()

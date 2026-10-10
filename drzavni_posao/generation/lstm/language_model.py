"""Sampling shared by every model that gives a next-token distribution."""

from typing import Any
from collections.abc import Sequence

import numpy as np


class LanguageModel:
    def next_token_probability(self, history: Sequence[int]) -> np.ndarray:
        """p(token | history) for every token in the vocabulary."""
        raise NotImplementedError

    def generate(
        self,
        prompt: Sequence[int],
        stop: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
        max_tokens: int = 300,
    ) -> list[int]:
        """Tokens sampled after *prompt*, ending with *stop* unless *max_tokens* runs out."""
        ids = list(prompt)
        for _ in range(max_tokens):
            probabilities = self.next_token_probability(ids) ** (1 / temperature)
            next_token_id = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
            ids.append(next_token_id)
            if next_token_id == stop:
                break
        return ids[len(prompt) :]

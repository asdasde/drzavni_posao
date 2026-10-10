"""Model-agnostic language-model metrics."""

import math


def bits_per_character(total_nll: float, characters: int) -> float:
    """Total negative log-likelihood in nats, spread over the scored characters.

    Normalising by characters rather than tokens keeps BPE models comparable.
    """
    return total_nll / characters / math.log(2)


def perplexity(bpc: float) -> float:
    """Per-character perplexity."""
    return 2**bpc

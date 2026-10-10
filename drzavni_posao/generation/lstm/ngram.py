"""Interpolated Kneser-Ney n-gram baseline, optionally mixed with per-speaker models."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...transcription.constants import CORE_CHARACTERS
from .bpe import build_vocabulary
from .dataset import Vocabulary, encode_turn
from .language_model import LanguageModel
from .metrics import bits_per_character, scored_characters
from .prepare import group_by_split, read_jsonl


MIXTURE_WEIGHTS = tuple(np.round(np.linspace(0, 1, 11), 1))


def max_order(vocab_size: int) -> int:
    """n-grams are encoded as base-|V| integers, which must fit in int64."""
    order = 1
    while vocab_size ** (order + 1) <= 2**63:
        order += 1
    return order


@dataclass(frozen=True, slots=True)
class Level:
    """One order's n-gram counts, plus per-history totals and distinct followers."""

    grams: np.ndarray
    counts: np.ndarray
    histories: np.ndarray
    totals: np.ndarray
    types: np.ndarray
    discount: float

    @classmethod
    def build(cls, grams: np.ndarray, counts: np.ndarray, vocab_size: int) -> "Level":
        prefixes, prefix_starts, prefix_continuation_counts = np.unique(
            grams // vocab_size, return_index=True, return_counts=True
        )
        
        #prefix counts how many times each prefix appears in the original text not how many times it appears in the grams // vocab_size
        #not to be confused with prefix_continuation_counts which counts how many times each prefix appears in the grams which means how many different symbolys follow each prefix
        #first appreance of each prefix is important since grams are sorted so all continuations of a prefix are together and start at the index of first appreance of each prefix
        prefix_counts_manual = np.zeros(len(prefixes), dtype=np.int64)
        for i, (_, prefix_start, prefix_continuation_count) in enumerate(zip(prefixes, prefix_starts, prefix_continuation_counts)):
            prefix_counts_manual[i] = (np.sum(counts[prefix_start:prefix_start + prefix_continuation_count]))
        
        n1 = np.count_nonzero(counts == 1)
        n2 = np.count_nonzero(counts == 2)
        return cls(
            grams=grams,
            counts=counts,
            histories=prefixes,
            totals=prefix_counts_manual,
            types= prefix_continuation_counts,
            discount=n1 / (n1 + 2 * n2) if n1 and n2 else 0.5,
        )

    def interpolate(self, grams: np.ndarray, lower: np.ndarray, vocab_size: int) -> np.ndarray:
        count = _lookup(self.grams, self.counts, grams)
        histories = grams // vocab_size
        total = _lookup(self.histories, self.totals, histories)
        types = _lookup(self.histories, self.types, histories)
        smoothed = np.maximum(count - self.discount, 0) + self.discount * types * lower
        return np.where(total > 0, smoothed / np.maximum(total, 1), lower)


def _lookup(keys: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    index = np.searchsorted(keys, queries).clip(max=len(keys) - 1)
    return np.where(keys[index] == queries, values[index], 0)


class KneserNeyLM(LanguageModel):
    """The top order uses raw counts, lower orders continuation counts.

    Every sequence is left-padded with token 0, which marks the sequence start.
    """

    def __init__(self, order: int, vocab_size: int) -> None:
        if not 1 <= order <= max_order(vocab_size):
            raise ValueError(f"order must be between 1 and {max_order(vocab_size)}")
        self.order = order
        self.vocab_size = vocab_size
        self._powers = vocab_size ** np.arange(order - 1, -1, -1, dtype=np.int64)
        self._levels: list[Level] = []

    def fit(self, sequences: Sequence[Sequence[int]]) -> "KneserNeyLM":
        ngrams = [self.padded_windows(s) for s in sequences] # padded with zeros to the left so each window has order elements represents ngrams
        encoded_ngrams = [n @ self._powers for n in ngrams] # encoded ngrams in base V
        codes = np.concatenate(encoded_ngrams) # all n grams in all turns
        unique_ngrams, unique_counts = np.unique(codes, return_counts=True) # unique ngram and their counts sorted in increasing ngram
        levels = [Level.build(unique_ngrams, unique_counts, self.vocab_size)]
        for k in range(self.order - 1, 0, -1):
            suffixes = levels[0].grams % self.vocab_size**k
            levels.insert(
                0, Level.build(*np.unique(suffixes, return_counts=True), self.vocab_size)
            )
        self._levels = levels
        return self

    def log_probs(
        self, sequences: Sequence[Sequence[int]], masks: Sequence[Sequence[bool]]
    ) -> np.ndarray:
        """Natural-log p(token | history) at every masked position."""
        windows = np.concatenate(
            [self.padded_windows(s)[np.asarray(m, dtype=bool)] for s, m in zip(sequences, masks)]
        )
        probabilities = np.full(len(windows), 1 / self.vocab_size) # vector of probabilities for last token in each window, uniform at the start
       # print(probabilities)
        for k, level in enumerate(self._levels, start=1):
            grams = windows[:, -k:] @ self._powers[-k:]
            probabilities = level.interpolate(grams, probabilities, self.vocab_size)
        return np.log(probabilities)

    def next_token_probability(self, history: Sequence[int]) -> np.ndarray:
        context = list(history[max(0, len(history) - self.order + 1) :])
        options = [[*context, token] for token in range(self.vocab_size)]
        masks = [[False] * len(context) + [True]] * self.vocab_size
        return np.exp(self.log_probs(options, masks))

    def padded_windows(self, sequence: Sequence[int]) -> np.ndarray:
        """One row per token: the token and the order - 1 tokens before it."""
        padded = np.concatenate(
            [np.zeros(self.order - 1, dtype=np.int64), np.asarray(sequence, np.int64)]
        )
        return sliding_window_view(padded, self.order)


class SpeakerMixtureLM(LanguageModel):
    """A model over all turns mixed with one model per speaker.

    Every model reads one turn at a time, so a turn's history starts at its speaker
    tag. The global model forgets the speaker a few symbols in; each speaker model
    is trained on that speaker's turns alone, so it never does. *weight* is the
    speaker model's share of the mixture.
    """

    def __init__(self, order: int, vocab: Vocabulary, weight: float = 0.5) -> None:
        self.vocab = vocab
        self.weight = weight
        self.global_model = KneserNeyLM(order, len(vocab))
        self.speaker_models = {
            speaker: KneserNeyLM(order, len(vocab)) for speaker in CORE_CHARACTERS
        }
        self._tag_speakers = {vocab.ids[f"[{s}]"]: s for s in CORE_CHARACTERS}

    def fit(self, episodes: Sequence[dict]) -> "SpeakerMixtureLM":
        speakers, turns = self._encode(episodes)
        ids = [turn_ids for turn_ids, _ in turns]
        self.global_model.fit(ids)
        for speaker, model in self.speaker_models.items():
            model.fit([turn_ids for owner, turn_ids in zip(speakers, ids) if owner == speaker])
        return self

    def component_probs(self, episodes: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Global-model and speaker-model probabilities at every scored position."""
        speakers, turns = self._encode(episodes)
        global_probs = np.exp(self.global_model.log_probs(*zip(*turns)))
        owners = np.repeat(speakers, [sum(scored) for _, scored in turns])
        speaker_probs = np.empty_like(global_probs)
        for speaker, model in self.speaker_models.items():
            own = [pair for owner, pair in zip(speakers, turns) if owner == speaker]
            if own:
                speaker_probs[owners == speaker] = np.exp(model.log_probs(*zip(*own)))
        return global_probs, speaker_probs

    def next_token_probability(self, history: Sequence[int]) -> np.ndarray:
        """Only the current turn counts: the tokens after the last newline, tag first."""
        newline = self.vocab.ids["\n"]
        turn = list(history)
        if newline in turn:
            turn = turn[len(turn) - turn[::-1].index(newline) :]
        speaker_id = self._tag_speakers[turn[0]]
        speaker_model = self.speaker_models[speaker_id]
        speaker_component = speaker_model.next_token_probability(turn)
        global_component = self.global_model.next_token_probability(turn)
        return self.weight * speaker_component + (1 - self.weight) * global_component

    def _encode(self, episodes: Sequence[dict]) -> tuple[list[str], list]:
        """Every turn's speaker and encoding."""
        turns = [turn for e in episodes for turn in e["turns"]]
        return [t["speaker"] for t in turns], [encode_turn(t, self.vocab) for t in turns]


def mixture_bpc(
    components: tuple[np.ndarray, np.ndarray], weight: float, characters: int
) -> float:
    global_probs, speaker_probs = components
    probs = weight * speaker_probs + (1 - weight) * global_probs
    return bits_per_character(-np.log(probs).sum(), characters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/lstm/episodes.jsonl"))
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument(
        "--bpe-vocab-size", type=int, help="Model BPE tokens instead of characters."
    )
    parser.add_argument("--samples", type=int, default=0, help="Turns to sample per speaker.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    by_split = group_by_split(read_jsonl(args.episodes))
    vocab = build_vocabulary(by_split["train"], args.bpe_vocab_size)
    characters = {split: scored_characters(group) for split, group in by_split.items()}
    rng = np.random.default_rng(0)

    print("order   global_val  global_test  weight  mixture_val  mixture_test")
    for order in args.orders:
        model = SpeakerMixtureLM(order, vocab, weight=1).fit(by_split["train"])
        components = {
            split: model.component_probs(by_split[split]) for split in ("validation", "test")
        }

        def bpc(split: str, weight: float) -> float:
            return mixture_bpc(components[split], weight, characters[split])

        weight = min(MIXTURE_WEIGHTS, key=lambda w: bpc("validation", w))
        print(
            f"{order:5d}  {bpc('validation', 0):11.4f}  {bpc('test', 0):12.4f}"
            f"  {weight:6.1f}  {bpc('validation', weight):11.4f}  {bpc('test', weight):12.4f}"
        )
        model.weight = weight
        for speaker in CORE_CHARACTERS:
            tag = f"[{speaker}]"
            for _ in range(args.samples):
                ids = model.generate([vocab.ids[tag]], vocab.ids["\n"], rng, args.temperature)
                print(f"  {tag} {vocab.decode(ids).strip()}")


if __name__ == "__main__":
    main()

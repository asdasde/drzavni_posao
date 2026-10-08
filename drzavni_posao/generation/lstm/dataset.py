"""Rolling-window examples: premise, previous turns, and one target turn."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ...transcription.constants import CORE_CHARACTERS
from .text import ALPHABET


PAD = "<pad>"
START = "<start>"
PREMISE = "<premise>"
SPEAKER_TAGS = tuple(f"[{speaker}]" for speaker in CORE_CHARACTERS)
ROLES = (PAD, PREMISE, START, *CORE_CHARACTERS)
PREMISE_MODES = ("none", "prefix", "encoder")

_ROLE_IDS = {role: index for index, role in enumerate(ROLES)}


class CharVocabulary:
    """Characters plus atomic speaker tags; id 0 is padding."""

    def __init__(self) -> None:
        self.tokens = (PAD, START, PREMISE, *SPEAKER_TAGS, "\n", *sorted(ALPHABET))
        self.ids = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.ids[character] for character in text]

    def decode(self, ids: Iterable[int]) -> str:
        tokens = (self.tokens[index] for index in ids)
        return "".join(f"{t} " if t in SPEAKER_TAGS else t for t in tokens)


@dataclass(frozen=True, slots=True)
class _Turn:
    role: int
    tag: int
    text: list[int]


class TurnWindowDataset(Dataset):
    """One example per turn: the turn is the target, earlier turns the context.

    Loss covers only the target text and its closing newline. The context keeps
    at most *context_turns* turns, dropping the oldest while it exceeds
    *max_context_chars*. A start token marks windows that reach the episode start.
    """

    def __init__(
        self,
        episodes: Sequence[dict],
        vocab: CharVocabulary,
        *,
        context_turns: int,
        max_context_chars: int = 1000,
        premise_mode: str = "none",
    ) -> None:
        if premise_mode not in PREMISE_MODES:
            raise ValueError(f"premise_mode must be one of {PREMISE_MODES}")
        if context_turns < 0:
            raise ValueError("context_turns cannot be negative")
        self.vocab = vocab
        self.context_turns = context_turns
        self.max_context_chars = max_context_chars
        self.premise_mode = premise_mode
        self._episodes = [
            (
                vocab.encode(episode["premise"]),
                [
                    _Turn(
                        role=_ROLE_IDS[turn["speaker"]],
                        tag=vocab.ids[f"[{turn['speaker']}]"],
                        text=vocab.encode(turn["text"]),
                    )
                    for turn in episode["turns"]
                ],
            )
            for episode in episodes
        ]
        self._index = [
            (episode_index, turn_index)
            for episode_index, (_, turns) in enumerate(self._episodes)
            for turn_index in range(len(turns))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value) for key, value in self.example(index).items()}

    def example(self, index: int) -> dict[str, list]:
        episode_index, target_index = self._index[index]
        premise, turns = self._episodes[episode_index]
        start = self._context_start(turns, target_index)
        newline = self.vocab.ids["\n"]

        tokens: list[int] = []
        roles: list[int] = []
        in_loss: list[bool] = []

        def add(ids: list[int], role: int, loss: bool = False) -> None:
            tokens.extend(ids)
            roles.extend([role] * len(ids))
            in_loss.extend([loss] * len(ids))

        if self.premise_mode == "prefix":
            add([self.vocab.ids[PREMISE], *premise, newline], _ROLE_IDS[PREMISE])
        if start == 0:
            add([self.vocab.ids[START]], _ROLE_IDS[START])
        for turn in turns[start:target_index]:
            add([turn.tag, *turn.text, newline], turn.role)
        target = turns[target_index]
        add([target.tag], target.role)
        add([*target.text, newline], target.role, loss=True)

        example = {
            "input_ids": tokens[:-1],
            "target_ids": tokens[1:],
            "roles": roles[:-1],
            "loss_mask": in_loss[1:],
        }
        if self.premise_mode == "encoder":
            example["premise_ids"] = premise
        return example

    def _context_start(self, turns: list[_Turn], target_index: int) -> int:
        start = max(0, target_index - self.context_turns)
        length = sum(len(turn.text) + 2 for turn in turns[start:target_index])
        while start < target_index and length > self.max_context_chars:
            length -= len(turns[start].text) + 2
            start += 1
        return start


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad every field with 0, which is both the pad token and pad role."""
    collated = {
        key: pad_sequence([item[key] for item in batch], batch_first=True)
        for key in batch[0]
    }
    collated["lengths"] = torch.tensor([len(item["input_ids"]) for item in batch])
    if "premise_ids" in batch[0]:
        collated["premise_lengths"] = torch.tensor(
            [len(item["premise_ids"]) for item in batch]
        )
    return collated

"""Byte-pair encoding vocabulary, a drop-in replacement for CharVocabulary."""

from collections.abc import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from .dataset import SPECIAL_TOKENS, CharVocabulary, Vocabulary
from .text import ALPHABET


WORD_START = "▁"


class BpeVocabulary(Vocabulary):
    """Merges are learned on turn text only, so no token spans a tag or a turn."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(map(tokenizer.id_to_token, range(tokenizer.get_vocab_size())))
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BpeVocabulary":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=WORD_START)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=sorted(ALPHABET - {" "}),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.tokens[index] for index in ids).replace(WORD_START, " ")


def build_vocabulary(train_episodes: Iterable[dict], bpe_vocab_size: int | None) -> Vocabulary:
    """Characters, or BPE merges learned on the training turns."""
    if bpe_vocab_size is None:
        return CharVocabulary()
    texts = (turn["text"] for e in train_episodes for turn in e["turns"])
    return BpeVocabulary.train(texts, bpe_vocab_size)

from collections.abc import Iterable
from typing import Any

from datasets import Dataset, DatasetDict

from ..dataset import SPLITS
from .prompt import build_episode_messages


class SFTDatasetBuilder:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        enable_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def prepare_rows(
        self,
        rows: Iterable[dict[str, str]],
    ) -> tuple[
        dict[str, list[dict[str, object]]],
        list[str],
    ]:
        rows_by_split: dict[str, list[dict[str, object]]] = {
            split: [] for split in SPLITS
        }
        overlong_episode_keys = []

        for row in rows:
            split = row["split"]
            if split not in rows_by_split:
                raise ValueError(f"Unknown split {split!r}")

            prompt = build_episode_messages(row["premise"])
            completion = [{"role": "assistant", "content": row["dialogue"]}]
            token_count = self._count_tokens(prompt + completion)
            if token_count > self.max_length:
                overlong_episode_keys.append(row["episode_key"])
                continue

            rows_by_split[split].append(
                {
                    "episode_key": row["episode_key"],
                    "split": split,
                    "premise": row["premise"],
                    "prompt": prompt,
                    "completion": completion,
                    "token_count": token_count,
                }
            )

        if any(not rows_by_split[split] for split in SPLITS):
            raise ValueError(
                "Every split must contain data: "
                f"{ {split: len(rows) for split, rows in rows_by_split.items()} }"
            )

        return rows_by_split, overlong_episode_keys

    def build_dataset(
        self,
        rows_by_split: dict[str, list[dict[str, object]]],
    ) -> DatasetDict:
        return DatasetDict(
            {
                split: Dataset.from_list(
                    [
                        {
                            "prompt": row["prompt"],
                            "completion": row["completion"],
                            "chat_template_kwargs": {
                                "enable_thinking": self.enable_thinking
                            },
                        }
                        for row in split_rows
                    ]
                )
                for split, split_rows in rows_by_split.items()
            }
        )

    def _count_tokens(self, messages: list[dict[str, str]]) -> int:
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            enable_thinking=self.enable_thinking,
        )
        return len(tokenized["input_ids"])

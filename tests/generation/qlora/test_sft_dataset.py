import unittest

from drzavni_posao.generation.qlora.dataset import SFTDatasetBuilder


class FixedLengthTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": messages[-1]["content"].split()}


class SFTDatasetBuilderTests(unittest.TestCase):
    def test_builds_prompt_completion_dataset(self):
        tokenizer = FixedLengthTokenizer()
        builder = SFTDatasetBuilder(
            tokenizer,
            max_length=2,
            enable_thinking=False,
        )

        rows_by_split, overlong = builder.prepare_rows(self.rows())
        dataset = builder.build_dataset(rows_by_split)

        self.assertEqual(list(dataset), ["train", "validation", "test"])
        self.assertEqual(
            dataset["train"].column_names,
            ["prompt", "completion", "chat_template_kwargs"],
        )
        self.assertEqual(
            dataset["train"][0]["completion"],
            [{"role": "assistant", "content": "train dialogue"}],
        )
        self.assertEqual(rows_by_split["train"][0]["token_count"], 2)
        self.assertEqual(overlong, [])
        self.assertTrue(tokenizer.calls)
        self.assertTrue(
            all(call["enable_thinking"] is False for call in tokenizer.calls)
        )

    def test_filters_overlong_rows(self):
        builder = SFTDatasetBuilder(FixedLengthTokenizer(), max_length=2)
        rows = self.rows()
        rows.insert(
            1,
            {
                "episode_key": "4",
                "split": "train",
                "premise": "Too long.",
                "dialogue": "one two three",
            },
        )

        rows_by_split, overlong = builder.prepare_rows(rows)

        self.assertEqual(
            [row["episode_key"] for row in rows_by_split["train"]],
            ["1"],
        )
        self.assertEqual(overlong, ["4"])

    def test_requires_every_split(self):
        builder = SFTDatasetBuilder(FixedLengthTokenizer(), max_length=2)

        with self.assertRaisesRegex(ValueError, "Every split must contain data"):
            builder.prepare_rows(self.rows()[:2])

    @staticmethod
    def rows():
        return [
            {
                "episode_key": "1",
                "split": "train",
                "premise": "Train premise.",
                "dialogue": "train dialogue",
            },
            {
                "episode_key": "2",
                "split": "validation",
                "premise": "Validation premise.",
                "dialogue": "validation dialogue",
            },
            {
                "episode_key": "3",
                "split": "test",
                "premise": "Test premise.",
                "dialogue": "test dialogue",
            },
        ]


if __name__ == "__main__":
    unittest.main()

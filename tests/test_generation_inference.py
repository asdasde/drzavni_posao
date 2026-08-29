import unittest
from contextlib import contextmanager

import torch

from drzavni_posao.generation.inference import generate_episode
from drzavni_posao.generation.prompt import EPISODE_SYSTEM_PROMPT


class GenerationInferenceTests(unittest.TestCase):
    def test_generates_with_adapter_disabled(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()

        result = generate_episode(
            model,
            tokenizer,
            "  Problem u arhivi.  ",
            adapter_enabled=False,
            seed=42,
            max_new_tokens=16,
            temperature=0.7,
        )

        self.assertEqual(result, "generated episode")
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": EPISODE_SYSTEM_PROMPT},
                {"role": "user", "content": "Problem u arhivi."},
            ],
        )
        self.assertTrue(model.adapter_disabled)
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 16)
        self.assertEqual(model.generation_kwargs["temperature"], 0.7)


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {"input_ids": torch.tensor([[10, 11]])}

    def decode(self, token_ids, **kwargs):
        return "  generated episode  "


class FakeModel:
    device = torch.device("cpu")

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        yield

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 11, 20, 21]])


if __name__ == "__main__":
    unittest.main()

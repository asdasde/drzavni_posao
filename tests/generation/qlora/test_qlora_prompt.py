import unittest

from drzavni_posao.generation.qlora.prompt import (
    EPISODE_SYSTEM_PROMPT,
    build_episode_messages,
)


class GenerationPromptTests(unittest.TestCase):
    def test_builds_messages_from_trimmed_premise(self):
        self.assertEqual(
            build_episode_messages("  Problem u arhivi.  "),
            [
                {"role": "system", "content": EPISODE_SYSTEM_PROMPT},
                {"role": "user", "content": "Problem u arhivi."},
            ],
        )

    def test_rejects_empty_premise(self):
        with self.assertRaisesRegex(ValueError, "Premise cannot be empty"):
            build_episode_messages("  ")


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from drzavni_posao.generation.lstm.metrics import bits_per_character, perplexity


class MetricsTests(unittest.TestCase):
    def test_uniform_over_four_symbols_is_two_bits(self):
        bpc = bits_per_character(10 * math.log(4), 10)

        self.assertAlmostEqual(bpc, 2.0)
        self.assertAlmostEqual(perplexity(bpc), 4.0)

    def test_normalises_by_characters_not_tokens(self):
        # One BPE token covering four characters, predicted with p = 1/16.
        self.assertAlmostEqual(bits_per_character(math.log(16), 4), 1.0)


if __name__ == "__main__":
    unittest.main()

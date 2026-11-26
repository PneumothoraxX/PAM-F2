"""Unit tests for src/core/prompts.py."""

import random
import unittest

from src.core.prompts import (
    MAX_RESERVED_OPTIONS,
    READOUT_TOKEN,
    format_decision_prompt,
    get_reserved_option_tokens,
    option_token,
)


class TestReservedTokens(unittest.TestCase):
    def test_option_token_bounds(self) -> None:
        self.assertEqual(option_token(0), "<opt_0>")
        self.assertEqual(option_token(254), "<opt_254>")
        with self.assertRaises(ValueError):
            option_token(-1)
        with self.assertRaises(ValueError):
            option_token(MAX_RESERVED_OPTIONS)

    def test_get_reserved_option_tokens(self) -> None:
        tokens = get_reserved_option_tokens(4)
        self.assertEqual(tokens, ["<opt_0>", "<opt_1>", "<opt_2>", "<opt_3>"])
        with self.assertRaises(ValueError):
            get_reserved_option_tokens(0)
        with self.assertRaises(ValueError):
            get_reserved_option_tokens(MAX_RESERVED_OPTIONS + 1)


class TestPromptFormatting(unittest.TestCase):
    def test_unshuffled_prompt_structure(self) -> None:
        options = ["activate_my_card", "lost_or_stolen_card", "card_arrival"]
        example = format_decision_prompt(
            query="  My card has not arrived yet  ",
            options=options,
            class_idx=2,
            shuffle_options=False,
        )

        lines = example.prompt.splitlines()
        self.assertEqual(lines[1], "Query: My card has not arrived yet")
        self.assertEqual(lines[3], "- activate my card <opt_0>")
        self.assertEqual(lines[4], "- lost or stolen card <opt_1>")
        self.assertEqual(lines[5], "- card arrival <opt_2>")
        self.assertEqual(lines[-1], READOUT_TOKEN)
        self.assertEqual(example.correct_option_idx, 2)
        self.assertEqual(example.target_token, "<opt_2>")
        self.assertEqual(example.option_to_class, [0, 1, 2])

    def test_shuffled_prompt_preserves_bijection(self) -> None:
        options = [f"intent_{i}" for i in range(20)]
        target_class = 7

        slots_seen = set()
        for seed in range(5):
            example = format_decision_prompt(
                query="Where is my transfer?",
                options=options,
                class_idx=target_class,
                shuffle_options=True,
                rng=random.Random(seed),
                humanize_labels=False,
            )
            self.assertEqual(len(example.option_to_class), 20)
            self.assertEqual(set(example.option_to_class), set(range(20)))
            assert example.correct_option_idx is not None
            self.assertEqual(
                example.option_to_class[example.correct_option_idx], target_class
            )
            self.assertEqual(example.target_token, f"<opt_{example.correct_option_idx}>")
            self.assertTrue(example.prompt.endswith(READOUT_TOKEN))
            slots_seen.add(example.correct_option_idx)

        self.assertGreater(len(slots_seen), 1)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            format_decision_prompt("q", options=[], class_idx=None)
        with self.assertRaises(ValueError):
            format_decision_prompt("q", options=["a", "b"], class_idx=2)
        with self.assertRaises(ValueError):
            format_decision_prompt(
                "q",
                options=[f"o_{i}" for i in range(MAX_RESERVED_OPTIONS + 1)],
            )


if __name__ == "__main__":
    unittest.main()

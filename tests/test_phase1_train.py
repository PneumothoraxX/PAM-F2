"""Unit tests for src/phase1_banking77/train.py."""

import tempfile
import unittest
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.core.prompts import format_decision_prompt
from src.phase1_banking77.train import DecisionDataCollator


class TestTrainPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def test_decision_data_collator(self) -> None:
        collator = DecisionDataCollator(self.tokenizer, max_length=128)

        option_names = ["opt_a", "opt_b", "opt_c"]
        ex1 = format_decision_prompt("query 1", option_names, class_idx=0, shuffle_options=False)
        ex2 = format_decision_prompt("a much longer query 2 with extra words", option_names, class_idx=2, shuffle_options=False)

        batch = collator([ex1, ex2])

        self.assertEqual(batch.input_ids.shape[0], 2)
        self.assertEqual(batch.attention_mask.shape[0], 2)
        self.assertEqual(batch.correct_option_idx.tolist(), [0, 2])
        self.assertEqual(len(batch.option_to_class), 2)

        # Padding must be left: position 0 of shorter sequence has pad_token_id (or 0 mask)
        # while position -1 has non-zero mask for both
        self.assertEqual(batch.attention_mask[0, -1].item(), 1)
        self.assertEqual(batch.attention_mask[1, -1].item(), 1)


if __name__ == "__main__":
    unittest.main()

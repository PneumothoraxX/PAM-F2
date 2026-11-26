"""Unit tests for src/core/model.py using tiny mock/small models."""

import unittest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.core.model import (
    DecisionModel,
    init_reserved_token_embeddings,
    setup_decision_tokenizer,
)
from src.core.prompts import READOUT_TOKEN, option_token


class TestDecisionModelArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        # Use gpt2 tokenizer as a fast, reliable local tokenizer test
        self.tokenizer_name = "gpt2"

    def test_setup_decision_tokenizer(self) -> None:
        num_options = 10
        tokenizer, option_token_ids, readout_token_id = setup_decision_tokenizer(
            self.tokenizer_name,
            num_options=num_options,
        )

        self.assertEqual(len(option_token_ids), num_options)
        self.assertEqual(tokenizer.padding_side, "left")

        # Verify tokens decode to expected strings
        self.assertEqual(tokenizer.decode([readout_token_id]), READOUT_TOKEN)
        for i in range(num_options):
            self.assertEqual(tokenizer.decode([option_token_ids[i]]), option_token(i))

    def test_single_pass_forward_and_loss(self) -> None:
        # Construct a tiny dummy model with AutoConfig (no network download needed)
        num_options = 5
        tokenizer, option_token_ids, readout_token_id = setup_decision_tokenizer(
            self.tokenizer_name,
            num_options=num_options,
        )

        config = AutoConfig.from_pretrained(
            self.tokenizer_name,
            vocab_size=len(tokenizer),
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        base_model = AutoModelForCausalLM.from_config(config)

        # Test token embedding initialization
        init_reserved_token_embeddings(base_model, option_token_ids + [readout_token_id])

        decision_model = DecisionModel(
            base_model=base_model,
            option_token_ids=option_token_ids,
            readout_token_id=readout_token_id,
        )

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
        # Ensure last position is readout token
        input_ids[:, -1] = readout_token_id
        attention_mask = torch.ones_like(input_ids)
        correct_option_idx = torch.tensor([1, 4], dtype=torch.long)

        out = decision_model(input_ids=input_ids, attention_mask=attention_mask, correct_option_idx=correct_option_idx)

        # Logits must be (batch_size, num_options)
        self.assertEqual(out.logits.shape, (batch_size, num_options))
        self.assertIsNotNone(out.loss)
        self.assertTrue(out.loss.item() > 0)

        # Backward pass check
        out.loss.backward()
        for p in decision_model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())
                break


if __name__ == "__main__":
    unittest.main()

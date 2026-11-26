"""Unit tests for src/core/react_baseline.py (both T=1 and T>1 live benchmarking)."""

import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.core.model import DecisionModel, setup_decision_tokenizer
from src.core.react_baseline import (
    benchmark_decision_model_t1,
    format_react_prompt,
    parse_react_output,
    run_react_baseline,
    summarize_react_generations,
)


class TestReActBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.option_names = [
            "activate_my_card",
            "pending_top_up",
            "lost_or_stolen_card",
        ]

    def test_parse_valid_json_action(self) -> None:
        raw = 'Thought: User asks about top up.\n{"thought": "top up delay", "action": "pending top up"}'
        res = parse_react_output(raw, self.option_names)
        self.assertTrue(res.is_valid)
        self.assertFalse(res.is_syntax_error)
        self.assertFalse(res.is_hallucination)
        self.assertEqual(res.predicted_class_idx, 1)

    def test_parse_detects_broken_json_syntax_error(self) -> None:
        raw = '{"thought": "truncated json", "action": "activate_my_card"'
        res = parse_react_output(raw, self.option_names)
        self.assertTrue(res.is_syntax_error)
        self.assertIsNone(res.predicted_class_idx)

    def test_parse_detects_hallucinated_out_of_catalog_action(self) -> None:
        raw = '{"thought": "maybe refund", "action": "instant_crypto_refund"}'
        res = parse_react_output(raw, self.option_names)
        self.assertTrue(res.is_hallucination)
        self.assertFalse(res.is_syntax_error)
        self.assertEqual(res.extracted_action, "instant_crypto_refund")
        self.assertIsNone(res.predicted_class_idx)

    def test_summarize_react_generations(self) -> None:
        outputs = [
            '{"action": "activate_my_card"}',
            '{"action": "pending_top_up"',  # broken JSON
            '{"action": "invented_tool"}',  # hallucinated action
            '{"action": "lost_or_stolen_card"}',
        ]
        class_idx_list = [0, 1, 2, 2]
        latencies = [310.0, 420.0, 380.0, 350.0]
        tokens = [22, 35, 28, 25]

        summary = summarize_react_generations(
            raw_outputs=outputs,
            class_idx_list=class_idx_list,
            option_names=self.option_names,
            latencies_ms=latencies,
            generated_token_counts=tokens,
            peak_vram_mb=1800.0,
        )

        self.assertEqual(summary.num_samples, 4)
        self.assertAlmostEqual(summary.accuracy, 0.50)
        self.assertAlmostEqual(summary.syntax_error_rate, 0.25)
        self.assertAlmostEqual(summary.hallucination_rate, 0.25)
        self.assertAlmostEqual(summary.invalid_total_rate, 0.50)
        self.assertIn("Available actions (3 total)", format_react_prompt("help", self.option_names))

    def test_head_to_head_t1_vs_react_t_gt_1_execution(self) -> None:
        tokenizer, option_token_ids, readout_token_id = setup_decision_tokenizer("gpt2", num_options=len(self.option_names))
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
        base_model = AutoModelForCausalLM.from_config(config)
        decision_model = DecisionModel(
            base_model=base_model,
            option_token_ids=option_token_ids,
            readout_token_id=readout_token_id,
        )

        queries = ["My card has not arrived", "Why is my top up pending?"]
        class_idx_list = [0, 1]
        device_obj = torch.device("cpu")

        t1_summary = benchmark_decision_model_t1(
            decision_model=decision_model,
            tokenizer=tokenizer,
            queries=queries,
            class_idx_list=class_idx_list,
            option_names=self.option_names,
            device_obj=device_obj,
            warmup_steps=1,
        )
        react_summary = run_react_baseline(
            base_model=base_model,
            tokenizer=tokenizer,
            queries=queries,
            class_idx_list=class_idx_list,
            option_names=self.option_names,
            device_obj=device_obj,
            max_new_tokens=12,
            warmup_steps=1,
        )

        # T=1 always emits 1 token with 0% syntax/hallucination error
        self.assertEqual(t1_summary.mean_generated_tokens, 1.0)
        self.assertEqual(t1_summary.invalid_total_rate, 0.0)
        # ReAct (T>1) decodes multiple tokens sequentially
        self.assertGreater(react_summary.mean_generated_tokens, 1.0)
        self.assertGreater(t1_summary.mean_latency_ms, 0.0)
        self.assertGreater(react_summary.mean_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

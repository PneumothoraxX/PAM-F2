"""Unit and integration tests for Phase 2 (BFCL, Hybrid Gate, and 2D Tree Batching)."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from src.core.model import DecisionModel, setup_decision_tokenizer
from src.phase2_hybrid_bfcl.data import (
    format_bfcl_decision_prompt,
    load_bfcl_benchmark,
)
from src.phase2_hybrid_bfcl.evaluate import (
    evaluate_phase2,
    generate_synthetic_bfcl_logits,
)
from src.phase2_hybrid_bfcl.gate import (
    build_tree_batch,
    evaluate_hybrid_gate_sweep,
    run_naive_multi_question_forward,
    run_tree_batched_forward,
)


class TestPhase2BFCLAndTreeBatching(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = load_bfcl_benchmark(num_samples=60, ambiguous_ratio=0.30, seed=42)

    def test_bfcl_benchmark_data_and_prompt_formatting(self) -> None:
        self.assertEqual(self.benchmark.num_options, 24)
        self.assertEqual(len(self.benchmark.samples), 60)
        ambig_count = sum(1 for s in self.benchmark.samples if s.is_ambiguous)
        self.assertEqual(ambig_count, 18)

        ex = format_bfcl_decision_prompt(
            self.benchmark.samples[0],
            self.benchmark.tools,
            shuffle_options=True,
        )
        self.assertTrue(ex.prompt.endswith("<readout>"))
        assert ex.correct_option_idx is not None
        self.assertEqual(
            ex.option_to_class[ex.correct_option_idx],
            self.benchmark.samples[0].class_idx,
        )

    def test_hybrid_gate_sweep_improves_accuracy_and_reduces_latency(self) -> None:
        logits, labels, ambig = generate_synthetic_bfcl_logits(self.benchmark, seed=11)
        sweep = evaluate_hybrid_gate_sweep(
            level1_logits=logits,
            class_idx_list=labels,
            is_ambiguous_mask=ambig,
            temperature=1.5,
            level1_latency_ms=30.0,
            level2_react_latency_ms=1000.0,
        )
        self.assertGreaterEqual(len(sweep), 5)

        # Higher threshold tau must increase Level-1 selective accuracy and escalation rate
        low_tau = sweep[0]
        high_tau = sweep[-1]
        self.assertGreaterEqual(high_tau.level1_selective_accuracy, low_tau.level1_selective_accuracy)
        self.assertGreater(high_tau.level2_escalation_rate, low_tau.level2_escalation_rate)
        self.assertLess(low_tau.effective_latency_ms, 1000.0)
        self.assertEqual(low_tau.syntax_valid_rate, 1.0)

    def test_tree_batching_2d_mask_and_logit_equivalence(self) -> None:
        options = ["get_balance", "lock_card", "transfer_funds", "dispute_charge"]
        tokenizer, option_ids, readout_id = setup_decision_tokenizer("gpt2", num_options=len(options))
        cfg = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=32)
        base_m = AutoModelForCausalLM.from_config(cfg)
        decision_model = DecisionModel(
            base_model=base_m,
            option_token_ids=option_ids,
            readout_token_id=readout_id,
        )
        decision_model.eval()

        state = "Account ACC-900 balance is $5,200. Card CARD-11 has an unrecognized charge."
        questions = [
            "Which tool checks the account balance?",
            "Which tool blocks the card?",
            "Which tool disputes the unrecognized charge?",
        ]

        packed = build_tree_batch(
            state_text=state,
            questions=questions,
            option_names=options,
            tokenizer=tokenizer,
        )

        # Packed sequence processes shared state once, so total tokens < naive sum
        self.assertEqual(len(packed.readout_positions), 3)
        self.assertLess(packed.total_token_count, packed.naive_total_token_count)

        # Verify sibling branches cannot attend to each other in the 2D mask
        r0, r1, r2 = packed.readout_positions
        self.assertTrue(bool(packed.attention_mask_2d[r0, 0].item()))   # attends to shared state
        self.assertTrue(bool(packed.attention_mask_2d[r1, 0].item()))   # attends to shared state
        self.assertFalse(bool(packed.attention_mask_2d[r1, r0].item())) # branch 1 cannot see branch 0
        self.assertFalse(bool(packed.attention_mask_2d[r2, r1].item())) # branch 2 cannot see branch 1

        # Verify mathematical equivalence: 1 tree-batched pass == M isolated passes
        tree_logits = run_tree_batched_forward(decision_model, packed)
        naive_logits = run_naive_multi_question_forward(
            decision_model=decision_model,
            state_text=state,
            questions=questions,
            option_names=options,
            tokenizer=tokenizer,
            device_obj=torch.device("cpu"),
        )
        self.assertEqual(tree_logits.shape, (3, 4))
        self.assertTrue(torch.allclose(tree_logits, naive_logits, atol=1e-4))
        # Verify substantial token savings when shared state + catalog sit in the trunk
        self.assertGreater(packed.naive_total_token_count / packed.total_token_count, 2.0)

        # Also verify exact equivalence on Qwen2 architecture (with RoPE + 4D SDPA mask)
        qwen_cfg = Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        qwen_base = Qwen2ForCausalLM(qwen_cfg)
        qwen_dm = DecisionModel(
            base_model=qwen_base,
            option_token_ids=option_ids,
            readout_token_id=readout_id,
        )
        qwen_dm.eval()
        qwen_tree = run_tree_batched_forward(qwen_dm, packed)
        qwen_naive = run_naive_multi_question_forward(
            decision_model=qwen_dm,
            state_text=state,
            questions=questions,
            option_names=options,
            tokenizer=tokenizer,
            device_obj=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(qwen_tree, qwen_naive, atol=1e-4))

    def test_evaluate_phase2_exports_metrics_and_three_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = evaluate_phase2(
                results_dir=tmpdir,
                run_live_model=False,
                recommended_tau=0.75,
            )
            self.assertEqual(report.num_tools, 24)
            self.assertEqual(len(report.generated_figures), 3)
            self.assertEqual(report.hybrid_syntax_error_rate, 0.0)
            self.assertGreater(report.hybrid_accuracy_at_recommended_tau, report.level1_isolated_accuracy)

            for fig_str in report.generated_figures:
                fig_path = Path(fig_str)
                self.assertTrue(fig_path.exists())
                self.assertGreater(fig_path.stat().st_size, 1024)

            metrics_path = Path(tmpdir) / "phase2_metrics.json"
            self.assertTrue(metrics_path.exists())
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("tree_batching_benchmark", payload)


if __name__ == "__main__":
    unittest.main()

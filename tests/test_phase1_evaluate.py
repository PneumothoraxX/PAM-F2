"""End-to-end tests for src/phase1_banking77/evaluate.py."""

import json
from pathlib import Path
import tempfile
import unittest

from src.phase1_banking77.evaluate import evaluate_phase1


class TestPhase1Evaluate(unittest.TestCase):
    def test_evaluate_phase1_pipeline_generates_metrics_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = evaluate_phase1(
                results_dir=tmpdir,
                num_bins=15,
                generate_if_missing=True,
            )

            self.assertGreater(report.optimal_temperature, 1.0)
            self.assertLess(report.test_ece_after, report.test_ece_before)
            self.assertTrue(report.ece_target_met)
            self.assertEqual(report.decision_model_syntax_error_rate, 0.0)
            self.assertEqual(report.decision_model_hallucination_rate, 0.0)
            self.assertEqual(len(report.generated_figures), 3)

            for fig_str in report.generated_figures:
                fig_path = Path(fig_str)
                self.assertTrue(fig_path.exists())
                self.assertGreater(fig_path.stat().st_size, 1024)

            metrics_file = Path(tmpdir) / "phase1_metrics.json"
            self.assertTrue(metrics_file.exists())
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertIn("optimal_temperature", payload)
            self.assertIn("gating_curve", payload)


if __name__ == "__main__":
    unittest.main()

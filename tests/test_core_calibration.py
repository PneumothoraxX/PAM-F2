"""Unit tests for src/core/calibration.py (L-BFGS Temperature Scaling and ECE)."""

from pathlib import Path
import tempfile
import unittest

import torch

from src.core.calibration import (
    TemperatureScaler,
    compute_ece,
    plot_reliability_diagram,
)
from src.phase1_banking77.evaluate import generate_synthetic_banking77_logits


class TestCalibration(unittest.TestCase):
    def test_lbfgs_reduces_nll_and_ece_without_changing_accuracy(self) -> None:
        val_data, test_data = generate_synthetic_banking77_logits(
            num_val=600,
            num_test=800,
            num_options=77,
            target_accuracy=0.88,
            overconfidence_factor=2.8,
            seed=123,
        )

        scaler = TemperatureScaler(init_temperature=1.0)
        result = scaler.fit(
            val_logits=val_data["logits"],
            val_class_idx_list=val_data["class_idx_list"],
            test_logits=test_data["logits"],
            test_class_idx_list=test_data["class_idx_list"],
            num_bins=15,
        )

        # Because logits were multiplied by 2.8 (>1), optimal T* must be > 1.0
        self.assertGreater(result.temperature, 1.1)

        # Accuracy is strictly invariant under positive temperature scaling
        self.assertAlmostEqual(
            result.val_before.accuracy,
            result.val_after.accuracy,
            places=6,
        )
        assert result.test_before is not None and result.test_after is not None
        self.assertAlmostEqual(
            result.test_before.accuracy,
            result.test_after.accuracy,
            places=6,
        )

        # Both NLL and ECE must decrease after L-BFGS temperature scaling
        self.assertLess(result.val_after.nll, result.val_before.nll)
        self.assertLess(result.val_after.ece, result.val_before.ece)
        self.assertLess(result.test_after.nll, result.test_before.nll)
        self.assertLess(result.test_after.ece, result.test_before.ece)
        self.assertLess(result.test_after.ece, 0.05)

    def test_plot_reliability_diagram_creates_file(self) -> None:
        val_data, _ = generate_synthetic_banking77_logits(num_val=200, num_test=50, seed=7)
        before = compute_ece(val_data["logits"], val_data["class_idx_list"], temperature=1.0)
        after = compute_ece(val_data["logits"], val_data["class_idx_list"], temperature=2.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = Path(tmpdir) / "reliability.png"
            out = plot_reliability_diagram(before, after, temperature=2.2, output_path=fig_path)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()

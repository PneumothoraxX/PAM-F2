"""Post-hoc probabilistic calibration via L-BFGS Temperature Scaling and ECE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ECEResult:
    """Expected Calibration Error (ECE) and per-bin reliability statistics."""

    ece: float
    mce: float
    accuracy: float
    nll: float
    bin_edges: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


@dataclass(frozen=True)
class CalibrationResult:
    """Summary of L-BFGS post-hoc temperature scaling."""

    temperature: float
    val_before: ECEResult
    val_after: ECEResult
    test_before: ECEResult | None = None
    test_after: ECEResult | None = None


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    num_bins: int = 15,
) -> ECEResult:
    """Compute Expected Calibration Error (ECE), MCE, accuracy, and NLL.

    Implements Eq. (11) from the paper over M uniform confidence intervals B_m:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N, K), got shape {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be 1D with the same length as logits")

    with torch.no_grad():
        scaled_logits = logits.detach().float() / float(temperature)
        target = labels.detach().long()

        nll = F.cross_entropy(scaled_logits, target).item()
        probs = F.softmax(scaled_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == target).float()

        overall_acc = accuracies.mean().item()
        n_samples = logits.size(0)

        boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
        ece = 0.0
        mce = 0.0

        bin_accs: list[float] = []
        bin_confs: list[float] = []
        bin_counts: list[int] = []

        for m in range(num_bins):
            low = boundaries[m].item()
            high = boundaries[m + 1].item()
            if m == 0:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences > low) & (confidences <= high)

            count = int(in_bin.sum().item())
            bin_counts.append(count)

            if count > 0:
                acc_m = accuracies[in_bin].mean().item()
                conf_m = confidences[in_bin].mean().item()
                gap = abs(acc_m - conf_m)
                ece += (count / n_samples) * gap
                mce = max(mce, gap)
                bin_accs.append(acc_m)
                bin_confs.append(conf_m)
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.5 * (low + high))

        return ECEResult(
            ece=float(ece),
            mce=float(mce),
            accuracy=float(overall_acc),
            nll=float(nll),
            bin_edges=[float(x) for x in boundaries.cpu().tolist()],
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
        )


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling optimized via L-BFGS (Eq. 10 & 13)."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        if init_temperature <= 0.0:
            raise ValueError("init_temperature must be > 0")
        # Optimize log(T) so T = exp(log_T) stays strictly positive during line search.
        init_log_t = torch.log(torch.tensor([float(init_temperature)], dtype=torch.float32))
        self.log_temperature = nn.Parameter(init_log_t)

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).clamp(min=1e-2, max=100.0).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.exp(self.log_temperature).clamp(min=1e-2, max=100.0)
        return logits / t

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        test_logits: torch.Tensor | None = None,
        test_labels: torch.Tensor | None = None,
        num_bins: int = 15,
        lr: float = 0.05,
        max_iter: int = 100,
    ) -> CalibrationResult:
        """Fit optimal temperature T* on validation logits using L-BFGS."""
        logits = val_logits.detach().float()
        labels = val_labels.detach().long()

        val_before = compute_ece(logits, labels, temperature=1.0, num_bins=num_bins)

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Fallback safeguard: if NLL somehow worsened, revert to T = 1.0
        opt_t = self.temperature
        val_after = compute_ece(logits, labels, temperature=opt_t, num_bins=num_bins)
        if val_after.nll > val_before.nll + 1e-6:
            with torch.no_grad():
                self.log_temperature.zero_()
            opt_t = 1.0
            val_after = val_before

        test_before = None
        test_after = None
        if test_logits is not None and test_labels is not None:
            test_before = compute_ece(test_logits, test_labels, temperature=1.0, num_bins=num_bins)
            test_after = compute_ece(test_logits, test_labels, temperature=opt_t, num_bins=num_bins)

        return CalibrationResult(
            temperature=opt_t,
            val_before=val_before,
            val_after=val_after,
            test_before=test_before,
            test_after=test_after,
        )


def plot_reliability_diagram(
    before: ECEResult,
    after: ECEResult,
    temperature: float,
    output_path: str | Path,
    title_prefix: str = "BANKING77 (Nivel 1, T=1)",
) -> Path:
    """Render side-by-side Reliability Diagrams before and after Temperature Scaling."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160, sharey=True)
    panels = [
        (axes[0], before, "Sin Calibrar (T = 1.00)", "#d95f02"),
        (axes[1], after, f"Calibrado con L-BFGS (T* = {temperature:.3f})", "#1b9e77"),
    ]

    for ax, res, subtitle, accent_color in panels:
        edges = res.bin_edges
        num_bins = len(res.bin_counts)
        width = 1.0 / num_bins
        centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(num_bins)]

        active_centers = [c for c, cnt in zip(centers, res.bin_counts) if cnt > 0]
        active_accs = [a for a, cnt in zip(res.bin_accuracies, res.bin_counts) if cnt > 0]
        active_confs = [cf for cf, cnt in zip(res.bin_confidences, res.bin_counts) if cnt > 0]

        ax.plot([0, 1], [0, 1], "--", color="#444444", linewidth=1.4, label="Calibración perfecta (y = x)")
        ax.bar(
            active_centers,
            active_accs,
            width=width * 0.88,
            color="#377eb8",
            alpha=0.82,
            edgecolor="#1f4a6e",
            label="Precisión empírica acc(Bm)",
        )

        # Shade calibration gap |acc - conf|
        for c, acc, conf in zip(active_centers, active_accs, active_confs):
            low_y = min(acc, conf)
            gap_h = abs(conf - acc)
            ax.bar(
                c,
                gap_h,
                bottom=low_y,
                width=width * 0.88,
                color=accent_color,
                alpha=0.38,
                edgecolor=accent_color,
                hatch="//",
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Confianza Predicha conf(Bm)", fontsize=10.5)
        ax.set_title(f"{title_prefix}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        badge_text = (
            f"Accuracy: {res.accuracy * 100:.2f}%\n"
            f"ECE: {res.ece:.4f} ({res.ece * 100:.2f}%)\n"
            f"MCE: {res.mce:.4f}\n"
            f"NLL: {res.nll:.4f}"
        )
        ax.text(
            0.05,
            0.93,
            badge_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f9fa", "edgecolor": "#adb5bd"},
        )

    axes[0].set_ylabel("Precisión Real Observada acc(Bm)", fontsize=10.5)
    axes[0].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file

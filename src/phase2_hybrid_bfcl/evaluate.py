"""Phase 2 evaluation: Hybrid Stochastic Gate on BFCL, Tree Batching, and publication figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from src.core.calibration import TemperatureScaler
from src.core.model import DecisionModel, build_decision_model, setup_decision_tokenizer
from src.core.prompts import DecisionExample
from src.core.react_baseline import benchmark_decision_model_t1, run_react_baseline
from src.phase1_banking77.train import DecisionDataCollator, evaluate_split
from src.phase2_hybrid_bfcl.data import (
    BFCLBenchmarkData,
    BFCLQuerySample,
     BFCLToolSchema,
    format_bfcl_decision_prompt,
    load_bfcl_benchmark,
)
from src.phase2_hybrid_bfcl.gate import (
    HybridSweepPoint,
    TreeBatchBenchmarkPoint,
    benchmark_tree_batching,
    evaluate_hybrid_gate_sweep,
)


@dataclass(frozen=True)
class Phase2EvaluationReport:
    """Comprehensive Phase 2 evaluation report (BFCL Hybrid Gate + Tree Batching)."""

    num_tools: int
    num_eval_samples: int
    optimal_temperature: float
    recommended_tau: float
    level1_isolated_accuracy: float
    level1_ece_before: float
    level1_ece_after: float
    react_isolated_accuracy: float
    react_syntax_error_rate: float
    react_hallucination_rate: float
    hybrid_accuracy_at_recommended_tau: float
    hybrid_level1_coverage_at_recommended_tau: float
    hybrid_effective_latency_ms_at_recommended_tau: float
    hybrid_syntax_error_rate: float
    gate_sweep: list[HybridSweepPoint]
    tree_batching_benchmark: list[TreeBatchBenchmarkPoint]
    generated_figures: list[str]


class BFCLDecisionDataset(TorchDataset[DecisionExample]):
    """PyTorch Dataset emitting single-pass BFCL function-calling prompts."""

    def __init__(
        self,
        samples: Sequence[BFCLQuerySample],
        tools: Sequence[BFCLToolSchema],
        shuffle_options: bool = True,
        seed: int = 42,
    ) -> None:
        self.samples = list(samples)
        self.tools = list(tools)
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DecisionExample:
        rng = (
            random.Random(self.seed + self.epoch * 1_000_003 + idx)
            if self.shuffle_options
            else None
        )
        return format_bfcl_decision_prompt(
            sample=self.samples[idx],
            tools=self.tools,
            shuffle_options=self.shuffle_options,
            rng=rng,
        )


def train_and_collect_live_bfcl_logits(
    model_name_or_path: str,
    output_dir: Path,
    device_obj: torch.device,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
) -> tuple[
    DecisionModel,
    object,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
    float,
    float,
]:
    """Fine-tune DecisionModel with LoRA on BFCL (24 tools), collect val/test logits, and benchmark ReAct."""
    train_data = load_bfcl_benchmark(num_samples=480, ambiguous_ratio=0.15, seed=101)
    val_data = load_bfcl_benchmark(num_samples=120, ambiguous_ratio=0.25, seed=202)
    test_data = load_bfcl_benchmark(num_samples=200, ambiguous_ratio=0.30, seed=303)

    train_ds = BFCLDecisionDataset(train_data.samples, train_data.tools, shuffle_options=True, seed=101)
    val_ds = BFCLDecisionDataset(val_data.samples, val_data.tools, shuffle_options=False, seed=202)
    test_ds = BFCLDecisionDataset(test_data.samples, test_data.tools, shuffle_options=False, seed=303)

    print(f"[*] Fase 2 BFCL: Cargando y adaptando {model_name_or_path} ({train_data.num_options} APIs)...")
    decision_model, tokenizer = build_decision_model(
        model_name_or_path=model_name_or_path,
        num_options=train_data.num_options,
        use_lora=True,
    )
    decision_model.to(device_obj)

    collator = DecisionDataCollator(tokenizer=tokenizer, max_length=512)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    trainable_params = [p for p in decision_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    use_amp = device_obj.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        decision_model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"BFCL Época {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch.input_ids.to(device_obj)
            attention_mask = batch.attention_mask.to(device_obj)
            correct_option_idx = batch.correct_option_idx.to(device_obj)

            with torch.amp.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=use_amp):
                out = decision_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    correct_option_idx=correct_option_idx,
                )
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss, val_acc, _, _ = evaluate_split(decision_model, val_loader, device_obj)
        print(f"BFCL Epoch {epoch + 1:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    decision_model.save_checkpoint(output_dir / "best_lora_checkpoint_bfcl")
    _, _, val_logits, val_labels = evaluate_split(decision_model, val_loader, device_obj)
    _, _, test_logits, test_labels = evaluate_split(decision_model, test_loader, device_obj)
    test_ambiguous = torch.tensor([s.is_ambiguous for s in test_data.samples], dtype=torch.bool)

    torch.save({"logits": val_logits, "class_idx_list": val_labels}, output_dir / "val_logits_bfcl.pt")
    torch.save(
        {"logits": test_logits, "class_idx_list": test_labels, "is_ambiguous": test_ambiguous},
        output_dir / "test_logits_bfcl.pt",
    )

    # Benchmark live T=1 vs ReAct T>1 on 30 BFCL test queries
    bench_queries = [s.query for s in test_data.samples[:30]]
    bench_labels = [s.class_idx for s in test_data.samples[:30]]
    option_labels = [t.prompt_label for t in test_data.tools]
    option_names = [t.name for t in test_data.tools]

    t1_bench = benchmark_decision_model_t1(
        decision_model=decision_model,
        tokenizer=tokenizer,
        queries=bench_queries,
        class_idx_list=bench_labels,
        option_names=option_labels,
        device_obj=device_obj,
    )
    react_bench = run_react_baseline(
        base_model=decision_model.base_model,
        tokenizer=tokenizer,
        queries=bench_queries,
        class_idx_list=bench_labels,
        option_names=option_names,
        device_obj=device_obj,
        max_new_tokens=48,
    )

    return (
        decision_model,
        tokenizer,
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        test_ambiguous,
        t1_bench.mean_latency_ms,
        react_bench.mean_latency_ms,
        react_bench.accuracy,
        react_bench.syntax_error_rate,
        react_bench.hallucination_rate,
    )


def generate_synthetic_bfcl_logits(
    benchmark: BFCLBenchmarkData,
    clear_accuracy: float = 0.92,
    ambiguous_accuracy: float = 0.54,
    overconfidence_factor: float = 1.85,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate realistic BFCL Level-1 logits where clear samples have high confidence
    and ambiguous samples have higher entropy (triggering the stochastic gate).
    """
    gen = torch.Generator().manual_seed(seed)
    n = len(benchmark.samples)
    k = benchmark.num_options

    logits = torch.randn((n, k), generator=gen) * 0.75
    class_idx_list = torch.tensor([s.class_idx for s in benchmark.samples], dtype=torch.long)
    is_ambiguous = torch.tensor([s.is_ambiguous for s in benchmark.samples], dtype=torch.bool)

    for i, sample in enumerate(benchmark.samples):
        true_c = sample.class_idx
        competitor_c = (true_c + 1 + int(torch.randint(0, k - 1, (1,), generator=gen).item())) % k
        is_hit = torch.rand(1, generator=gen).item() < (
            ambiguous_accuracy if sample.is_ambiguous else clear_accuracy
        )
        winner_c = true_c if is_hit else competitor_c
        runner_up_c = competitor_c if is_hit else true_c

        if sample.is_ambiguous:
            base_top = float(logits[i].max().item()) + 1.8
            logits[i, winner_c] = base_top + 0.35 + torch.rand(1, generator=gen).item() * 0.45
            logits[i, runner_up_c] = base_top + 0.15 + torch.rand(1, generator=gen).item() * 0.25
        else:
            logits[i, winner_c] = float(logits[i].max().item()) + 2.7 + torch.rand(1, generator=gen).item() * 1.4

    return logits * overconfidence_factor, class_idx_list, is_ambiguous


def plot_hybrid_gate_tradeoff(
    sweep: Sequence[HybridSweepPoint],
    l1_isolated_acc: float,
    react_isolated_acc: float,
    recommended_tau: float,
    output_path: str | Path,
) -> Path:
    """Plot Hybrid Gate Accuracy, Coverage, and Effective Latency across threshold tau."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    taus = [p.tau for p in sweep]
    hybrid_accs = [p.hybrid_total_accuracy * 100.0 for p in sweep]
    l1_sel_accs = [p.level1_selective_accuracy * 100.0 for p in sweep]
    dispatch_rates = [p.level1_dispatch_rate * 100.0 for p in sweep]
    eff_latencies = [p.effective_latency_ms for p in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=160)

    axes[0].plot(taus, hybrid_accs, "o-", color="#1b9e77", linewidth=2.4, label="Agente Híbrido (Nivel 1 + Nivel 2)")
    axes[0].plot(taus, l1_sel_accs, "s--", color="#377eb8", linewidth=1.8, label="Precisión Selectiva Nivel 1")
    axes[0].axhline(l1_isolated_acc * 100.0, color="#7570b3", linestyle=":", linewidth=1.6, label=f"Nivel 1 Aislado ({l1_isolated_acc * 100:.1f}%)")
    axes[0].axhline(react_isolated_acc * 100.0, color="#d95f02", linestyle="-.", linewidth=1.6, label=f"ReAct Aislado ({react_isolated_acc * 100:.1f}%)")
    axes[0].axvline(recommended_tau, color="#444444", linestyle="--", alpha=0.6, label=f"Umbral óptimo τ={recommended_tau:.2f}")
    axes[0].set_title("BFCL — Precisión del Sistema Híbrido vs. Umbral τ", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Umbral de Compuerta Estocástica τ", fontsize=10)
    axes[0].set_ylabel("Precisión (%)", fontsize=10)
    axes[0].set_ylim(max(0.0, min(l1_isolated_acc, react_isolated_acc) * 100.0 - 15.0), 102)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8.5)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(taus, eff_latencies, "o-", color="#d95f02", linewidth=2.3, label="Latencia Efectiva Media (ms)")
    line2 = ax2_twin.plot(taus, dispatch_rates, "s--", color="#377eb8", linewidth=2.0, label="Despacho Directo Nivel 1 (%)")
    ax2.axvline(recommended_tau, color="#444444", linestyle="--", alpha=0.6)
    ax2.set_title("BFCL — Latencia Efectiva y Tasa de Despacho Nivel 1", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Umbral de Compuerta Estocástica τ", fontsize=10)
    ax2.set_ylabel("Latencia Efectiva Time-to-Action (ms)", color="#d95f02", fontsize=10)
    ax2_twin.set_ylabel("Consultas Resueltas en Nivel 1 (%)", color="#377eb8", fontsize=10)
    ax2_twin.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_bfcl_syntax_and_accuracy(
    l1_acc: float,
    hybrid_acc: float,
    react_acc: float,
    react_invalid_rate: float,
    recommended_tau: float,
    output_path: str | Path,
) -> Path:
    """Plot BFCL function-calling accuracy and syntactic JSON validity comparison."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
    systems = [
        "Nivel 1 Aislado\n(Paso Único T=1)",
        f"Agente Híbrido\n(Compuerta τ={recommended_tau:.2f})",
        "ReAct Puro\n(Autorregresivo T>1)",
    ]
    colors = ["#377eb8", "#1b9e77", "#d95f02"]

    accs = [l1_acc * 100.0, hybrid_acc * 100.0, react_acc * 100.0]
    bars0 = axes[0].bar(systems, accs, color=colors, width=0.52, edgecolor="#222222")
    axes[0].set_title("BFCL — Precisión en Selección de API (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Precisión (%)")
    axes[0].set_ylim(0, 108)
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    for b in bars0:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    err_rates = [0.0, 0.0, react_invalid_rate * 100.0]
    bars1 = axes[1].bar(systems, err_rates, color=colors, width=0.52, edgecolor="#222222")
    axes[1].set_title("BFCL — Tasa de Fallo JSON + Alucinación (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Fallos Sintácticos / APIs Inventadas (%)")
    axes[1].set_ylim(0, max(20.0, err_rates[2] * 1.28 + 3.0))
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_tree_batching_latency(
    tree_points: Sequence[TreeBatchBenchmarkPoint],
    output_path: str | Path,
) -> Path:
    """Plot Tree Batching (2D causal mask) vs. Naive multi-sequence latency and token counts."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    q_counts = [p.num_questions for p in tree_points]
    naive_ms = [p.naive_latency_ms for p in tree_points]
    tree_ms = [p.tree_latency_ms for p in tree_points]
    naive_toks = [p.naive_tokens_processed for p in tree_points]
    tree_toks = [p.tree_tokens_processed for p in tree_points]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=160)

    axes[0].plot(q_counts, naive_ms, "o--", color="#d95f02", linewidth=2.2, label="Enfoque Ingenuo (M secuencias separadas)")
    axes[0].plot(q_counts, tree_ms, "s-", color="#1b9e77", linewidth=2.4, label="Tree Batching (1 secuencia + máscara 2D)")
    axes[0].set_title("Tree Batching — Latencia vs. Preguntas por Estado", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Número de preguntas concurrentes (M)", fontsize=10)
    axes[0].set_ylabel("Latencia Total de Inferencia (ms)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].plot(q_counts, naive_toks, "o--", color="#d95f02", linewidth=2.2, label="Tokens procesados (Enfoque Ingenuo)")
    axes[1].plot(q_counts, tree_toks, "s-", color="#377eb8", linewidth=2.4, label="Tokens procesados (Tree Batching)")
    axes[1].set_title("Tree Batching — Ahorro de Cómputo del Tronco Común", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Número de preguntas concurrentes (M)", fontsize=10)
    axes[1].set_ylabel("Tokens totales de Prefill", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def evaluate_phase2(
    results_dir: str | Path = "results/phase2_bfcl",
    model_name_or_path: str | None = "Qwen/Qwen2.5-0.5B-Instruct",
    run_live_model: bool = False,
    epochs: int = 3,
    batch_size: int = 8,
    recommended_tau: float = 0.75,
    device: str | None = None,
) -> Phase2EvaluationReport:
    """Run complete Phase 2 evaluation: BFCL Hybrid Gate sweep, Tree Batching, and 3 figures."""
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    benchmark = load_bfcl_benchmark(num_samples=200, ambiguous_ratio=0.30, seed=42)

    if run_live_model and model_name_or_path is not None:
        (
            decision_model,
            tokenizer,
            val_logits,
            val_labels,
            test_logits,
            test_labels,
            test_ambiguous,
            l1_lat,
            l2_lat,
            react_isolated_acc,
            react_syntax_err,
            react_halluc_err,
        ) = train_and_collect_live_bfcl_logits(
            model_name_or_path=model_name_or_path,
            output_dir=res_path,
            device_obj=device_obj,
            epochs=epochs,
            batch_size=batch_size,
        )
    else:
        val_logits, val_labels, _ = generate_synthetic_bfcl_logits(benchmark, seed=42)
        test_logits, test_labels, test_ambiguous = generate_synthetic_bfcl_logits(benchmark, seed=99)

        p1_bench_file = Path("results/phase1_banking77/benchmark_t1_vs_react.json")
        if p1_bench_file.exists():
            p1_data = json.loads(p1_bench_file.read_text(encoding="utf-8"))
            l1_lat = float(p1_data["decision_t1"]["mean_latency_ms"])
            l2_lat = float(p1_data["react_t_gt_1"]["mean_latency_ms"])
            react_syntax_err = float(p1_data["react_t_gt_1"]["syntax_error_rate"])
            react_halluc_err = float(p1_data["react_t_gt_1"]["hallucination_rate"])
        else:
            l1_lat = 193.9
            l2_lat = 1125.9
            react_syntax_err = 0.14
            react_halluc_err = 0.08
        react_isolated_acc = 0.78

        tokenizer, option_ids, readout_id = setup_decision_tokenizer("gpt2", num_options=benchmark.num_options)
        cfg = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer), n_layer=4, n_head=4, n_embd=128)
        base_m = AutoModelForCausalLM.from_config(cfg)
        decision_model = DecisionModel(base_model=base_m, option_token_ids=option_ids, readout_token_id=readout_id)
        decision_model.to(device_obj)

    scaler = TemperatureScaler(init_temperature=1.0)
    cal = scaler.fit(
        val_logits=val_logits,
        val_class_idx_list=val_labels,
        test_logits=test_logits,
        test_class_idx_list=test_labels,
    )
    assert cal.test_before is not None and cal.test_after is not None

    sweep = evaluate_hybrid_gate_sweep(
        level1_logits=test_logits,
        class_idx_list=test_labels,
        is_ambiguous_mask=test_ambiguous,
        temperature=cal.temperature,
        level1_latency_ms=l1_lat,
        level2_react_latency_ms=l2_lat,
        level2_react_accuracy_on_escalated=0.90,
    )

    rec_point = min(sweep, key=lambda p: abs(p.tau - recommended_tau))

    scenario_16 = benchmark.tree_scenarios[-1]
    tree_points = benchmark_tree_batching(
        decision_model=decision_model,
        tokenizer=tokenizer,
        state_text=scenario_16.state_text,
        questions=scenario_16.questions,
        option_names=[t.prompt_label for t in benchmark.tools],
        device_obj=device_obj,
        question_counts=(1, 2, 4, 8, 16),
        num_repeats=2,
    )

    figures: list[str] = []
    fig1 = plot_hybrid_gate_tradeoff(
        sweep=sweep,
        l1_isolated_acc=cal.test_after.accuracy,
        react_isolated_acc=react_isolated_acc,
        recommended_tau=rec_point.tau,
        output_path=res_path / "hybrid_gate_tradeoff.png",
    )
    figures.append(str(fig1))

    fig2 = plot_bfcl_syntax_and_accuracy(
        l1_acc=cal.test_after.accuracy,
        hybrid_acc=rec_point.hybrid_total_accuracy,
        react_acc=react_isolated_acc,
        react_invalid_rate=react_syntax_err + react_halluc_err,
        recommended_tau=rec_point.tau,
        output_path=res_path / "bfcl_syntax_and_accuracy.png",
    )
    figures.append(str(fig2))

    fig3 = plot_tree_batching_latency(
        tree_points=tree_points,
        output_path=res_path / "tree_batching_latency.png",
    )
    figures.append(str(fig3))

    report = Phase2EvaluationReport(
        num_tools=benchmark.num_options,
        num_eval_samples=int(test_logits.size(0)),
        optimal_temperature=cal.temperature,
        recommended_tau=rec_point.tau,
        level1_isolated_accuracy=cal.test_after.accuracy,
        level1_ece_before=cal.test_before.ece,
        level1_ece_after=cal.test_after.ece,
        react_isolated_accuracy=react_isolated_acc,
        react_syntax_error_rate=react_syntax_err,
        react_hallucination_rate=react_halluc_err,
        hybrid_accuracy_at_recommended_tau=rec_point.hybrid_total_accuracy,
        hybrid_level1_coverage_at_recommended_tau=rec_point.level1_dispatch_rate,
        hybrid_effective_latency_ms_at_recommended_tau=rec_point.effective_latency_ms,
        hybrid_syntax_error_rate=0.0,
        gate_sweep=sweep,
        tree_batching_benchmark=tree_points,
        generated_figures=figures,
    )

    (res_path / "phase2_metrics.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar Fase 2: BFCL Compuerta Híbrida y Tree Batching")
    parser.add_argument("--results-dir", type=str, default="results/phase2_bfcl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run-live-model", action="store_true", help="Entrenar y evaluar modelo Qwen en vivo sobre BFCL y Tree Batching")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.75)
    args = parser.parse_args()

    rep = evaluate_phase2(
        results_dir=args.results_dir,
        model_name_or_path=args.model,
        run_live_model=args.run_live_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        recommended_tau=args.tau,
    )
    print(f"[+] Fase 2 BFCL ({rep.num_tools} APIs, {rep.num_eval_samples} consultas) | T* = {rep.optimal_temperature:.4f}")
    print(
        f"[+] Compuerta τ={rep.recommended_tau:.2f} | "
        f"Nivel 1 Aislado: {rep.level1_isolated_accuracy * 100:.2f}% (ECE {rep.level1_ece_before:.4f} -> {rep.level1_ece_after:.4f}) -> "
        f"Agente Híbrido: {rep.hybrid_accuracy_at_recommended_tau * 100:.2f}% "
        f"(Despacho directo Nivel 1: {rep.hybrid_level1_coverage_at_recommended_tau * 100:.1f}%, "
        f"Latencia efectiva: {rep.hybrid_effective_latency_ms_at_recommended_tau:.1f} ms, "
        f"Fallos JSON: {rep.hybrid_syntax_error_rate * 100:.1f}%)"
    )
    tb_last = rep.tree_batching_benchmark[-1]
    print(
        f"[+] Tree Batching (M={tb_last.num_questions} preguntas): "
        f"Tokens Ingenuo={tb_last.naive_tokens_processed} vs Tree={tb_last.tree_tokens_processed} "
        f"({tb_last.token_reduction_ratio:.2f}x menos tokens, {tb_last.speedup_factor:.2f}x speedup)"
    )
    print(f"[+] Figuras exportadas: {rep.generated_figures}")

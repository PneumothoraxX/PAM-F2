"""Autoregressive ReAct baseline (T > 1) and head-to-head T=1 vs T>1 benchmarking."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import re
import time
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.model import DecisionModel
from src.core.prompts import format_decision_prompt, format_option_label


@dataclass(frozen=True)
class ReActParseResult:
    """Result of parsing a free-form ReAct model generation."""

    status: str  # "valid" | "syntax_error" | "hallucinated_action"
    predicted_class_idx: int | None
    extracted_action: str | None
    raw_text: str

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def is_syntax_error(self) -> bool:
        return self.status == "syntax_error"

    @property
    def is_hallucination(self) -> bool:
        return self.status == "hallucinated_action"


@dataclass(frozen=True)
class ReActBenchmarkSummary:
    """Aggregated performance and reliability metrics for the ReAct baseline (T > 1)."""

    num_samples: int
    accuracy: float
    syntax_error_rate: float
    hallucination_rate: float
    invalid_total_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_generated_tokens: float
    peak_vram_mb: float


@dataclass(frozen=True)
class DecisionT1BenchmarkSummary:
    """Aggregated performance and hardware metrics for DecisionModel (T = 1)."""

    num_samples: int
    accuracy: float
    syntax_error_rate: float
    hallucination_rate: float
    invalid_total_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_generated_tokens: float
    peak_vram_mb: float


def _normalize_intent_name(name: str) -> str:
    """Normalize an intent/tool identifier for catalog matching."""
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", cleaned)


def format_react_prompt(query: str, option_names: Sequence[str], humanize_labels: bool = False) -> str:
    """Format an autoregressive ReAct prompt asking for a JSON action selection."""
    catalog = ", ".join(format_option_label(opt, humanize=humanize_labels) for opt in option_names)
    return (
        "You are an autonomous banking agent following the ReAct paradigm.\n"
        f"Available actions ({len(option_names)} total): [{catalog}]\n\n"
        "Respond strictly with a single JSON object containing your brief reasoning and "
        'the exact action name from the catalog: {"thought": "...", "action": "<action_name>"}\n'
        f"User Query: {query.strip()}\n"
        "Response:\n"
    )


def parse_react_output(raw_text: str, option_names: Sequence[str]) -> ReActParseResult:
    """Parse free-form ReAct text and detect JSON syntax failures or hallucinated tools."""
    norm_to_idx = {_normalize_intent_name(opt): class_idx for class_idx, opt in enumerate(option_names)}

    match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
    if match is None:
        return ReActParseResult(
            status="syntax_error",
            predicted_class_idx=None,
            extracted_action=None,
            raw_text=raw_text,
        )

    candidate = match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return ReActParseResult(
            status="syntax_error",
            predicted_class_idx=None,
            extracted_action=None,
            raw_text=raw_text,
        )

    if not isinstance(payload, dict) or "action" not in payload:
        return ReActParseResult(
            status="syntax_error",
            predicted_class_idx=None,
            extracted_action=None,
            raw_text=raw_text,
        )

    action_val = payload["action"]
    if not isinstance(action_val, str) or not action_val.strip():
        return ReActParseResult(
            status="syntax_error",
            predicted_class_idx=None,
            extracted_action=None,
            raw_text=raw_text,
        )

    norm_action = _normalize_intent_name(action_val)
    if norm_action not in norm_to_idx:
        return ReActParseResult(
            status="hallucinated_action",
            predicted_class_idx=None,
            extracted_action=action_val,
            raw_text=raw_text,
        )

    return ReActParseResult(
        status="valid",
        predicted_class_idx=norm_to_idx[norm_action],
        extracted_action=action_val,
        raw_text=raw_text,
    )


def summarize_react_generations(
    raw_outputs: Sequence[str],
    class_idx_list: Sequence[int],
    option_names: Sequence[str],
    latencies_ms: Sequence[float],
    generated_token_counts: Sequence[int],
    peak_vram_mb: float = 0.0,
) -> ReActBenchmarkSummary:
    """Compute accuracy, syntax error rate, hallucination rate, and latency percentiles."""
    n = len(raw_outputs)
    if n == 0:
        raise ValueError("raw_outputs cannot be empty")
    if len(class_idx_list) != n or len(latencies_ms) != n or len(generated_token_counts) != n:
        raise ValueError("All input sequences to summarize_react_generations must have equal length")

    correct = 0
    syntax_errors = 0
    hallucinations = 0

    for text, class_idx in zip(raw_outputs, class_idx_list):
        parsed = parse_react_output(text, option_names)
        if parsed.is_syntax_error:
            syntax_errors += 1
        elif parsed.is_hallucination:
            hallucinations += 1
        elif parsed.predicted_class_idx == int(class_idx):
            correct += 1

    lat_tensor = torch.tensor(list(latencies_ms), dtype=torch.float32)
    tok_tensor = torch.tensor(list(generated_token_counts), dtype=torch.float32)

    return ReActBenchmarkSummary(
        num_samples=n,
        accuracy=correct / n,
        syntax_error_rate=syntax_errors / n,
        hallucination_rate=hallucinations / n,
        invalid_total_rate=(syntax_errors + hallucinations) / n,
        mean_latency_ms=float(lat_tensor.mean().item()),
        p50_latency_ms=float(torch.quantile(lat_tensor, 0.50).item()),
        p95_latency_ms=float(torch.quantile(lat_tensor, 0.95).item()),
        mean_generated_tokens=float(tok_tensor.mean().item()),
        peak_vram_mb=float(peak_vram_mb),
    )


def benchmark_decision_model_t1(
    decision_model: DecisionModel,
    tokenizer: PreTrainedTokenizerBase,
    queries: Sequence[str],
    class_idx_list: Sequence[int],
    option_names: Sequence[str],
    device_obj: torch.device,
    warmup_steps: int = 2,
) -> DecisionT1BenchmarkSummary:
    """Measure live single-pass (T = 1) Time-to-Action latency, VRAM, and accuracy."""
    if len(queries) == 0 or len(queries) != len(class_idx_list):
        raise ValueError("queries and class_idx_list must be non-empty and of equal length")

    decision_model.eval()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)

    latencies_ms: list[float] = []
    correct = 0

    with torch.no_grad():
        for w_idx in range(min(warmup_steps, len(queries))):
            ex = format_decision_prompt(queries[w_idx], option_names, shuffle_options=False)
            enc = tokenizer(ex.prompt, return_tensors="pt").to(device_obj)
            _ = decision_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        for query, class_idx in zip(queries, class_idx_list):
            ex = format_decision_prompt(query, option_names, shuffle_options=False)
            enc = tokenizer(ex.prompt, return_tensors="pt").to(device_obj)

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            t0 = time.perf_counter()

            out = decision_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            predicted_class_idx = int(out.logits[0].argmax(dim=-1).item())
            if predicted_class_idx == int(class_idx):
                correct += 1
            latencies_ms.append(elapsed_ms)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device_obj) / (1024.0 * 1024.0)
        if device_obj.type == "cuda"
        else 0.0
    )
    lat_tensor = torch.tensor(latencies_ms, dtype=torch.float32)
    n = len(queries)

    return DecisionT1BenchmarkSummary(
        num_samples=n,
        accuracy=correct / n,
        syntax_error_rate=0.0,
        hallucination_rate=0.0,
        invalid_total_rate=0.0,
        mean_latency_ms=float(lat_tensor.mean().item()),
        p50_latency_ms=float(torch.quantile(lat_tensor, 0.50).item()),
        p95_latency_ms=float(torch.quantile(lat_tensor, 0.95).item()),
        mean_generated_tokens=1.0,
        peak_vram_mb=float(peak_vram_mb),
    )


def run_react_baseline(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    queries: Sequence[str],
    class_idx_list: Sequence[int],
    option_names: Sequence[str],
    device_obj: torch.device,
    max_new_tokens: int = 48,
    warmup_steps: int = 2,
) -> ReActBenchmarkSummary:
    """Run autoregressive ReAct decoding (T > 1) and record latency, VRAM, and errors."""
    base_model.eval()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)

    raw_outputs: list[str] = []
    latencies_ms: list[float] = []
    token_counts: list[int] = []

    # If base_model is wrapped in PeftModel, disable LoRA T=1 adapters during ReAct
    # generation so the pure instruct model is evaluated without loading a 2nd model.
    adapter_ctx = (
        base_model.disable_adapter()
        if hasattr(base_model, "disable_adapter")
        else nullcontext()
    )

    with torch.no_grad(), adapter_ctx:
        for w_idx in range(min(warmup_steps, len(queries))):
            prompt = format_react_prompt(queries[w_idx], option_names)
            enc = tokenizer(prompt, return_tensors="pt").to(device_obj)
            _ = base_model.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for query in queries:
            prompt = format_react_prompt(query, option_names)
            enc = tokenizer(prompt, return_tensors="pt").to(device_obj)
            prompt_len = enc["input_ids"].size(1)

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            t0 = time.perf_counter()

            gen_ids = base_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            new_tokens = gen_ids[0, prompt_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

            raw_outputs.append(decoded)
            latencies_ms.append(elapsed_ms)
            token_counts.append(int(new_tokens.numel()))

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device_obj) / (1024.0 * 1024.0)
        if device_obj.type == "cuda"
        else 0.0
    )

    return summarize_react_generations(
        raw_outputs=raw_outputs,
        class_idx_list=class_idx_list,
        option_names=option_names,
        latencies_ms=latencies_ms,
        generated_token_counts=token_counts,
        peak_vram_mb=peak_vram_mb,
    )

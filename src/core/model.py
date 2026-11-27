"""Decision Model module: Qwen backbone with reserved token one-pass forward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from peft import LoraConfig, PeftModel, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.prompts import (
    READOUT_TOKEN,
    get_reserved_option_tokens,
)


@dataclass
class DecisionModelOutput:
    """Output container for DecisionModel forward pass."""

    logits: torch.Tensor  # Shape: (batch_size, num_options)
    loss: torch.Tensor | None = None


def setup_decision_tokenizer(
    model_name_or_path: str,
    num_options: int = 77,
) -> tuple[PreTrainedTokenizerBase, list[int], int]:
    """Load tokenizer, register reserved option tokens and readout token.

    Returns:
        (tokenizer, option_token_ids, readout_token_id)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Crucial: left-padding ensures the last position (-1) is always <readout>
    tokenizer.padding_side = "left"

    reserved_tokens = get_reserved_option_tokens(num_options)
    special_tokens_to_add = [READOUT_TOKEN] + reserved_tokens

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})

    option_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in reserved_tokens]
    readout_token_id = tokenizer.convert_tokens_to_ids(READOUT_TOKEN)

    return tokenizer, option_token_ids, readout_token_id


def _select_distinct_reference_ids(
    tokenizer: PreTrainedTokenizerBase | None,
    vocab_size: int,
    count: int,
) -> list[int]:
    """Pick `count` distinct pretrained token IDs (letters, digits, words) as semantic anchors."""
    ref_ids: list[int] = []
    seen: set[int] = set()

    if tokenizer is not None:
        candidates = (
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
            + [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]
            + [str(i) for i in range(10, 300)]
        )
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for text in candidates:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                tid = ids[0]
                if tid != unk_id and 0 <= tid < vocab_size and tid not in seen:
                    seen.add(tid)
                    ref_ids.append(tid)
                    if len(ref_ids) >= count:
                        return ref_ids

    # Fallback: pick evenly spaced existing token IDs from the pretrained vocabulary
    step = max(1, vocab_size // max(1, count + 1))
    for idx in range(1, vocab_size):
        tid = (idx * step) % vocab_size
        if tid not in seen:
            seen.add(tid)
            ref_ids.append(tid)
            if len(ref_ids) >= count:
                break

    return ref_ids


def init_reserved_token_embeddings(
    base_model: PreTrainedModel,
    new_token_ids: Sequence[int],
    reference_token_ids: Sequence[int] | None = None,
    noise_std: float = 1e-3,
) -> None:
    """Initialize each newly added token from a DISTINCT reference token row + small noise.

    Following the Jev recipe: each reserved option token <opt_k> starts from its own
    distinct pretrained label/letter row (with aligned input embedding and output head row)
    so option tokens are immediately distinguishable by causal attention from step 0.
    """
    input_embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()

    with torch.no_grad():
        in_weights = input_embeddings.weight
        existing_limit = min(new_token_ids) if len(new_token_ids) > 0 else in_weights.size(0)
        existing_limit = max(1, existing_limit)

        if reference_token_ids is None or len(reference_token_ids) < len(new_token_ids):
            ref_list = _select_distinct_reference_ids(
                tokenizer=None,
                vocab_size=existing_limit,
                count=max(len(new_token_ids), 1),
            )
        else:
            ref_list = list(reference_token_ids)

        gen = torch.Generator(device=in_weights.device).manual_seed(42)

        for idx, tid in enumerate(new_token_ids):
            ref_id = ref_list[idx % len(ref_list)]
            base_in_row = in_weights[ref_id].detach().clone()
            noise_in = torch.randn(
                base_in_row.shape,
                generator=gen,
                device=base_in_row.device,
                dtype=base_in_row.dtype,
            ) * noise_std
            in_weights[tid] = base_in_row + noise_in

        if output_embeddings is not None and output_embeddings.weight is not in_weights:
            out_weights = output_embeddings.weight
            for idx, tid in enumerate(new_token_ids):
                ref_id = ref_list[idx % len(ref_list)]
                base_out_row = out_weights[ref_id].detach().clone()
                noise_out = torch.randn(
                    base_out_row.shape,
                    generator=gen,
                    device=base_out_row.device,
                    dtype=base_out_row.dtype,
                ) * noise_std
                out_weights[tid] = base_out_row + noise_out


class DecisionModel(nn.Module):
    """Single-pass (T=1) Decision Model wrapper around a Causal LM base_model.

    Freezes the 152k pretrained vocabulary embeddings so Qwen's language comprehension
    remains intact, and trains only the newly added reserved token rows (<opt_k>, <readout>)
    plus the LoRA adapters. Logits at <readout> are projected strictly onto the K reserved
    option vectors rather than computing the full 152k vocabulary projection across all tokens.
    """

    def __init__(
        self,
        base_model: PreTrainedModel | PeftModel,
        option_token_ids: Sequence[int],
        readout_token_id: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.readout_token_id = int(readout_token_id)

        opt_ids_list = [int(x) for x in option_token_ids]
        all_reserved_ids = opt_ids_list + [self.readout_token_id]

        self.register_buffer("option_token_ids", torch.tensor(opt_ids_list, dtype=torch.long))
        self.register_buffer("all_reserved_ids", torch.tensor(all_reserved_ids, dtype=torch.long))

        # Extract initial representations for reserved tokens from the resized embedding/head tables
        in_emb = self.base_model.get_input_embeddings()
        out_emb = self.base_model.get_output_embeddings()

        with torch.no_grad():
            init_in = in_emb.weight[self.all_reserved_ids].detach().clone().float()
            if out_emb is not None:
                init_out = out_emb.weight[self.option_token_ids].detach().clone().float()
            else:
                init_out = in_emb.weight[self.option_token_ids].detach().clone().float()

        # Trainable parameters ONLY for the K+1 reserved input tokens and K output option heads
        self.reserved_input_embeds = nn.Parameter(init_in)
        self.reserved_output_weight = nn.Parameter(init_out)

        # Keep the massive 152k pretrained vocabulary tables frozen so we don't corrupt
        # English token representations or pay 272M-param AdamW overhead per step.
        in_emb.weight.requires_grad_(False)
        if out_emb is not None:
            out_emb.weight.requires_grad_(False)

        # Map each reserved token ID to its index in `reserved_input_embeds` (0..K)
        min_res_id = min(all_reserved_ids)
        max_res_id = max(all_reserved_ids)
        self._min_reserved_id = min_res_id
        self._max_reserved_id = max_res_id
        lookup = torch.full((max_res_id - min_res_id + 1,), -1, dtype=torch.long)
        for local_idx, tok_id in enumerate(all_reserved_ids):
            lookup[tok_id - min_res_id] = local_idx
        self.register_buffer("_reserved_id_lookup", lookup)

        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def num_options(self) -> int:
        return len(self.option_token_ids)

    def sync_reserved_weights_to_base(self) -> None:
        """Copy learned reserved token embeddings back into the base embedding/head tables."""
        in_emb = self.base_model.get_input_embeddings()
        out_emb = self.base_model.get_output_embeddings()
        with torch.no_grad():
            in_emb.weight[self.all_reserved_ids] = self.reserved_input_embeds.to(in_emb.weight.dtype)
            if out_emb is not None and out_emb.weight is not in_emb.weight:
                out_emb.weight[self.option_token_ids] = self.reserved_output_weight.to(out_emb.weight.dtype)

    def _build_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Combine frozen pretrained token embeddings with trainable reserved token embeddings."""
        in_emb = self.base_model.get_input_embeddings()
        with torch.no_grad():
            base_embeds = in_emb(input_ids)

        # Locate positions belonging to reserved tokens (<opt_0>..<opt_K-1>, <readout>)
        in_range = (input_ids >= self._min_reserved_id) & (input_ids <= self._max_reserved_id)
        if not in_range.any():
            return base_embeds

        shifted = (input_ids - self._min_reserved_id).clamp(
            min=0, max=self._reserved_id_lookup.numel() - 1
        )
        local_indices = self._reserved_id_lookup[shifted]
        is_reserved = in_range & (local_indices >= 0)

        if not is_reserved.any():
            return base_embeds

        out_embeds = base_embeds.clone()
        matched_local = local_indices[is_reserved]
        out_embeds[is_reserved] = self.reserved_input_embeds[matched_local].to(out_embeds.dtype)
        return out_embeds

    def _get_inner_transformer(self) -> nn.Module | None:
        """Unwrap PeftModel / CausalLM to reach the inner transformer body (skipping 152k lm_head)."""
        candidate = self.base_model
        if isinstance(candidate, PeftModel):
            candidate = candidate.get_base_model()
        # Standard HF CausalLM architectures expose the body as `.model` (Qwen, Llama, Mistral)
        # or `.transformer` (GPT-2)
        if hasattr(candidate, "model") and isinstance(candidate.model, nn.Module):
            return candidate.model
        if hasattr(candidate, "transformer") and isinstance(candidate.transformer, nn.Module):
            return candidate.transformer
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        correct_option_idx: torch.Tensor | None = None,
    ) -> DecisionModelOutput:
        """Run single-pass prefill, extract representation at <readout> (-1), score K options."""
        inputs_embeds = self._build_inputs_embeds(input_ids)
        inner_body = self._get_inner_transformer()

        if inner_body is not None:
            body_out = inner_body(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            readout_hidden = body_out.last_hidden_state[:, -1, :]
        else:
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            readout_hidden = outputs.hidden_states[-1][:, -1, :]

        # Project only the <readout> state onto the K reserved option rows: (batch_size, K)
        logits = F.linear(
            readout_hidden.to(self.reserved_output_weight.dtype),
            self.reserved_output_weight,
        )

        loss = None
        if correct_option_idx is not None:
            loss = self.loss_fn(logits, correct_option_idx)

        return DecisionModelOutput(logits=logits, loss=loss)

    def save_checkpoint(self, checkpoint_dir: str | Path) -> Path:
        """Save LoRA adapter weights and learned reserved token parameters."""
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self.sync_reserved_weights_to_base()
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(ckpt_path)
        torch.save(
            {
                "reserved_input_embeds": self.reserved_input_embeds.detach().cpu(),
                "reserved_output_weight": self.reserved_output_weight.detach().cpu(),
            },
            ckpt_path / "reserved_tokens.pt",
        )
        return ckpt_path

    def load_reserved_tokens(self, checkpoint_dir: str | Path) -> None:
        """Restore learned reserved token parameters from a checkpoint directory."""
        res_file = Path(checkpoint_dir) / "reserved_tokens.pt"
        if res_file.exists():
            payload = torch.load(res_file, map_location="cpu", weights_only=True)
            with torch.no_grad():
                self.reserved_input_embeds.copy_(
                    payload["reserved_input_embeds"].to(
                        device=self.reserved_input_embeds.device,
                        dtype=self.reserved_input_embeds.dtype,
                    )
                )
                self.reserved_output_weight.copy_(
                    payload["reserved_output_weight"].to(
                        device=self.reserved_output_weight.device,
                        dtype=self.reserved_output_weight.dtype,
                    )
                )
            self.sync_reserved_weights_to_base()


def build_decision_model(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_options: int = 77,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    torch_dtype: torch.dtype = torch.float32,
    device_map: str | None = None,
) -> tuple[DecisionModel, PreTrainedTokenizerBase]:
    """Build a complete DecisionModel with tokenizer and optional LoRA adapters."""
    tokenizer, option_token_ids, readout_token_id = setup_decision_tokenizer(
        model_name_or_path=model_name_or_path,
        num_options=num_options,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    orig_vocab_size = base_model.get_input_embeddings().weight.size(0)
    base_model.resize_token_embeddings(len(tokenizer))

    new_token_ids = option_token_ids + [readout_token_id]
    reference_token_ids = _select_distinct_reference_ids(
        tokenizer=tokenizer,
        vocab_size=orig_vocab_size,
        count=len(new_token_ids),
    )

    # Initialize each reserved token from a DISTINCT existing label/token representation
    init_reserved_token_embeddings(
        base_model,
        new_token_ids,
        reference_token_ids=reference_token_ids,
    )

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        base_model = get_peft_model(base_model, lora_config)

    decision_model = DecisionModel(
        base_model=base_model,
        option_token_ids=option_token_ids,
        readout_token_id=readout_token_id,
    )

    return decision_model, tokenizer


def load_trained_decision_model(
    checkpoint_dir: str | Path,
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_options: int = 77,
    torch_dtype: torch.dtype = torch.float32,
    device_map: str | None = None,
) -> tuple[DecisionModel, PreTrainedTokenizerBase]:
    """Load a base model, attach trained LoRA checkpoint + reserved tokens, and return DecisionModel."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    tokenizer, option_token_ids, readout_token_id = setup_decision_tokenizer(
        model_name_or_path=model_name_or_path,
        num_options=num_options,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    base_model = PeftModel.from_pretrained(base_model, str(ckpt_path))

    decision_model = DecisionModel(
        base_model=base_model,
        option_token_ids=option_token_ids,
        readout_token_id=readout_token_id,
    )
    decision_model.load_reserved_tokens(ckpt_path)
    decision_model.eval()

    return decision_model, tokenizer

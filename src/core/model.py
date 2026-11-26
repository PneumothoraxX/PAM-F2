"""Decision Model module: Qwen backbone with reserved token one-pass forward."""

# Configuramos y cargamos al TOKENIZADOR [TokensEspeciales y <readout>]
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, PeftModel

from src.core.prompts import (
    MAX_RESERVED_OPTIONS,
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

    # Only add tokens that aren't already recognized as special
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    
    option_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in reserved_tokens]
    readout_token_id = tokenizer.convert_tokens_to_ids(READOUT_TOKEN)

    return tokenizer, option_token_ids, readout_token_id


def init_reserved_token_embeddings(
    model: PreTrainedModel,
    new_token_ids: Sequence[int],
    reference_token_ids: Sequence[int] | None = None,
    noise_std: float = 1e-4,
) -> None:
    """Initialize newly added token embeddings with average of reference tokens + noise.

    Following the Jev recipe: instead of random noise, initialize from existing
    letter/digit token representations so option tokens live in a sensible semantic space.
    """
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    with torch.no_grad():
        in_weights = input_embeddings.weight
        if reference_token_ids is None or len(reference_token_ids) == 0:
            # Fallback: sample first 50 common ASCII tokens
            ref_in = in_weights[:256].mean(dim=0, keepdim=True)
        else:
            ref_tensor = torch.tensor(reference_token_ids, device=in_weights.device)
            ref_in = in_weights[ref_tensor].mean(dim=0, keepdim=True)

        for tid in new_token_ids:
            noise = torch.randn_like(ref_in) * noise_std
            in_weights[tid] = (ref_in + noise).squeeze(0)

        # Output head (if separate from input embeddings)
        if output_embeddings is not None and output_embeddings.weight is not in_weights:
            out_weights = output_embeddings.weight
            if reference_token_ids is None or len(reference_token_ids) == 0:
                ref_out = out_weights[:256].mean(dim=0, keepdim=True)
            else:
                ref_tensor = torch.tensor(reference_token_ids, device=out_weights.device)
                ref_out = out_weights[ref_tensor].mean(dim=0, keepdim=True)

            for tid in new_token_ids:
                noise = torch.randn_like(ref_out) * noise_std
                out_weights[tid] = (ref_out + noise).squeeze(0)


class DecisionModel(nn.Module):
    """Single-pass (T=1) Decision Model wrapper around a Causal LM backbone."""

    def __init__(
        self,
        backbone: PreTrainedModel | PeftModel,
        option_token_ids: Sequence[int],
        readout_token_id: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Register option token IDs as buffer so they transfer with .to(device)
        self.register_buffer("option_token_ids", torch.tensor(option_token_ids, dtype=torch.long))
        self.readout_token_id = readout_token_id
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def num_options(self) -> int:
        return len(self.option_token_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_option_idx: torch.Tensor | None = None,
    ) -> DecisionModelOutput:
        """Run single-pass prefill, extract logits at last token (-1) restricted to options."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # No autoregressive decoding, no KV-cache retained!
            return_dict=True,
        )

        # Last token logits: (batch_size, vocab_size)
        last_token_logits = outputs.logits[:, -1, :]

        # Slice only option tokens: (batch_size, num_options)
        decision_logits = last_token_logits[:, self.option_token_ids]

        loss = None
        if target_option_idx is not None:
            loss = self.loss_fn(decision_logits, target_option_idx)

        return DecisionModelOutput(logits=decision_logits, loss=loss)


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

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Resize embedding matrices for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token representations from alphanumeric chars
    ref_chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ref_ids = [tokenizer.convert_tokens_to_ids(c) for c in ref_chars if tokenizer.convert_tokens_to_ids(c) != tokenizer.unk_token_id]
    all_new_ids = option_token_ids + [readout_token_id]
    init_reserved_token_embeddings(model, all_new_ids, reference_token_ids=ref_ids)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # CRITICAL: embed_tokens and lm_head MUST be saved/trained for new tokens!
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, lora_config)

    decision_model = DecisionModel(
        backbone=model,
        option_token_ids=option_token_ids,
        readout_token_id=readout_token_id,
    )

    return decision_model, tokenizer

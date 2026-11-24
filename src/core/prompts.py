"""Reserved option tokens and single-pass prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

MAX_RESERVED_OPTIONS = 255  # <opt_0> .. <opt_254>
READOUT_TOKEN = "<readout>"


def option_token(slot_idx: int) -> str:
    """Return the reserved token string for a given option slot index."""
    if not 0 <= slot_idx < MAX_RESERVED_OPTIONS:
        raise ValueError(
            f"slot_idx must be in [0, {MAX_RESERVED_OPTIONS - 1}], got {slot_idx}"
        )
    return f"<opt_{slot_idx}>"


def get_reserved_option_tokens(num_options: int = MAX_RESERVED_OPTIONS) -> list[str]:
    """Return the ordered list of reserved option tokens up to num_options."""
    if not 1 <= num_options <= MAX_RESERVED_OPTIONS:
        raise ValueError(
            f"num_options must be in [1, {MAX_RESERVED_OPTIONS}], got {num_options}"
        )
    return [option_token(i) for i in range(num_options)]


def format_option_label(label: str, humanize: bool = True) -> str:
    """Format a raw class identifier for prompt presentation."""
    return label.replace("_", " ") if humanize else label


@dataclass(frozen=True)
class DecisionExample:
    """Single formatted decision-model prompt and slot-to-class mapping."""

    prompt: str
    target_option_idx: int | None
    target_token: str | None
    option_to_class: list[int]
    class_to_option: dict[int, int]


def format_decision_prompt(
    query: str,
    options: Sequence[str],
    correct_idx: int | None = None,
    shuffle_options: bool = True,
    rng: random.Random | None = None,
    humanize_labels: bool = True,
) -> DecisionExample:
    """Format a closed-set classification query into a one-pass decision prompt.

    Each reserved <opt_i> token is placed immediately *after* its option text so
    left-to-right causal attention lets the option token attend to its full label,
    and the prompt terminates with <readout> for single-position logit extraction.
    """
    num_options = len(options)
    if not 1 <= num_options <= MAX_RESERVED_OPTIONS:
        raise ValueError(
            f"Number of options must be in [1, {MAX_RESERVED_OPTIONS}], got {num_options}"
        )
    if correct_idx is not None and not 0 <= correct_idx < num_options:
        raise ValueError(
            f"correct_idx must be in [0, {num_options - 1}], got {correct_idx}"
        )

    perm = list(range(num_options))
    if shuffle_options:
        shuffler = rng if rng is not None else random.Random()
        shuffler.shuffle(perm)

    class_to_option = {orig_idx: slot_idx for slot_idx, orig_idx in enumerate(perm)}

    lines = [
        "Select the intent that best matches the user query.",
        f"Query: {query.strip()}",
        "Options:",
    ]
    for slot_idx, orig_idx in enumerate(perm):
        display_text = format_option_label(options[orig_idx], humanize=humanize_labels)
        lines.append(f"- {display_text} {option_token(slot_idx)}")

    lines.append(READOUT_TOKEN)
    prompt = "\n".join(lines)

    target_option_idx = class_to_option[correct_idx] if correct_idx is not None else None
    target_token = (
        option_token(target_option_idx) if target_option_idx is not None else None
    )

    return DecisionExample(
        prompt=prompt,
        target_option_idx=target_option_idx,
        target_token=target_token,
        option_to_class=perm,
        class_to_option=class_to_option,
    )

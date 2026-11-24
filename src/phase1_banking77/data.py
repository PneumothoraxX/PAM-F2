"""Stratified dataset splits and PyTorch wrapper for BANKING77 (Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from src.core.prompts import DecisionExample, format_decision_prompt


@dataclass(frozen=True)
class Banking77Splits:
    """Container for stratified BANKING77 splits and class metadata."""

    train: Dataset
    val: Dataset
    test: Dataset
    label_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def load_banking77_splits(
    val_ratio: float = 0.2,
    seed: int = 42,
    dataset_name: str = "PolyAI/banking77",
) -> Banking77Splits:
    """Load BANKING77 and carve a stratified validation split from train.

    The validation split stays untouched during LoRA training so post-hoc
    temperature scaling (L-BFGS) runs on independent held-out samples.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be strictly between 0 and 1, got {val_ratio}")

    raw = load_dataset(dataset_name)
    label_feature = raw["train"].features["label"]
    label_names = list(label_feature.names)

    split = raw["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    return Banking77Splits(
        train=split["train"],
        val=split["test"],
        test=raw["test"],
        label_names=label_names,
    )


class Banking77DecisionDataset(TorchDataset[DecisionExample]):
    """PyTorch dataset wrapper emitting single-pass decision prompts.

    Uses a deterministic per-(seed, epoch, index) RNG when shuffling options so
    DataLoader workers remain reproducible while preventing positional overfitting
    across epochs.
    """

    def __init__(
        self,
        hf_split: Dataset,
        label_names: Sequence[str],
        shuffle_options: bool = True,
        seed: int = 42,
        humanize_labels: bool = True,
    ) -> None:
        self.hf_split = hf_split
        self.label_names = list(label_names)
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.humanize_labels = humanize_labels
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter to refresh option shuffles cleanly."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> DecisionExample:
        row = self.hf_split[idx]
        # Mixing epoch and idx with a large prime keeps shuffles disjoint per epoch
        # even under multi-worker DataLoaders.
        rng = (
            random.Random(self.seed + self.epoch * 1_000_003 + idx)
            if self.shuffle_options
            else None
        )
        return format_decision_prompt(
            query=row["text"],
            options=self.label_names,
            correct_idx=int(row["label"]),
            shuffle_options=self.shuffle_options,
            rng=rng,
            humanize_labels=self.humanize_labels,
        )


def _preview() -> None:
    splits = load_banking77_splits(val_ratio=0.2, seed=42)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    print(f"Dataset BANKING77 cargado ({splits.num_classes} clases, {total} consultas)")
    print(
        f"Splits -> train: {len(splits.train)}, "
        f"val: {len(splits.val)}, test: {len(splits.test)}"
    )

    train_ds = Banking77DecisionDataset(
        splits.train,
        splits.label_names,
        shuffle_options=True,
        seed=42,
    )
    sample_row = splits.train[0]
    example = train_ds[0]

    assert example.target_option_idx is not None
    mapped_class = example.option_to_class[example.target_option_idx]
    assert mapped_class == sample_row["label"]

    prompt_lines = example.prompt.splitlines()
    print("\n--- Ejemplo formateado (resumen) ---")
    print("\n".join(prompt_lines[:8]))
    print(f"... ({len(splits.label_names) - 5} opciones intermedias) ...")
    print("\n".join(prompt_lines[-3:]))
    print(
        f"\nClase real: {sample_row['label']} ({splits.label_names[sample_row['label']]})"
        f" -> Slot asignado: {example.target_option_idx} ({example.target_token})"
    )


if __name__ == "__main__":
    _preview()

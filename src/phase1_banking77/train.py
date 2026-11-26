"""Training pipeline for Level-1 Decision Model on BANKING77 (Phase 1)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from src.core.model import DecisionModel, build_decision_model
from src.core.prompts import DecisionExample
from src.phase1_banking77.data import (
    Banking77DecisionDataset,
    Banking77Splits,
    load_banking77_splits,
)


@dataclass
class DecisionBatch:
    """Colocated batch tensors for DecisionModel forward pass."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_option_idx: torch.Tensor
    option_to_class: list[list[int]]


class DecisionDataCollator:
    """Collates a list of DecisionExample objects into padded tensor batches.

    Padding is placed on the left side so that the final position (-1)
    consistently holds the <readout> token across all batch elements.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[DecisionExample]) -> DecisionBatch:
        prompts = [ex.prompt for ex in batch]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        targets = [ex.target_option_idx if ex.target_option_idx is not None else -1 for ex in batch]
        target_tensor = torch.tensor(targets, dtype=torch.long)
        option_to_class = [ex.option_to_class for ex in batch]

        return DecisionBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            target_option_idx=target_tensor,
            option_to_class=option_to_class,
        )


def evaluate_split(
    model: DecisionModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Evaluate loss, accuracy, and collect raw logits + true labels.

    Returns:
        (avg_loss, accuracy, all_canonical_logits, all_true_classes)
        all_canonical_logits are remapped to standard 0..K-1 class space.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    collected_canonical_logits = []
    collected_true_classes = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            targets = batch.target_option_idx.to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_option_idx=targets if (targets >= 0).all() else None,
            )

            batch_size = input_ids.size(0)
            if out.loss is not None:
                total_loss += out.loss.item() * batch_size
            total_samples += batch_size

            logits_cpu = out.logits.cpu()
            preds = logits_cpu.argmax(dim=-1)
            correct += (preds == batch.target_option_idx).sum().item()

            # Remap logits from shuffled slot order to canonical class indices (0..76)
            num_classes = model.num_options
            canonical_logits = torch.empty((batch_size, num_classes), dtype=torch.float32)
            true_classes = torch.empty(batch_size, dtype=torch.long)

            for b in range(batch_size):
                opt_to_cls = batch.option_to_class[b]
                for slot_idx, cls_idx in enumerate(opt_to_cls):
                    canonical_logits[b, cls_idx] = logits_cpu[b, slot_idx]
                true_slot = batch.target_option_idx[b].item()
                true_classes[b] = opt_to_cls[true_slot] if true_slot >= 0 else -1

            collected_canonical_logits.append(canonical_logits)
            collected_true_classes.append(true_classes)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    all_logits = torch.cat(collected_canonical_logits, dim=0)
    all_labels = torch.cat(collected_true_classes, dim=0)

    return avg_loss, accuracy, all_logits, all_labels


def train_banking77(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "results/phase1_banking77",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 3e-4,
    max_length: int = 512,
    device: str | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    limit_train_samples: int | None = None,
) -> dict[str, float | str]:
    """Train LoRA on BANKING77 and export raw val/test logits for L-BFGS calibration."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"[*] Usando dispositivo: {device_obj}")

    # 1. Dataset splits
    splits: Banking77Splits = load_banking77_splits(val_ratio=val_ratio, seed=seed)
    train_hf = splits.train
    if limit_train_samples is not None:
        train_hf = train_hf.select(range(min(limit_train_samples, len(train_hf))))

    train_ds = Banking77DecisionDataset(train_hf, splits.label_names, shuffle_options=True, seed=seed)
    # Val and test evaluated canonically (no shuffle needed)
    val_ds = Banking77DecisionDataset(splits.val, splits.label_names, shuffle_options=False, seed=seed)
    test_ds = Banking77DecisionDataset(splits.test, splits.label_names, shuffle_options=False, seed=seed)

    # 2. Build model and tokenizer
    print(f"[*] Cargando y adaptando modelo: {model_name_or_path}")
    model, tokenizer = build_decision_model(
        model_name_or_path=model_name_or_path,
        num_options=splits.num_classes,
        use_lora=True,
    )
    model.to(device_obj)

    collator = DecisionDataCollator(tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # 3. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 4. Training loop
    print(f"[*] Iniciando entrenamiento ({epochs} épocas, {len(train_ds)} ejemplos de train)...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Época {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch.input_ids.to(device_obj)
            attention_mask = batch.attention_mask.to(device_obj)
            targets = batch.target_option_idx.to(device_obj)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_option_idx=targets,
            )

            loss = out.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        elapsed = time.time() - start_time
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate_split(model, val_loader, device_obj)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Tiempo: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            if hasattr(model.backbone, "save_pretrained"):
                model.backbone.save_pretrained(out_path / "best_lora_checkpoint")

    # 5. Export val and test logits for Phase 1 calibration & evaluation
    print("[*] Exportando logits crudos para calibración L-BFGS y evaluación final...")
    val_loss, val_acc, val_logits, val_labels = evaluate_split(model, val_loader, device_obj)
    test_loss, test_acc, test_logits, test_labels = evaluate_split(model, test_loader, device_obj)

    torch.save({"logits": val_logits, "labels": val_labels}, out_path / "val_logits.pt")
    torch.save({"logits": test_logits, "labels": test_labels}, out_path / "test_logits.pt")

    print(f"[+] Archivos guardados en: {out_path}")
    print(f"[+] Resultados finales | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    return {
        "best_val_acc": best_val_acc,
        "final_test_acc": test_acc,
        "output_dir": str(out_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar DecisionModel sobre BANKING77")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=str, default="results/phase1_banking77")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    train_banking77(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit_train_samples=args.limit_samples,
    )

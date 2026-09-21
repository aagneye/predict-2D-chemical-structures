"""De novo decoder training loop.

Intended for the Azure box, not the Kaggle submission notebook (inference
only, no internet). Trains :class:`casmi.models.denovo.FingerprintToSmilesDecoder`
with teacher-forced cross-entropy over SMILES tokens, conditioned on each
training structure's own true fingerprint.

**Training rows must come from the split's train side only**, for the same
leakage reason :mod:`casmi.models.train` enforces it for FPNet — this dataset's
compounds recur across all 11 libraries, and training on a held-out structure
invalidates every validation number downstream.

**Bit-flip augmentation on the input fingerprint** narrows the train/inference
mismatch: at inference this decoder is fed FPNet's *predicted* (noisy) logits,
not a molecule's true fingerprint, so training exclusively on clean
fingerprints would teach the decoder to expect input it will never actually
see. Randomly flipping a small fraction of bits (default 2%) during training
is a cheap proxy for that noise.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from casmi.models.denovo import (
    DenovoConfig,
    FingerprintToSmilesDecoder,
    fingerprint_to_bit_sequence,
)
from casmi.models.smiles_tokenizer import SmilesVocabulary


@dataclass
class DenovoTrainingConfig:
    """Optimisation settings."""

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 15_000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1_000
    checkpoint_every: int = 2_500
    #: Fraction of fingerprint bits randomly flipped during training, to
    #: narrow the train/inference mismatch (true fingerprint at train time,
    #: FPNet-predicted logits at inference time).
    bit_flip_rate: float = 0.02
    use_amp: bool = True
    seed: int = 0
    num_workers: int = 0


@dataclass
class DenovoTrainingRow:
    """One training example: a structure's fingerprint and tokenized SMILES."""

    fingerprint: np.ndarray
    token_ids: list[int]


class DenovoDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[DenovoTrainingRow]) -> None:
        if not rows:
            raise ValueError("dataset is empty")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DenovoTrainingRow:
        return self.rows[index]


def collate_denovo_batch(
    batch: list[DenovoTrainingRow],
    pad_index: int,
    max_fingerprint_bits: int,
    bit_flip_rate: float = 0.0,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate into ``(bit_indices, bit_padding_mask, input_ids, target_ids)``.

    ``bit_indices``/``bit_padding_mask`` are the Transformer encoder's input
    (see :func:`casmi.models.denovo.fingerprint_to_bit_sequence`) — a batch of
    variable-length set-bit-index sequences, right-padded to the batch max
    and masked, rather than the dense fingerprint vectors the old GRU decoder
    consumed directly.

    ``input_ids`` is every token except the last; ``target_ids`` is every
    token except the first — the standard shift-by-one teacher-forcing setup.
    """
    bit_sequences: list[np.ndarray] = []
    generator = rng or np.random.default_rng()
    for row in batch:
        fingerprint = row.fingerprint
        if bit_flip_rate > 0:
            flips = generator.random(fingerprint.shape) < bit_flip_rate
            fingerprint = np.where(flips, 1.0 - fingerprint, fingerprint).astype(np.float32)
        bit_sequences.append(
            fingerprint_to_bit_sequence(fingerprint, max_bits=max_fingerprint_bits, threshold=0.5)
        )

    max_bits_len = max((seq.size for seq in bit_sequences), default=1)
    max_bits_len = max(max_bits_len, 1)
    bit_indices = np.zeros((len(batch), max_bits_len), dtype=np.int64)
    bit_padding_mask = np.ones((len(batch), max_bits_len), dtype=bool)
    for i, seq in enumerate(bit_sequences):
        if seq.size:
            bit_indices[i, : seq.size] = seq
            bit_padding_mask[i, : seq.size] = False

    max_len = max(len(row.token_ids) for row in batch)
    input_ids = np.full((len(batch), max_len - 1), pad_index, dtype=np.int64)
    target_ids = np.full((len(batch), max_len - 1), pad_index, dtype=np.int64)
    for i, row in enumerate(batch):
        ids = row.token_ids
        input_ids[i, : len(ids) - 1] = ids[:-1]
        target_ids[i, : len(ids) - 1] = ids[1:]

    def tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device)

    return (
        tensor(bit_indices, torch.long),
        tensor(bit_padding_mask, torch.bool),
        tensor(input_ids, torch.long),
        tensor(target_ids, torch.long),
    )


def build_denovo_training_rows(
    smiles_list: list[str],
    fingerprint_of_smiles,
    vocabulary: SmilesVocabulary,
    max_length: int,
    max_rows: int | None = None,
) -> list[DenovoTrainingRow]:
    """Assemble training rows from a list of (train-side only) SMILES.

    Args:
        smiles_list: One SMILES per unique training structure.
        fingerprint_of_smiles: Callable ``smiles -> fingerprint or None``.
        vocabulary: Built beforehand from the same corpus (or a superset).
        max_length: Passed through to :meth:`SmilesVocabulary.encode`.
        max_rows: Optional cap, useful for smoke tests.

    Returns:
        Rows for every structure that both fingerprints and tokenizes.
    """
    rows: list[DenovoTrainingRow] = []
    for smiles in smiles_list:
        if max_rows is not None and len(rows) >= max_rows:
            break
        fingerprint = fingerprint_of_smiles(smiles)
        if fingerprint is None:
            continue
        token_ids = vocabulary.encode(smiles, max_length=max_length)
        if len(token_ids) < 2:
            continue
        rows.append(
            DenovoTrainingRow(
                fingerprint=np.asarray(fingerprint, dtype=np.float32), token_ids=token_ids
            )
        )
    return rows


def _lr_at_step(step: int, config: DenovoTrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_denovo(
    model: nn.Module,
    rows: list[DenovoTrainingRow],
    pad_index: int,
    device: str,
    max_fingerprint_bits: int,
    batch_size: int = 128,
) -> float:
    """Mean per-token cross-entropy loss on held-out rows (lower is better)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        if not chunk:
            continue
        bit_indices, bit_padding_mask, input_ids, target_ids = collate_denovo_batch(
            chunk,
            pad_index=pad_index,
            max_fingerprint_bits=max_fingerprint_bits,
            bit_flip_rate=0.0,
            device=device,
        )
        token_padding_mask = input_ids == pad_index
        logits = model(bit_indices, bit_padding_mask, input_ids, token_padding_mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        n_tokens = int((target_ids != pad_index).sum().item())
        total_loss += float(loss.item())
        total_tokens += n_tokens
    return total_loss / total_tokens if total_tokens else float("inf")


def train_denovo(
    rows: list[DenovoTrainingRow],
    model_config: DenovoConfig,
    vocabulary: SmilesVocabulary,
    training_config: DenovoTrainingConfig | None = None,
    device: str | None = None,
    output_dir: str | Path = "checkpoints",
    validation_rows: list[DenovoTrainingRow] | None = None,
) -> dict:
    """Train the de novo decoder and write checkpoints."""
    cfg = training_config or DenovoTrainingConfig()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    dataset = DenovoDataset(rows)
    pad_index = vocabulary.pad_index
    max_fingerprint_bits = model_config.max_fingerprint_bits
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_denovo_batch(
            b,
            pad_index=pad_index,
            max_fingerprint_bits=max_fingerprint_bits,
            bit_flip_rate=cfg.bit_flip_rate,
            device=resolved_device,
            rng=rng,
        ),
        drop_last=True,
    )

    model = FingerprintToSmilesDecoder(model_config).to(resolved_device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    use_amp = cfg.use_amp and resolved_device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list] = {"step": [], "loss": [], "val_loss": []}
    step = 0
    started = time.time()
    model.train()

    while step < cfg.max_steps:
        for bit_indices, bit_padding_mask, input_ids, target_ids in loader:
            if step >= cfg.max_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = _lr_at_step(step, cfg)

            optimizer.zero_grad(set_to_none=True)
            token_padding_mask = input_ids == pad_index
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(bit_indices, bit_padding_mask, input_ids, token_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"step {step:>6}  loss {loss.item():.4f}  "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}  {elapsed:.0f}s",
                    flush=True,
                )
                history["step"].append(step)
                history["loss"].append(float(loss.item()))

            if validation_rows and cfg.eval_every and step and step % cfg.eval_every == 0:
                val_loss = evaluate_denovo(
                    model, validation_rows, pad_index, resolved_device, max_fingerprint_bits
                )
                print(f"  [val] loss {val_loss:.4f}", flush=True)
                history["val_loss"].append((step, val_loss))
                model.train()

            if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
                save_denovo_checkpoint(
                    model, model_config, vocabulary, step, destination / f"denovo_{step}.pt"
                )

            step += 1

    save_denovo_checkpoint(model, model_config, vocabulary, step, destination / "denovo_final.pt")
    with open(destination / "denovo_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history


def save_denovo_checkpoint(
    model: nn.Module,
    model_config: DenovoConfig,
    vocabulary: SmilesVocabulary,
    step: int,
    path: str | Path,
) -> None:
    """Save weights, config, and vocabulary together.

    The vocabulary travels with the checkpoint (like FPNet's config) because
    inference happens in a separate no-internet notebook where rebuilding it
    from the training corpus is not possible.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model_config),
            "vocabulary": vocabulary.to_dict(),
            "step": step,
        },
        str(path),
    )


def load_denovo_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[FingerprintToSmilesDecoder, DenovoConfig, SmilesVocabulary]:
    """Load a checkpoint saved by :func:`save_denovo_checkpoint`."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    config = DenovoConfig(**payload["config"])
    model = FingerprintToSmilesDecoder(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    vocabulary = SmilesVocabulary.from_dict(payload["vocabulary"])
    return model, config, vocabulary

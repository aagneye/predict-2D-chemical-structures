"""FPNet training loop.

Intended for the Azure 4x T4 box, not the Kaggle submission notebook (which is
inference-only and has no internet). Trains one spectrum -> fingerprint model
with multi-label binary cross-entropy over fingerprint bits.

Two training details are specific to this dataset:

**Training rows must come from the split's train side only.** The dataset's
compounds recur across all 11 libraries, so a model trained on unfiltered
``train.parquet`` has seen the held-out structures and every validation number
downstream is inflated. :func:`build_training_rows` takes an explicit row mask
for that reason.

**Bits are weighted by inverse frequency, capped.** Most fingerprint bits are
almost always zero; unweighted BCE converges to predicting all-zero, which is
useless for ranking. Capping the weight prevents the rarest bits (set in a
handful of molecules) from dominating the gradient.
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

from casmi.models.fpnet import FPNet, FPNetConfig, collate_spectra, prepare_peaks


@dataclass
class TrainingConfig:
    """Optimisation settings."""

    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 20_000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1_000
    checkpoint_every: int = 2_000
    #: Upper bound on inverse-frequency bit weights.
    max_bit_weight: float = 20.0
    #: Mixed precision. T4s have tensor cores, so this is a large speedup.
    use_amp: bool = True
    seed: int = 0
    num_workers: int = 0


@dataclass
class TrainingRow:
    """One training example: a prepared spectrum plus its target fingerprint."""

    mz: np.ndarray
    intensity: np.ndarray
    precursor_mz: float
    adduct: str
    instrument: str | None
    collision_energy: float
    ionization_mode: str
    fingerprint: np.ndarray


class SpectrumFingerprintDataset(torch.utils.data.Dataset):
    """In-memory dataset of prepared spectra and target fingerprints."""

    def __init__(self, rows: list[TrainingRow]) -> None:
        if not rows:
            raise ValueError("dataset is empty")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingRow:
        return self.rows[index]


def collate_training_batch(
    batch: list[TrainingRow], device: torch.device | str = "cpu"
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate a batch into ``(model_inputs, targets)``."""
    inputs = collate_spectra(
        prepared=[(row.mz, row.intensity) for row in batch],
        precursor_mz=[row.precursor_mz for row in batch],
        adducts=[row.adduct for row in batch],
        instruments=[row.instrument for row in batch],
        collision_energies=[row.collision_energy for row in batch],
        ionization_modes=[row.ionization_mode for row in batch],
        device=device,
    )
    targets = torch.as_tensor(
        np.stack([row.fingerprint for row in batch]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    return inputs, targets


def build_training_rows(
    table,
    fingerprint_of_key,
    model_config: FPNetConfig,
    row_mask: np.ndarray | None = None,
    max_rows: int | None = None,
) -> list[TrainingRow]:
    """Assemble training rows from a train-shaped Arrow table.

    Args:
        table: Arrow table with train.parquet's columns.
        fingerprint_of_key: Callable ``inchikey14 -> fingerprint or None``.
            Passing a precomputed mapping avoids re-fingerprinting the same
            structure for each of its spectra.
        model_config: Used for peak preparation.
        row_mask: Boolean mask selecting the split's train side. **Omitting
            this trains on held-out structures and invalidates validation.**
        max_rows: Optional cap, useful for smoke tests.

    Returns:
        Rows whose structure has a usable fingerprint and whose spectrum has
        at least one peak after preparation.
    """
    frame = table.to_pandas()
    if row_mask is not None:
        mask = np.asarray(row_mask, dtype=bool)
        if mask.size != len(frame):
            raise ValueError(f"row_mask size {mask.size} != table rows {len(frame)}")
        frame = frame[mask]
    if max_rows is not None:
        frame = frame.iloc[:max_rows]

    rows: list[TrainingRow] = []
    for record in frame.itertuples():
        key = getattr(record, "inchikey14", None)
        if not key:
            continue
        fingerprint = fingerprint_of_key(key)
        if fingerprint is None:
            continue
        precursor = float(getattr(record, "precursor_mz", float("nan")))
        if not np.isfinite(precursor):
            continue
        mz, intensity = prepare_peaks(
            np.asarray(record.ms2_mzs, dtype=np.float64),
            np.asarray(record.ms2_normalized_intensities, dtype=np.float64),
            precursor,
            model_config,
        )
        if mz.size == 0:
            continue

        energy = getattr(record, "collision_energy_ev", None)
        if energy is None:
            energy_value = float("nan")
        else:
            arr = np.atleast_1d(np.asarray(energy, dtype=np.float64))
            arr = arr[np.isfinite(arr)]
            energy_value = float(arr.mean()) if arr.size else float("nan")

        rows.append(
            TrainingRow(
                mz=mz,
                intensity=intensity,
                precursor_mz=precursor,
                adduct=str(getattr(record, "adduct", "") or ""),
                instrument=getattr(record, "instrument_type", None),
                collision_energy=energy_value,
                ionization_mode=str(getattr(record, "ionization_mode", "positive") or "positive"),
                fingerprint=np.asarray(fingerprint, dtype=np.uint8),
            )
        )
    return rows


def compute_bit_weights(
    fingerprints: np.ndarray, max_weight: float = 20.0
) -> np.ndarray:
    """Inverse-frequency positive-class weights per fingerprint bit.

    Most bits are rarely set, so unweighted BCE is minimised by predicting
    zero everywhere. Weights are ``(1 - f) / f`` clipped to ``[1, max_weight]``.

    The floor of 1.0 means bits present in half or more of the corpus all
    receive weight exactly 1.0: the intent is to *boost* rare, informative bits,
    never to suppress common ones below baseline. The cap prevents bits set in a
    handful of molecules from dominating the gradient and destabilising training.
    """
    stacked = np.asarray(fingerprints, dtype=np.float32)
    if stacked.ndim != 2:
        raise ValueError("expected a 2-D fingerprint stack")
    frequency = stacked.mean(axis=0)
    weights = np.where(frequency > 0, (1.0 - frequency) / np.maximum(frequency, 1e-6), 1.0)
    return np.clip(weights, 1.0, max_weight).astype(np.float32)


def _lr_at_step(step: int, config: TrainingConfig) -> float:
    """Linear warmup then cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def cosine_similarity_metric(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean cosine similarity between predicted probabilities and targets.

    Reported alongside loss because it tracks what the model is actually used
    for — agreement between a predicted fingerprint and a real one — whereas
    weighted BCE is not comparable across bit-weight settings.
    """
    probabilities = torch.sigmoid(logits)
    return float(
        torch.nn.functional.cosine_similarity(probabilities, targets, dim=1).mean().item()
    )


def train_fpnet(
    rows: list[TrainingRow],
    model_config: FPNetConfig,
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    output_dir: str | Path = "checkpoints",
    validation_rows: list[TrainingRow] | None = None,
    multi_gpu: bool = True,
) -> dict:
    """Train FPNet and write checkpoints.

    Args:
        rows: Training examples from the split's train side only.
        model_config: Architecture; ``n_bits`` must match the fingerprints.
        training_config: Optimisation settings.
        device: Defaults to CUDA when available.
        output_dir: Checkpoint destination.
        validation_rows: Optional held-out rows for periodic evaluation.
        multi_gpu: Wrap in ``DataParallel`` when several GPUs are visible. The
            Azure box has 4x T4, and batches here are large enough for the
            split to pay off.

    Returns:
        A history dict with step-wise loss and metric traces.
    """
    cfg = training_config or TrainingConfig()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    dataset = SpectrumFingerprintDataset(rows)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_training_batch(b, device=resolved_device),
        drop_last=True,
    )

    model = FPNet(model_config).to(resolved_device)
    n_gpus = torch.cuda.device_count() if resolved_device.startswith("cuda") else 0
    if multi_gpu and n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[INFO] DataParallel across {n_gpus} GPUs", flush=True)

    bit_weights = torch.as_tensor(
        compute_bit_weights(
            np.stack([r.fingerprint for r in rows[: min(len(rows), 20_000)]]),
            max_weight=cfg.max_bit_weight,
        ),
        device=resolved_device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=bit_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    use_amp = cfg.use_amp and resolved_device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list] = {"step": [], "loss": [], "val_cosine": []}
    step = 0
    started = time.time()
    model.train()

    while step < cfg.max_steps:
        for inputs, targets in loader:
            if step >= cfg.max_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = _lr_at_step(step, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**inputs)
                loss = criterion(logits, targets)

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
                metric = evaluate_fpnet(model, validation_rows, resolved_device, cfg.batch_size)
                print(f"  [val] cosine similarity {metric:.4f}", flush=True)
                history["val_cosine"].append((step, metric))
                model.train()

            if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
                save_checkpoint(model, model_config, step, destination / f"fpnet_{step}.pt")

            step += 1

    save_checkpoint(model, model_config, step, destination / "fpnet_final.pt")
    with open(destination / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history


@torch.no_grad()
def evaluate_fpnet(
    model: nn.Module,
    rows: list[TrainingRow],
    device: str,
    batch_size: int = 64,
) -> float:
    """Mean cosine similarity between predicted and true fingerprints."""
    model.eval()
    totals, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        if not chunk:
            continue
        inputs, targets = collate_training_batch(chunk, device=device)
        logits = model(**inputs)
        totals += cosine_similarity_metric(logits, targets) * len(chunk)
        count += len(chunk)
    return totals / count if count else 0.0


def save_checkpoint(
    model: nn.Module, model_config: FPNetConfig, step: int, path: str | Path
) -> None:
    """Save weights plus the config needed to rebuild the model.

    The config travels with the weights because inference happens in a separate
    no-internet Kaggle notebook, where a mismatched architecture would fail
    obscurely at load time.
    """
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"model": state, "config": asdict(model_config), "step": step}, str(path))


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[FPNet, FPNetConfig]:
    """Load a checkpoint saved by :func:`save_checkpoint`."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    config = FPNetConfig(**payload["config"])
    model = FPNet(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


@torch.no_grad()
def predict_logits(
    model: FPNet,
    spectra: list[tuple[np.ndarray, np.ndarray]],
    precursor_mz: list[float],
    adducts: list[str],
    instruments: list[str | None],
    collision_energies: list[float],
    ionization_modes: list[str],
    model_config: FPNetConfig,
    device: str = "cpu",
) -> np.ndarray:
    """Predict one fingerprint-logit vector for a molecule's spectra.

    Logits are averaged across the molecule's spectra, since predictions are
    made per molecule and each spectrum is an independent noisy observation of
    the same structure.
    """
    prepared = [
        prepare_peaks(mz, intensity, precursor, model_config)
        for (mz, intensity), precursor in zip(spectra, precursor_mz, strict=True)
    ]
    keep = [i for i, (mz, _) in enumerate(prepared) if mz.size > 0]
    if not keep:
        return np.zeros(model_config.n_bits, dtype=np.float32)

    inputs = collate_spectra(
        prepared=[prepared[i] for i in keep],
        precursor_mz=[precursor_mz[i] for i in keep],
        adducts=[adducts[i] for i in keep],
        instruments=[instruments[i] for i in keep],
        collision_energies=[collision_energies[i] for i in keep],
        ionization_modes=[ionization_modes[i] for i in keep],
        device=device,
    )
    logits = model(**inputs)
    return logits.float().mean(dim=0).cpu().numpy()

"""FPNet — spectrum to molecular fingerprint transformer.

Predicts per-bit logits over a molecular fingerprint directly from a spectrum.
Candidates are then ranked by how well their fingerprints agree with the
prediction, which sidesteps the combinatorial difficulty of generating a
structure outright.

Two design choices carry most of the value:

**Peaks are embedded by both m/z and neutral loss.** Neutral loss
(``precursor - m/z``) is the chemically interpretable quantity: a loss of
18.011 Da means water left the molecule regardless of the precursor's absolute
mass. Giving the model both views lets it learn loss-based rules that transfer
across molecules of different sizes, which matters here because the training
libraries and the test set occupy different regions of chemical space.

**Scoring is a single dot product.** Treating fingerprint bits as independent
Bernoulli variables makes a candidate's log-likelihood
``sum_i [f_i * z_i - log(1 + exp(z_i))]``. The second term does not depend on
the candidate, so ranking by ``f . z`` is exactly equivalent to ranking by
log-likelihood — one matmul scores an entire candidate pool. See
:func:`score_candidates`.

This module imports torch and is therefore optional; install the ``train``
extra. The baseline pipeline does not depend on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: Adduct vocabulary. ``<unk>`` absorbs the long tail of train-only spellings.
ADDUCT_VOCAB: tuple[str, ...] = (
    "[M+H]+",
    "[M+NH4]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+C2H4O2-H]-",
    "[M+Cl]-",
    "[M]-",
    "[M+2H]2+",
    "[M-2H]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M+NH4]+",
    "[2M-H]-",
    "[2M+K]+",
    "[M+Na-2H]-",
    "[M-H2O]+",
    "<unk>",
)
ADDUCT_INDEX = {name: i for i, name in enumerate(ADDUCT_VOCAB)}

#: Instrument families. train's ``instrument_type`` is free text with dozens of
#: spellings, so it is bucketed rather than embedded verbatim.
INSTRUMENT_FAMILIES: tuple[str, ...] = ("timsTOF", "Orbitrap", "QTOF", "IonTrap", "other")


def instrument_family(value: str | None) -> int:
    """Bucket a free-text instrument string into a family index."""
    if value is None:
        return 4
    text = str(value).lower()
    if "timstof" in text:
        return 0
    if any(t in text for t in ("orbitrap", "qft", "ftms", "exactive", "itft", "hybrid ft")):
        return 1
    if "tof" in text:
        return 2
    if "trap" in text or "qq" in text:
        return 3
    return 4


@dataclass
class FPNetConfig:
    """FPNet architecture and input-shaping parameters."""

    n_bits: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    ff_multiplier: int = 4
    #: Peaks retained per spectrum.
    max_peaks: int = 128
    #: Peaks kept per 50 Da window when truncating. A global top-N would drop
    #: low-mass fragments, which carry disproportionate structural information.
    peaks_per_window: int = 8
    window_da: float = 50.0
    intensity_floor: float = 1e-3


class SinusoidalMass(nn.Module):
    """Sinusoidal encoding for continuous mass values.

    Mass needs a representation that makes small differences visible: 0.01 Da
    distinguishes real fragment formulas, but masses span 50-1200 Da. A
    multi-wavelength sinusoidal basis gives the model both scales at once,
    which a single linear projection cannot.
    """

    def __init__(self, dim: int, low: float = -2.0, high: float = 3.2, power: float = 1.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        n = dim // 2
        wavelengths = torch.pow(
            10.0, (high - low) * torch.pow(torch.linspace(0, 1, n), power) + low
        )
        self.register_buffer("inv_wavelength", (2 * math.pi) / wavelengths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.inv_wavelength
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with masked self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_multiplier: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        normed = self.norm1(x)
        qkv = self.qkv(normed).view(batch, length, 3, self.n_heads, dim // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = (~pad_mask)[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.dropout(self.out(attended.transpose(1, 2).reshape(batch, length, dim)))
        return x + self.dropout(self.ff(self.norm2(x)))


class FPNet(nn.Module):
    """Transformer mapping a spectrum to fingerprint-bit logits."""

    def __init__(self, config: FPNetConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.mz_encoding = SinusoidalMass(d)
        self.neutral_loss_encoding = SinusoidalMass(d)
        self.peak_projection = nn.Linear(2 * d + 1, d)

        self.precursor_encoding = SinusoidalMass(d)
        self.adduct_embedding = nn.Embedding(len(ADDUCT_VOCAB), d)
        self.instrument_embedding = nn.Embedding(len(INSTRUMENT_FAMILIES), d)
        # 3 scalars: collision energy, ionisation mode, log precursor mass.
        self.global_projection = nn.Linear(d + 3, d)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d, config.n_heads, config.dropout, config.ff_multiplier)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        # Both the summary token and the mean over peaks feed the head: the
        # former captures global context, the latter resists a single token
        # dominating when a spectrum has many weakly-informative peaks.
        self.head = nn.Sequential(
            nn.Linear(2 * d, 2048),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2048, config.n_bits),
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        pad_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        adduct: torch.Tensor,
        instrument: torch.Tensor,
        collision_energy: torch.Tensor,
        ionization_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Return fingerprint logits of shape ``(batch, n_bits)``.

        Args:
            mz: ``(batch, peaks)`` fragment m/z, zero-padded.
            intensity: ``(batch, peaks)`` intensities aligned with ``mz``.
            pad_mask: ``(batch, peaks)`` True where the slot is padding.
            precursor_mz: ``(batch,)`` precursor m/z.
            adduct: ``(batch,)`` indices into :data:`ADDUCT_VOCAB`.
            instrument: ``(batch,)`` indices into :data:`INSTRUMENT_FAMILIES`.
            collision_energy: ``(batch,)`` eV.
            ionization_mode: ``(batch,)`` +1 positive, -1 negative.
        """
        batch = mz.shape[0]
        neutral_loss = (precursor_mz[:, None] - mz).clamp(min=0.0)
        peaks = self.peak_projection(
            torch.cat(
                [
                    self.mz_encoding(mz),
                    self.neutral_loss_encoding(neutral_loss),
                    intensity.unsqueeze(-1),
                ],
                dim=-1,
            )
        )

        summary = self.global_projection(
            torch.cat(
                [
                    self.precursor_encoding(precursor_mz),
                    (collision_energy / 100.0).unsqueeze(-1),
                    ionization_mode.unsqueeze(-1),
                    (torch.log1p(precursor_mz) / 10.0).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        summary = summary + self.adduct_embedding(adduct) + self.instrument_embedding(instrument)

        x = torch.cat([summary.unsqueeze(1), peaks], dim=1)
        mask = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        summary_out = x[:, 0]
        peak_mask = (~mask[:, 1:]).float().unsqueeze(-1)
        peak_mean = (x[:, 1:] * peak_mask).sum(1) / peak_mask.sum(1).clamp(min=1.0)
        return self.head(torch.cat([summary_out, peak_mean], dim=-1))


def prepare_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    config: FPNetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Shape one spectrum for FPNet.

    Truncation keeps up to ``peaks_per_window`` peaks per 50 Da window before
    falling back to a global top-N, so low-mass fragments survive. Intensities
    are square-rooted because raw values span orders of magnitude and would
    otherwise let the base peak dominate the representation.
    """
    mz_arr = np.asarray(mz, dtype=np.float64)
    int_arr = np.asarray(intensity, dtype=np.float64)
    if mz_arr.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    keep = mz_arr <= (precursor_mz + 1.5)
    mz_arr, int_arr = mz_arr[keep], int_arr[keep]
    if mz_arr.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    peak_max = int_arr.max()
    if peak_max <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = int_arr >= config.intensity_floor * peak_max
    mz_arr, int_arr = mz_arr[keep], int_arr[keep]
    if mz_arr.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    if mz_arr.size > config.max_peaks:
        order = np.argsort(-int_arr)
        bucket = (mz_arr // config.window_da).astype(np.int64)
        per_bucket: dict[int, int] = {}
        selected: list[int] = []
        for i in order:
            b = int(bucket[i])
            if per_bucket.get(b, 0) < config.peaks_per_window:
                per_bucket[b] = per_bucket.get(b, 0) + 1
                selected.append(int(i))
        if len(selected) > config.max_peaks:
            sel = np.asarray(selected)
            selected = sel[np.argsort(-int_arr[sel])[: config.max_peaks]].tolist()
        elif len(selected) < config.max_peaks:
            chosen = set(selected)
            for i in order:
                if len(selected) >= config.max_peaks:
                    break
                if int(i) not in chosen:
                    selected.append(int(i))
        index = np.asarray(selected, dtype=np.int64)
        mz_arr, int_arr = mz_arr[index], int_arr[index]

    order = np.argsort(mz_arr)
    mz_arr, int_arr = mz_arr[order], int_arr[order]
    scaled = np.sqrt(int_arr / int_arr.max())
    return mz_arr.astype(np.float32), scaled.astype(np.float32)


def collate_spectra(
    prepared: list[tuple[np.ndarray, np.ndarray]],
    precursor_mz: list[float],
    adducts: list[str],
    instruments: list[str | None],
    collision_energies: list[float],
    ionization_modes: list[str],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Batch prepared spectra into padded tensors for :class:`FPNet`."""
    batch = len(prepared)
    if batch == 0:
        raise ValueError("cannot collate an empty batch")
    length = max(1, max(len(mz) for mz, _ in prepared))

    mz = np.zeros((batch, length), np.float32)
    intensity = np.zeros((batch, length), np.float32)
    pad_mask = np.ones((batch, length), bool)
    for i, (m, v) in enumerate(prepared):
        if len(m):
            mz[i, : len(m)] = m
            intensity[i, : len(v)] = v
            pad_mask[i, : len(m)] = False

    def tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device)

    energies = np.asarray(
        [25.0 if (e is None or not np.isfinite(e)) else e for e in collision_energies],
        dtype=np.float32,
    )
    return {
        "mz": tensor(mz, torch.float32),
        "intensity": tensor(intensity, torch.float32),
        "pad_mask": tensor(pad_mask, torch.bool),
        "precursor_mz": tensor(np.asarray(precursor_mz, np.float32), torch.float32),
        "adduct": tensor(
            np.asarray([ADDUCT_INDEX.get(a, ADDUCT_INDEX["<unk>"]) for a in adducts]),
            torch.long,
        ),
        "instrument": tensor(
            np.asarray([instrument_family(i) for i in instruments]), torch.long
        ),
        "collision_energy": tensor(energies, torch.float32),
        "ionization_mode": tensor(
            np.asarray([1.0 if m == "positive" else -1.0 for m in ionization_modes], np.float32),
            torch.float32,
        ),
    }


def score_candidates(candidate_fingerprints: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Rank candidates against predicted logits via one dot product.

    Under independent Bernoulli bits, a candidate's log-likelihood is
    ``f . z - sum_i log(1 + exp(z_i))``. The second term is identical for every
    candidate of a given query, so ``f . z`` induces the same ranking at a
    fraction of the cost — the whole pool is scored by a single matmul.

    Args:
        candidate_fingerprints: ``(n_candidates, n_bits)`` binary fingerprints.
        logits: ``(n_bits,)`` predicted logits.

    Returns:
        ``(n_candidates,)`` scores, higher is better.
    """
    fps = np.asarray(candidate_fingerprints, dtype=np.float32)
    z = np.asarray(logits, dtype=np.float32)
    if fps.ndim != 2:
        raise ValueError("candidate_fingerprints must be 2-D")
    if fps.shape[1] != z.shape[0]:
        raise ValueError(f"width mismatch: {fps.shape[1]} bits vs {z.shape[0]} logits")
    return fps @ z


def normalised_score(candidate_fingerprints: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """:func:`score_candidates` divided by sqrt(bit count).

    The raw dot product grows with the number of set bits, biasing toward
    larger molecules. This variant is supplied to the reranker alongside the
    raw score so it can learn how much that bias matters.
    """
    fps = np.asarray(candidate_fingerprints, dtype=np.float32)
    raw = score_candidates(fps, logits)
    return raw / np.sqrt(np.maximum(fps.sum(axis=1), 1.0))

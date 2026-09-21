"""De novo generation — fingerprint to SMILES decoder (Channel 6).

Targets Class 3 (structures absent from every candidate database), the one
novelty class every retrieval channel (library search, analog propagation,
FPNet-scored pool lookup) is structurally incapable of reaching: if the true
structure was never a candidate, no amount of reranking recovers it.

Architecture: a from-scratch reproduction of **MolForge**
(Ucak, Ashyrmamatov & Lee, "Reconstruction of lossless molecular
representations from fingerprints", J. Cheminformatics 2023,
doi:10.1186/s13321-023-00693-0) — an autoregressive, encoder-decoder
Transformer that translates a molecular fingerprint into a SMILES string,
the same "translation" framing as neural machine translation. This replaces
an earlier GRU decoder (the MSNovelist lineage), justified by MolForge's own
reported numbers: 93.6% exact (Tanimoto=1.0) SMILES reconstruction from
ECFP4, the best of the 13 fingerprints they benchmarked, vs. MSNovelist-style
recurrent decoders trailing well behind on the same reconstruction task.

Hyperparameters follow the paper's own "Method / Training" section exactly,
which the paper states are themselves copied from the original Transformer
paper (Vaswani et al. 2017) and "The Annotated Transformer" (Rush 2018):

* 6 encoder layers, 6 decoder layers
* 8 attention heads, d_model = 512
* Feed-forward dimension = 2048
* Dropout 0.1, applied to the output of every sub-layer
* **Pre-norm**: LayerNorm applied *before* each sub-layer (attention/FFN),
  not after — the paper explicitly calls this out as following the
  Annotated Transformer rather than the original post-norm Transformer.
* Positional encoding retained on the fingerprint side despite attention
  being permutation-invariant and fingerprint bits having no inherent order
  — the paper measured this deliberately (ablated it) and found pos-encoding
  still helps by 0.2-0.9%, so kept here rather than omitted as "obviously
  unnecessary."

One deliberate deviation from the paper, documented because it is load-
bearing elsewhere in this codebase: MolForge's own best-performing input was
**ECFP4 (radius-2 Morgan, 2048-bit hashed)** in isolation. We instead
condition on this project's existing 10,407-bit fingerprint (Morgan r2 4096 +
Morgan r3 4096 + RDKit path 2048 + MACCS 167 bits, see
:class:`casmi.chem.FingerprintCalculator`) rather than adding a second,
de-novo-only fingerprint type, because :mod:`casmi.models.fpnet` (the
spectrum encoder that supplies this decoder's *input* at inference time) is
already trained to predict exactly that layout, and the candidate pool /
GBM reranker also depend on it. Introducing an incompatible fingerprint here
would require retraining FPNet and everything downstream of it for no
benefit MolForge's own ablation attributes to fingerprint *choice* rather
than fingerprint *width* — a 10k-bit fingerprint is, if anything, a strict
superset of the substructure information ECFP4 alone carries (it includes a
same-radius-2 Morgan block as one of its four segments).

Encoder input representation: since a fingerprint is an unordered binary
vector rather than a native token sequence, we follow MolForge's own
"index-order for hashed fingerprints" convention (explicitly named in their
positional-encoding discussion): the encoder consumes the sequence of
*set-bit indices*, in ascending bit-index order, each embedded by
:class:`nn.Embedding` and given a positional encoding by its rank in that
sequence. Sequence length is therefore the number of on-bits for that
molecule (typically ~100-400 of the 10,407 bits), padded to the batch max.

Training targets each structure's own *true* fingerprint (computed once from
its SMILES), not FPNet's noisy prediction — the decoder learns "fingerprint
-> structure" as a clean inverse problem. At inference time it is fed FPNet's
*predicted* logits instead (thresholded to 0/1, see
:mod:`casmi.channels.denovo`), so bit-flip augmentation during training (see
:class:`casmi.models.train_denovo.DenovoTrainingConfig`) keeps the decoder
from overfitting to noise-free fingerprints it will never see at inference.

Generation is bounded beam search (fixed width and max length), never
exhaustive enumeration, matching the 9h/no-internet Kaggle constraint (see
``docs/04-method-landscape.md``, "Compute / engineering constraints").

This module imports torch and is therefore optional, matching
:mod:`casmi.models.fpnet`'s pattern; the baseline pipeline does not depend on
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from casmi.models.smiles_tokenizer import SmilesVocabulary


@dataclass
class DenovoConfig:
    """Architecture parameters. Travels with a checkpoint (see :mod:`fpnet`'s
    identical rationale: inference happens in a separate no-internet
    notebook, where a mismatched architecture fails obscurely at load time).

    Defaults reproduce MolForge's own reported hyperparameters (see module
    docstring): 6+6 layers, 8 heads, d_model 512, FFN 2048, dropout 0.1.
    """

    n_bits: int
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_length: int = 120
    #: Cap on encoder sequence length (number of set fingerprint bits fed
    #: in), independent of ``max_length`` (SMILES token length). Fingerprints
    #: rarely exceed a few hundred on-bits; this is a safety bound only.
    max_fingerprint_bits: int = 512


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017, §3.5).

    Added to token/bit embeddings before the first encoder/decoder layer.
    Kept for both the fingerprint encoder side (per MolForge's own ablation,
    see module docstring) and the SMILES decoder side (standard practice for
    autoregressive generation, where order is semantically load-bearing).
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, length, d_model)``. Adds positional encoding in-place-safe."""
        length = x.shape[1]
        return x + self.pe[:length].unsqueeze(0).to(dtype=x.dtype, device=x.device)


def fingerprint_to_bit_sequence(
    fingerprint: np.ndarray, max_bits: int, threshold: float = 0.5
) -> np.ndarray:
    """Convert one dense fingerprint vector to a sequence of set-bit indices.

    Bits are kept in ascending index order (MolForge's stated convention for
    hashed fingerprints, see module docstring), then truncated to
    ``max_bits`` if there are more set bits than that (rare; a safety bound,
    not an expected case for our fingerprint's typical on-bit counts).

    Args:
        fingerprint: ``(n_bits,)`` array. Values are compared against
            ``threshold`` rather than requiring exact 0/1, so this also
            accepts FPNet's continuous sigmoid probabilities directly at
            inference time.

    Returns:
        ``int64`` array of set-bit indices, length in ``[0, max_bits]``.
    """
    indices = np.flatnonzero(np.asarray(fingerprint) >= threshold)
    if indices.size > max_bits:
        indices = indices[:max_bits]
    return indices.astype(np.int64)


class FingerprintToSmilesDecoder(nn.Module):
    """MolForge-style Transformer: fingerprint (as a set-bit-index sequence)
    -> SMILES token sequence, via a standard encoder-decoder Transformer with
    pre-norm sub-layers.

    The encoder consumes the fingerprint's set-bit indices (see
    :func:`fingerprint_to_bit_sequence`); the decoder autoregressively
    generates SMILES tokens, cross-attending to the encoder's output —
    architecturally identical to a text-to-text NMT Transformer, per the
    paper's own framing of this as a translation task.
    """

    def __init__(self, config: DenovoConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.bit_embedding = nn.Embedding(config.n_bits, d)
        self.bit_positional = PositionalEncoding(d, max_len=config.max_fingerprint_bits)
        self.encoder_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # pre-norm, per the paper's Annotated-Transformer choice
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        self.token_embedding = nn.Embedding(config.vocab_size, d)
        self.token_positional = PositionalEncoding(d, max_len=config.max_length + 1)
        self.decoder_dropout = nn.Dropout(config.dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        self.output_head = nn.Linear(d, config.vocab_size)

    def encode(
        self, bit_indices: torch.Tensor, bit_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch of fingerprint bit-index sequences.

        Args:
            bit_indices: ``(batch, n_set_bits)`` int64, right-padded with any
                value (padded positions are excluded via ``bit_padding_mask``).
            bit_padding_mask: ``(batch, n_set_bits)`` bool, ``True`` at padded
                (ignored) positions — matches ``nn.Transformer``'s
                ``src_key_padding_mask`` convention.

        Returns:
            ``(batch, n_set_bits, d_model)`` encoder memory.
        """
        embedded = self.bit_embedding(bit_indices)
        embedded = self.bit_positional(embedded)
        embedded = self.encoder_dropout(embedded)
        return self.encoder(embedded, src_key_padding_mask=bit_padding_mask)

    def decode(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode SMILES token logits given encoder memory.

        Args:
            token_ids: ``(batch, length)`` decoder input tokens (teacher-
                forced during training, the running hypothesis during
                generation).
            memory: ``(batch, n_set_bits, d_model)`` from :meth:`encode`.
            memory_padding_mask: passed straight through as
                ``memory_key_padding_mask``.
            token_padding_mask: ``(batch, length)`` bool, ``True`` at padded
                decoder-input positions (only needed for batched teacher
                forcing with variable-length targets).

        Returns:
            ``(batch, length, vocab_size)`` logits.
        """
        embedded = self.token_embedding(token_ids)
        embedded = self.token_positional(embedded)
        embedded = self.decoder_dropout(embedded)
        length = token_ids.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=token_ids.device, dtype=embedded.dtype
        )
        hidden = self.decoder(
            embedded,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=token_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_head(hidden)

    def forward(
        self,
        bit_indices: torch.Tensor,
        bit_padding_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass for training.

        Args:
            bit_indices, bit_padding_mask: see :meth:`encode`.
            token_ids: ``(batch, length)`` including the leading start token;
                the model predicts token ``t+1`` from tokens ``0..t``.
            token_padding_mask: ``(batch, length)`` bool, ``True`` at padded
                positions.

        Returns:
            ``(batch, length, vocab_size)`` logits, aligned so position ``t``
            predicts the token at position ``t + 1`` in the un-shifted target.
        """
        memory = self.encode(bit_indices, bit_padding_mask)
        return self.decode(token_ids, memory, bit_padding_mask, token_padding_mask)


@torch.no_grad()
def beam_search_generate(
    model: FingerprintToSmilesDecoder,
    fingerprint: np.ndarray,
    vocabulary: SmilesVocabulary,
    beam_width: int = 8,
    max_length: int | None = None,
    device: str = "cpu",
) -> list[tuple[str, float]]:
    """Beam search decode SMILES candidates from one fingerprint vector.

    Bounded by ``beam_width`` and ``max_length`` throughout — never grows
    beyond ``beam_width`` live hypotheses, matching the no-exhaustive-search
    constraint this project holds itself to for de novo generation.

    Args:
        fingerprint: ``(n_bits,)`` FPNet-predicted logits/probabilities, or a
            true 0/1 fingerprint, for one molecule.
        beam_width: Number of hypotheses kept at each step.
        max_length: Caps generated length; defaults to ``model.config.max_length``.

    Returns:
        Up to ``beam_width`` ``(smiles, log_probability)`` pairs, sorted best
        first, deduplicated by decoded SMILES string. Unparseable/empty
        decodes are dropped by the caller (see
        :func:`casmi.channels.denovo.generate_candidates`), not here — this
        function only concerns itself with the sequence model.
    """
    model.eval()
    config = model.config
    length_cap = max_length or config.max_length

    threshold = 0.5 if float(np.max(fingerprint, initial=0.0)) <= 1.0 else 0.0
    bit_indices = fingerprint_to_bit_sequence(
        fingerprint, max_bits=config.max_fingerprint_bits, threshold=threshold
    )
    if bit_indices.size == 0:
        # No set bits at all (degenerate input) — nothing to condition on.
        return []
    bit_tensor = torch.as_tensor(bit_indices, dtype=torch.long, device=device).unsqueeze(0)
    bit_padding_mask = torch.zeros(1, bit_tensor.shape[1], dtype=torch.bool, device=device)
    memory = model.encode(bit_tensor, bit_padding_mask)

    start = vocabulary.start_index
    end = vocabulary.end_index

    # Each beam: (token_id_sequence, cumulative_log_prob, finished)
    beams: list[tuple[list[int], float, bool]] = [([start], 0.0, False)]

    for _ in range(length_cap):
        if all(finished for _, _, finished in beams):
            break
        candidates: list[tuple[list[int], float, bool]] = []
        for sequence, log_prob, finished in beams:
            if finished:
                candidates.append((sequence, log_prob, True))
                continue
            token_tensor = torch.as_tensor([sequence], dtype=torch.long, device=device)
            logits = model.decode(token_tensor, memory, bit_padding_mask)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            top_values, top_indices = torch.topk(log_probs, k=min(beam_width, log_probs.shape[0]))
            for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                new_sequence = sequence + [index]
                is_end = index == end
                candidates.append((new_sequence, log_prob + value, is_end))

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]

    results = []
    seen: set[str] = set()
    for sequence, log_prob, _ in sorted(beams, key=lambda item: item[1], reverse=True):
        smiles = vocabulary.decode(sequence)
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        results.append((smiles, log_prob))
    return results

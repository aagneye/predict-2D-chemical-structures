"""Neural channel: spectrum -> fingerprint model (FPNet).

Requires the optional ``train`` extra (torch). Importing this package without
torch installed raises ImportError; the baseline pipeline does not import it.
"""

from casmi.models.fpnet import (
    ADDUCT_VOCAB,
    INSTRUMENT_FAMILIES,
    FPNet,
    FPNetConfig,
    collate_spectra,
    instrument_family,
    normalised_score,
    prepare_peaks,
    score_candidates,
)
from casmi.models.train import (
    TrainingConfig,
    TrainingRow,
    build_training_rows,
    compute_bit_weights,
    evaluate_fpnet,
    load_checkpoint,
    predict_logits,
    save_checkpoint,
    train_fpnet,
)

__all__ = [
    "ADDUCT_VOCAB",
    "INSTRUMENT_FAMILIES",
    "FPNet",
    "FPNetConfig",
    "TrainingConfig",
    "TrainingRow",
    "build_training_rows",
    "collate_spectra",
    "compute_bit_weights",
    "evaluate_fpnet",
    "instrument_family",
    "load_checkpoint",
    "normalised_score",
    "predict_logits",
    "prepare_peaks",
    "save_checkpoint",
    "score_candidates",
    "train_fpnet",
]

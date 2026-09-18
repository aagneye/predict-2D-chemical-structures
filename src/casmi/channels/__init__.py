"""Evidence channels and their fusion into a ranked candidate list."""

from casmi.channels.analog import (
    Analog,
    AnalogIndex,
    find_analogs,
    propagate_to_candidates,
)
from casmi.channels.fusion import (
    DEFAULT_WEIGHTS,
    ScoredCandidate,
    fuse,
    mass_error_penalty,
    to_smiles_list,
)
from casmi.channels.library import LibraryHit, library_search

__all__ = [
    "DEFAULT_WEIGHTS",
    "Analog",
    "AnalogIndex",
    "LibraryHit",
    "ScoredCandidate",
    "find_analogs",
    "fuse",
    "library_search",
    "mass_error_penalty",
    "propagate_to_candidates",
    "to_smiles_list",
]

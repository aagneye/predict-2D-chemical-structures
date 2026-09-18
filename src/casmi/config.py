"""Central hyperparameter configuration.

Values carrying a "(ref)" note came from the public-LB-leading notebook
(``docs/05-community-intel.md``). **Those were tuned against a leaked visible
test set** and must be re-swept against our own held-out split before being
treated as optimal — see ``docs/06-implementation-plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpectrumConfig:
    """Spectral cleaning / similarity parameters."""

    #: Drop peaks below this fraction of the base peak. (ref: 0.002)
    intensity_floor: float = 0.002
    #: Keep at most this many peaks, most intense first. (ref: 256)
    max_peaks: int = 256
    #: Peak alignment tolerance in Da. (ref: 0.01)
    mz_tol: float = 0.01
    #: Exponent applied to intensities before normalisation. (ref: 1.0 linear)
    intensity_power: float = 1.0
    #: Apply Li et al. (2021) entropy weighting to low-entropy spectra. (ref: True)
    entropy_weighting: bool = True
    #: Drop peaks above precursor m/z + this margin (Da).
    precursor_margin: float = 1.5


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate-pool retrieval parameters."""

    #: Precursor/neutral-mass window in ppm. (ref: 8.5)
    ppm_window: float = 8.5
    #: Wider fallback window used only when the tight window is empty. (ref: 30.0)
    ppm_fallback: float = 30.0
    #: Cap on candidates scored per molecule after coarse pruning. (ref: 80)
    max_candidates: int = 80


@dataclass(frozen=True)
class AnalogConfig:
    """Channel 2 (mass-shifted analog propagation) parameters."""

    #: Neutral-mass shift search window in Da. (ref: 200.0)
    window_da: float = 200.0
    #: Number of analogs retained per molecule. (ref: 100)
    n_analogs: int = 100
    #: Exponent on analog spectral similarity when weighting Tanimoto. (ref: 4.0)
    sim_power: float = 4.0


@dataclass(frozen=True)
class FingerprintConfig:
    """Molecular fingerprint layout.

    Concatenation order is load-bearing: any saved model or precomputed pool
    must use the identical layout, so changing these invalidates artifacts.
    """

    morgan2_bits: int = 4096
    morgan3_bits: int = 4096
    rdkit_bits: int = 2048
    rdkit_max_path: int = 6
    #: MACCS keys are a fixed 167-bit block appended last.
    maccs_bits: int = 167

    @property
    def total_bits(self) -> int:
        return self.morgan2_bits + self.morgan3_bits + self.rdkit_bits + self.maccs_bits


@dataclass(frozen=True)
class SplitConfig:
    """Honest validation split parameters (see docs/06-implementation-plan.md)."""

    #: Number of held-out structures (real test set is ~400 molecules; we hold
    #: out more to shrink the error bar on our own MRR estimate).
    n_holdout: int = 2000
    #: Relative sampling weight per ``ingest_lib``. Natural-product-relevant
    #: libraries are upweighted because ``enveda-180`` dominates raw counts
    #: but is off-domain synthetic screening chemistry.
    library_weights: dict[str, float] = field(
        default_factory=lambda: {
            "enveda-np-examples": 8.0,
            "riken": 6.0,
            "gnps": 6.0,
            "massbank": 3.0,
            "mona": 3.0,
            "spectraverse": 2.0,
            "msdial": 2.0,
            "masaryk": 2.0,
            "pluskal_ms2": 1.0,
            "drug_plus": 0.5,
            "enveda-180": 0.25,
        }
    )
    #: Fraction of held-out structures assigned to each synthetic novelty class.
    #: Class 1 keeps sibling spectra in train; class 2 removes spectra but keeps
    #: the structure in the candidate pool; class 3 removes both.
    class_fractions: tuple[float, float, float] = (0.25, 0.45, 0.30)
    random_seed: int = 0


@dataclass(frozen=True)
class RankerConfig:
    """HistGradientBoosting reranker settings. (ref: all values)"""

    max_depth: int = 6
    max_iter: int = 500
    learning_rate: float = 0.03
    min_samples_leaf: int = 80
    l2_regularization: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    #: Class-1 prior weights to hedge over. (ref: (0.30, 0.60))
    class1_priors: tuple[float, ...] = (0.30, 0.60)


@dataclass(frozen=True)
class Config:
    """Top-level config aggregate."""

    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    analog: AnalogConfig = field(default_factory=AnalogConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    #: Number of SMILES per molecule in a submission. Fixed by the competition.
    top_n: int = 25


#: Importable default. Construct your own :class:`Config` to override.
CFG = Config()

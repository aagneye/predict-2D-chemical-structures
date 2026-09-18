"""Leak-proof validation splitting with synthetic novelty classes.

Three design decisions here, each addressing a specific way this dataset
misleads naive validation (see ``docs/06-implementation-plan.md``):

1. **Split by ``inchikey14``, never by ``spectrum_id``.** The same compound
   recurs across the 11 source libraries and at multiple adducts/collision
   energies. Splitting by spectrum puts near-identical spectra of the same
   structure on both sides, and the resulting MRR is meaningless. Splitting by
   full ``inchikey`` still leaks stereoisomers of the same 2D skeleton, which
   is what actually gets scored.

2. **Weight the hold-out toward natural-product libraries.** ``enveda-180``
   contributes ~46% of spectra but is synthetic drug-like screening chemistry.
   A uniform random hold-out is therefore mostly off-domain and reports a
   number that will not transfer to the natural-product test set.

3. **Synthesise the three novelty classes.** The real class mix is hidden, so
   we construct proxies and always report per-class metrics. An aggregate can
   stay flat while the mix underneath shifts, which is precisely how the
   public-LB-leading solution failed to notice three of its four channels were
   contributing nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from casmi.config import CFG, SplitConfig

#: Novelty class labels, mirroring the competition's definitions.
CLASS_LIBRARY = 1  # public reference spectra exist -> library search can win
CLASS_DATABASE = 2  # no spectra, but structure is in a database -> retrieval
CLASS_NOVEL = 3  # not in any database -> de novo only


@dataclass
class Split:
    """A leak-proof train/validation split over structures.

    Attributes:
        train_mask: Boolean mask over input rows belonging to the train side.
        val_mask: Boolean mask over input rows belonging to the held-out side.
        novelty_class: Held-out ``inchikey14`` -> synthetic class (1/2/3).
        excluded_from_pool: Held-out keys that must be *removed* from the
            candidate pool to simulate class 3. Honouring this is what makes
            the class-3 measurement meaningful rather than decorative.
        holdout_keys: All held-out ``inchikey14`` values.
    """

    train_mask: np.ndarray
    val_mask: np.ndarray
    novelty_class: dict[str, int]
    excluded_from_pool: set[str]
    holdout_keys: list[str] = field(default_factory=list)

    def class_of(self, key: str) -> int | None:
        return self.novelty_class.get(key)

    def summary(self) -> dict[str, int]:
        counts = {"train_rows": int(self.train_mask.sum()), "val_rows": int(self.val_mask.sum())}
        for cls in (CLASS_LIBRARY, CLASS_DATABASE, CLASS_NOVEL):
            counts[f"class{cls}_structures"] = sum(
                1 for c in self.novelty_class.values() if c == cls
            )
        return counts


def _structure_libraries(
    inchikey14: np.ndarray, ingest_lib: np.ndarray
) -> dict[str, set[str]]:
    """Map each structure to the set of libraries that measured it."""
    out: dict[str, set[str]] = defaultdict(set)
    for key, lib in zip(inchikey14, ingest_lib, strict=True):
        if key:
            out[key].add(lib if lib else "unknown")
    return out


def _sampling_weight(libraries: set[str], weights: dict[str, float]) -> float:
    """Weight for a structure, taken as the max over its source libraries.

    Max rather than mean: a compound present in both ``enveda-180`` and
    ``riken`` is a natural product that happens to also appear in the
    off-domain library, and should be treated as in-domain.
    """
    if not libraries:
        return 1.0
    return max(weights.get(lib, 1.0) for lib in libraries)


def make_split(
    inchikey14: np.ndarray,
    ingest_lib: np.ndarray,
    config: SplitConfig | None = None,
    database_keys: set[str] | None = None,
) -> Split:
    """Build a structure-level split with synthetic novelty classes.

    Args:
        inchikey14: Per-row structure key (length = number of spectra).
        ingest_lib: Per-row source library, used only for sampling weights.
        config: Split parameters; defaults to :data:`casmi.config.CFG`.
        database_keys: Structures known to an external database (e.g. COCONUT).
            When given, class-3 candidates are drawn preferentially from
            structures *absent* from it, so the class-3 cohort is genuinely
            un-retrievable rather than merely un-spectra'd.

    Returns:
        A :class:`Split`. Note that class-1 rows are *not* fully removed from
        the train side: by definition a class-1 molecule has sibling reference
        spectra available, so one spectrum per class-1 structure is held out as
        the query while its remaining spectra stay in the library.
    """
    cfg = config or CFG.split
    inchikey14 = np.asarray(inchikey14, dtype=object)
    ingest_lib = np.asarray(ingest_lib, dtype=object)
    if inchikey14.shape != ingest_lib.shape:
        raise ValueError("inchikey14 and ingest_lib must be the same length")

    rng = np.random.default_rng(cfg.random_seed)
    structure_libs = _structure_libraries(inchikey14, ingest_lib)
    # Structures with at least 2 spectra are eligible for class 1, which needs a
    # sibling spectrum to remain in the library.
    spectra_per_key: dict[str, int] = defaultdict(int)
    for key in inchikey14:
        if key:
            spectra_per_key[key] += 1

    keys = sorted(structure_libs)
    if not keys:
        raise ValueError("no structures to split")

    weights = np.array(
        [_sampling_weight(structure_libs[k], cfg.library_weights) for k in keys], dtype=np.float64
    )
    weights /= weights.sum()

    n_holdout = min(cfg.n_holdout, len(keys))
    chosen_idx = rng.choice(len(keys), size=n_holdout, replace=False, p=weights)
    holdout = [keys[i] for i in chosen_idx]

    # Assign novelty classes. Class 1 requires a spare sibling spectrum;
    # class 3 prefers structures the external database does not know.
    f1, f2, _ = cfg.class_fractions
    n_class1 = int(round(n_holdout * f1))
    n_class2 = int(round(n_holdout * f2))

    multi_spectrum = [k for k in holdout if spectra_per_key[k] >= 2]
    single_spectrum = [k for k in holdout if spectra_per_key[k] < 2]

    rng.shuffle(multi_spectrum)
    rng.shuffle(single_spectrum)

    class1 = multi_spectrum[:n_class1]
    remaining = multi_spectrum[n_class1:] + single_spectrum

    if database_keys is not None:
        # Prefer database-absent structures for class 3 so the cohort is truly
        # unreachable by retrieval; fall back to arbitrary keys if too few.
        absent = [k for k in remaining if k not in database_keys]
        present = [k for k in remaining if k in database_keys]
        class2 = present[:n_class2]
        leftover = present[n_class2:]
        class3 = absent + leftover
    else:
        class2 = remaining[:n_class2]
        class3 = remaining[n_class2:]

    novelty_class: dict[str, int] = {}
    for k in class1:
        novelty_class[k] = CLASS_LIBRARY
    for k in class2:
        novelty_class[k] = CLASS_DATABASE
    for k in class3:
        novelty_class[k] = CLASS_NOVEL

    holdout_set = set(novelty_class)
    class1_set = set(class1)

    # Row assignment. For class 2 and 3, every spectrum of the structure leaves
    # the library. For class 1, exactly one spectrum leaves (the query) and the
    # rest remain, which is what "reference spectra exist" means.
    val_mask = np.zeros(inchikey14.shape, dtype=bool)
    train_mask = np.zeros(inchikey14.shape, dtype=bool)
    query_taken: set[str] = set()
    for row, key in enumerate(inchikey14):
        if not key or key not in holdout_set:
            train_mask[row] = True
            continue
        if key in class1_set:
            if key not in query_taken:
                val_mask[row] = True
                query_taken.add(key)
            else:
                train_mask[row] = True
        else:
            val_mask[row] = True

    return Split(
        train_mask=train_mask,
        val_mask=val_mask,
        novelty_class=novelty_class,
        excluded_from_pool=set(class3),
        holdout_keys=sorted(holdout_set),
    )


def verify_no_leakage(
    split: Split, inchikey14: np.ndarray, allow_class1_siblings: bool = True
) -> None:
    """Assert the split does not leak held-out structures into the train side.

    Called by the split CLI as a hard gate. Class-1 structures legitimately
    keep sibling spectra on the train side; everything else must be absent.

    Raises:
        AssertionError: If a class-2/3 structure appears in the train side, or
            if masks overlap or fail to cover every row.
    """
    inchikey14 = np.asarray(inchikey14, dtype=object)
    if np.any(split.train_mask & split.val_mask):
        raise AssertionError("train and val masks overlap")
    if not np.all(split.train_mask | split.val_mask):
        raise AssertionError("some rows belong to neither side")

    train_keys = set(inchikey14[split.train_mask])
    offenders = []
    for key, cls in split.novelty_class.items():
        if cls == CLASS_LIBRARY and allow_class1_siblings:
            continue
        if key in train_keys:
            offenders.append((key, cls))
    if offenders:
        raise AssertionError(
            f"{len(offenders)} held-out structures leak into train, e.g. {offenders[:5]}"
        )


def holdout_truth(
    split: Split, inchikey14: np.ndarray, smiles: np.ndarray, molecule_ids: np.ndarray | None = None
) -> tuple[dict[str, str], dict[str, int]]:
    """Build ``(truth, novelty_classes)`` maps for scoring the held-out side.

    Molecules are keyed by ``molecule_ids`` when supplied, otherwise by
    ``inchikey14`` (one synthetic molecule per held-out structure).
    """
    inchikey14 = np.asarray(inchikey14, dtype=object)
    smiles = np.asarray(smiles, dtype=object)
    truth: dict[str, str] = {}
    classes: dict[str, int] = {}
    rows = np.flatnonzero(split.val_mask)
    for row in rows:
        key = inchikey14[row]
        if not key:
            continue
        identifier = str(molecule_ids[row]) if molecule_ids is not None else str(key)
        if identifier in truth:
            continue
        truth[identifier] = smiles[row]
        cls = split.novelty_class.get(key)
        if cls is not None:
            classes[identifier] = cls
    return truth, classes

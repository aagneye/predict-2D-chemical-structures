# Dataset Deep-Dive

Source: Kaggle "Data" tab description for `enveda-CASMI26-molecule-id-mass-spectra`.
Files live under `/data/` locally (gitignored, not committed — see
`.gitignore`). Total size 3.04 GB, formats parquet + csv, license CC BY-NC 4.0.

## Files

| File | Rows (approx) | Notes |
|---|---|---|
| `train.parquet` | ~2.5M spectra, ~275k unique structures | 1 row = 1 spectrum. Structure given. |
| `test.parquet` | ~1,500 spectra, ~400 molecules | 1 row = 1 spectrum. **No structure.** Placeholder now — swapped for hidden set on rerun (similar size). |
| `sample_submission.csv` | 400 rows | Valid-format example; every `smiles` cell in the sample is literally `CCO;CCO;...` × 25 (ethanol) — a dummy filler, not a real baseline. |

## Test set specifics

- ~1,500 spectra covering ~400 molecules
- 1–16 spectra per molecule, **median 3**
- All acquired on a **Bruker timsTOF** (single instrument — matches
  `enveda-180` and `enveda-np-examples` train libraries, nothing else)
- Monoisotopic mass range 157–1,159 Da, **median 348 Da** — small molecules,
  consistent with specialized/natural-product metabolites, not large peptides
- Minimal cleaning already applied to test spectra:
  - dropped spectra whose measured precursor mass was inconsistent with the
    known structure+adduct (only possible because Enveda knows ground truth
    internally)
  - dropped spectra with base peak < 1,000 raw counts (likely noise)
  - removed peaks above precursor + 2 Da (can't outweigh singly-charged
    precursor)
- Test adducts (only 10, subset of what's in train):
  `[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-`
- Test set origin (from the Enveda press release, see
  `03-literature-review.md`): generated on Enveda's own isolation/profiling
  platforms — **unpublished**, ~400 molecules / ~2,500 spectra, real or
  plausible natural products and analogs.

## Novelty classes (hidden distribution)

| Class | Definition | Difficulty driver |
|---|---|---|
| 1 | Has public reference MS/MS spectra | Library/spectral similarity search suffices |
| 2 | No public spectra, but structure ∈ PubChem or COCONUT | Needs formula ID + fingerprint-based **database retrieval** |
| 3 | Not in PubChem at all | Needs **de novo** structure generation — no DB shortcut |

Distribution over classes is secret and per-molecule — so we cannot know at
inference time which regime a given test molecule falls into. Design
implication: run all three strategies for every molecule and merge/rank the
combined candidate pool (see `04-method-landscape.md`).

## Column reference

### Shared columns (present in both train and test)

| Column | Type | Description |
|---|---|---|
| `molecule_id` | str | Anonymous compound id. Predictions keyed on this (not spectrum_id). |
| `spectrum_id` | str | Unique per spectrum. |
| `ms2_mzs` | array[float] | Fragment m/z values. |
| `ms2_normalized_intensities` | array[float] | Aligned intensities, base peak = 1.0 per spectrum. |
| `base_peak_intensity` | float (nullable) | Raw intensity of largest peak pre-normalization. Higher = cleaner spectrum generally. Null for libraries that shipped pre-normalized intensities only. |
| `adduct` | str | Standardized adduct, e.g. `[M+H]+`. |
| `ionization_mode` | str | `positive` / `negative`. |
| `instrument_type` | str | Test: always `timsTOF`. Train: free text, dozens of distinct strings, null where unrecorded. |
| `precursor_mz` | float | Measured precursor m/z. |
| `collision_energy_ev` | array[float] | Best-effort eV conversion; a list because merged multi-energy acquisitions exist, e.g. `[20,40,60,80]`. **Use this column if consuming CE as a model feature** — it's the one designed to line up between train/test. |
| `collision_energy_orig` | str | Lossless original value as recorded, e.g. `"40"`, `"[20 40 60]"`, `"35HCD"`, `"6V"`. |
| `collision_energy_orig_units` | str | `eV`, `NCE`, `V`, or `unknown`. Test set: always `eV`. |

### Train-only columns

| Column | Type | Description |
|---|---|---|
| `normalized_smiles` | str | **Label.** RDKit-standardized SMILES. |
| `inchikey` | str | Full InChIKey of the structure. |
| `inchikey14` | str | First block only — 2D skeleton id, ignores stereo. **This is what scoring ultimately reduces predictions to** (after tautomer canonicalization). |
| `molecular_formula` | str | Neutral-molecule formula. |
| `ingest_lib` | str | Source library id (see table below). |
| `adduct_orig` | str | Adduct string as recorded by source lib, pre-standardization. |
| `precursor_error_ppm` | float | ppm error between measured precursor m/z and value implied by labeled structure+adduct. Ships uncleaned — doubles as a **label/spectrum quality signal** (large ppm error → suspect row). |
| `num_peaks` | int | Peak count for the spectrum. |

## Source libraries (`ingest_lib`) in train

| `ingest_lib` | Spectra | Unique structures | Notes |
|---|---|---|---|
| `enveda-180` | 1,153,785 | 182,941 | Same Bruker timsTOF as test. Instrument-matched but chemistry = synthetic drug-like screening compounds — **different chemical space from test NPs**. |
| `pluskal_ms2` | 527,581 | 46,821 | MSnLib (Pluskal lab), Orbitrap, commercial/bioactive screening libs, multi-CE, one consistent protocol. |
| `riken` | 347,171 | 15,892 | RIKEN public libs, strong plant specialized-metabolite focus — **closest in chemistry to test domain among large libs**. |
| `gnps` | 220,849 | 45,750 | GNPS community libraries — largest public NP reference collection, but most heterogeneous (community-contributed, variable quality). |
| `massbank` | 101,727 | 9,180 | MassBank curated consortium, many labs/instruments. |
| `mona` | 92,416 | 11,681 | MassBank of North America (Fiehn Lab, UC Davis). |
| `spectraverse` | 50,933 | 9,631 | Recent harmonized aggregation, includes obscure libs missing elsewhere. |
| `msdial` | 40,765 | 9,127 | Distributed with MS-DIAL software, multi-instrument. |
| `drug_plus` | 2,545 | 2,539 | ~2,500 drug substances, ~1 spectrum each, no CE metadata. |
| `enveda-np-examples` | 1,151 | 250 | **Closest library to test**: 250 common NPs, same instrument + same pipeline as test. Use for pipeline validation / domain adaptation sanity checks. |
| `masaryk` | 652 | 416 | Small RECETOX (Masaryk Univ) chemical standards library. |

Structure counts sum > 275,810 total because many compounds are measured by
multiple libraries (train has real train/val leakage risk across libraries
if splitting naively by spectrum — **split by structure/InChIKey14, not by
spectrum_id**, to avoid leaking near-identical spectra of the same compound
across train/val).

Key strategic read: **`enveda-180`** dominates by spectra count and is
instrument-matched but chemically off-domain (synthetic screening compounds).
**`riken` + `gnps` + `enveda-np-examples`** are the natural-product-relevant
slices — likely more valuable per-spectrum for this specific task despite
being smaller, and possibly worth upweighting or targeted fine-tuning on.

## Collision energy handling

Three parallel columns exist because instruments report CE inconsistently:
- `collision_energy_orig` — verbatim string, lossless
- `collision_energy_orig_units` — `eV`/`NCE`/`V`/`unknown`, inferred from
  string → documented library convention → instrument analyser family
- `collision_energy_ev` — converted to eV (test's native unit). NCE→eV uses
  nominal Thermo formula `NCE × precursor_mz / 500 × charge_factor`
  (approximate — depends on instrument tuning). Null if unit unresolved.

**Rule of thumb: always feature-engineer off `collision_energy_ev`,** since
that's the column aligned with test.

## Common curation recipes mentioned by organizers (not required, worth
## trying / ablating)

- Relative-intensity floor: drop peaks < 2% / 1% / 0.1% of base peak
- Drop peaks above precursor m/z (+1–2 Da for isotopes)
- Minimum peak count filter (~5–6) — too few peaks ⇒ little structural signal
- Maximum peak count: keep top-N (e.g. N=128) most intense after above
  filters; classic alternative is top-~6-per-50-Da-window (preserves
  low-mass fragments a global top-N would discard)
- Deisotoping: remove ¹³C companion peaks ~1.0033 Da above a real fragment
- Intensity transform: sqrt or log (raw intensities are heavy-tailed)

## External resources named by organizers

- **MIST-CF** / **SIRIUS** — molecular formula annotation from MS/MS
  (rule-based + learned). Predict formula first, condition structure search
  on it — standard practice.
- **matchms** — open Python lib for spectral cleaning/curation/similarity.
- **GNPS "suspect" propagated annotations** — network-propagated labels,
  explicitly EXCLUDED from train.parquet (inferred not measured) — but
  could still be used as weak-supervision external data if desired (check
  license/terms).
- **MassIVE** — raw unannotated MS/MS repository (billions of spectra) — self-
  supervised pretraining corpus.
- **GeMS / DreaMS** — DreaMS project's curated unannotated corpus mined from
  MassIVE + pretrained spectrum embedding models — a pretraining shortcut.
- **FragHub** — open aggregation/harmonization of major public MS/MS libs,
  overlaps train.parquet, may have extra spectra (experimental or predicted).

See `03-literature-review.md` for what each of these actually does /
how they map onto the three novelty classes.

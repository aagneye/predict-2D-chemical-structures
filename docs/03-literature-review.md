# Literature Review

Research pass done 2026-09-17. All claims below are from search/fetch results
gathered this session (web_search / web_fetch) — cite sources inline where
useful for follow-up reading. This is not exhaustive; flagged as a living
document, extend as we read more.

## 1. CASMI: history and why it's being revived

- CASMI (**Critical Assessment of Small Molecule Identification**) was
  founded in **2012** by Emma Schymanski and Steffen Neumann, run by academic
  teams through several rounds until **2022**. Purpose: give the small-
  molecule-ID community a common blinded dataset + evaluation protocol,
  analogous to CASP for protein structure prediction. (Source: MDPI "And the
  Winner is..." retrospective, mdpi.com/2218-1989/3/2/412; PubMed 24958137.)
- **CASMI 2026** revives the name (with the original founders' permission,
  per the Kaggle page acknowledgements), hosted by **Enveda Biosciences** on
  Kaggle, explicitly to reach "the broader machine learning community"
  rather than only the cheminformatics/MS academic niche (per Business Wire
  press release, carried by Yahoo Finance / Times Argus, 2026-09-15).
- Per the press release: Enveda generated the hidden test set on its own
  "isolation and profiling platforms" — **~400 molecules, ~2,500 mass
  spectra, none previously published**. Molecules are real or hypothesized
  natural products and their analogs. This matches the Kaggle dataset page
  numbers (~400 molecules / ~1,500 test spectra visible now, since the
  public test.parquet is a stand-in of similar size, not the real ~2,500-
  spectrum hidden set).
- Framing in press coverage: solving "90% of biological molecules that no
  one can identify" — refers to the well-known metabolomics "dark matter"
  problem (most LC-MS/MS peaks in untargeted experiments remain unannotated).

## 2. The three canonical MS/MS→structure tasks (MassSpecGym framing)

**MassSpecGym** (Bushuiev et al., arXiv:2410.23326, NeurIPS 2024 track,
repo: github.com/pluskal-lab/MassSpecGym) is the most relevant existing
benchmark and directly informs how to think about CASMI's 3 novelty classes.
It defines three MS/MS annotation challenges:

1. **De novo molecular structure generation** — generate structure with no
   database lookup. ≈ CASMI Class 3.
2. **Molecule retrieval** — rank candidates from a structure database given
   a spectrum. ≈ CASMI Class 2 (and a fallback/complement for Class 1).
3. **Spectrum simulation** — predict a spectrum from a structure (the
   inverse direction; useful as a training signal / scoring function, e.g.
   "does this candidate structure's simulated spectrum match the observed
   spectrum" — a re-ranking tool).

MassSpecGym uses a "generalization-demanding" data split (splits by
structure, not by spectrum) — same leakage concern flagged in
`02-dataset.md` for our own train/val splitting.

## 3. Molecular formula annotation (the standard first step)

- **CSI:FingerID** (Dührkop et al., 2015) — kernel SVM that predicts a
  molecular fingerprint from spectral similarity (probability-product
  kernels over fragmentation trees), then searches a structure DB (PubChem)
  by fingerprint similarity. Integrated into **SIRIUS**.
- **SIRIUS** (SIRIUS 4 paper, Nature Methods 2019 — Dührkop et al.) —
  combines isotope-pattern analysis (MS1) + fragmentation-tree analysis
  (MS2) for **molecular formula** determination, then calls CSI:FingerID for
  structure DB search. Reported >70% identification rate on challenging
  metabolomics datasets (structure found in top-k of a DB search, not de
  novo). Free/open, widely used as a baseline in the field, has a documented
  Java CLI (boecker-lab.github.io/docs.sirius.github.io).
- **MIST-CF** (Goldman et al., arXiv:2307.08240, github.com/samgoldman97/
  mist-cf) — "Metabolite Inference with Spectrum Transformers (Chemical
  Formula)": predicts MS1 precursor chemical formula from MS/MS using a
  spectrum-transformer, explicitly named by the competition organizers as a
  recommended first step ("narrow the search space by conditioning on
  formula"). Sibling project **MIST** (samgoldman97/mist) does spectrum→
  fingerprint prediction, and is the encoder used in several de novo
  pipelines described below.

**Takeaway for us:** formula-first is the standard, well-validated approach
across almost every method in this space. We should plan for a
formula-prediction stage (SIRIUS/MIST-CF, or reproduce lightweight versions)
before anything else, since molecular formula sharply constrains both
retrieval candidate sets (Class 2) and de novo search space (Class 3).

## 4. Fingerprint-based retrieval pipeline (dominant paradigm, esp. for Class 2)

Standard two-stage pattern across nearly all modern methods:
1. **Spectrum → molecular fingerprint** (encoder: CSI:FingerID's SVM, or
   MIST's transformer, or DeepEI's targeted-bit CNN for EI-MS)
2. **Fingerprint → candidates**, either:
   - (a) **retrieval**: search a structure DB (PubChem/COCONUT) by
     fingerprint/Tanimoto similarity — works only if the true structure is
     *in* the DB (Class 1/2, not Class 3)
   - (b) **de novo decoding**: generate a structure directly from the
     fingerprint via a generative decoder (works for Class 3 too, in
     principle)

## 5. De novo structure generation methods (the hard, Class-3-relevant part)

This is the frontier and current SOTA is still weak — important expectation-
setting for our own targets.

- **MSNovelist** (Stravs et al., Nature Methods 2022,
  DOI 10.1038/s41592-022-01486-3) — first major de novo system: SIRIUS/
  CSI:FingerID fingerprint + molecular-formula prediction feeding an
  **encoder-decoder RNN (LSTM)** that generates a SMILES string de novo.
  Reported 61% "correct reproduction" on a GNPS reference set in their own
  eval protocol, but on the harder, generalization-focused MassSpecGym
  benchmark it scores **top-1 accuracy ≈ 0%** (see table below) — a stark
  illustration of how much benchmark design (train/test leakage via shared
  structures) inflates apparent performance. Treat any "% accuracy" number
  from a paper's own eval with suspicion; only benchmark-standardized numbers
  (MassSpecGym) are safely comparable across methods.
- **Spec2Mol** (Litsa et al., Nature Communications Chemistry 2023,
  github.com/KavrakiLab/Spec2Mol) — end-to-end encoder-decoder: encoder
  embeds a set of MS/MS spectra, decoder reconstructs SMILES directly
  (skips explicit fingerprint step). On par with fragmentation-tree methods,
  particularly better when the true structure isn't in the reference DB —
  i.e., explicitly targets the de novo / Class 3 regime.
- **DiffMS** (Bohde et al., 2025) — current graph-diffusion SOTA on
  MassSpecGym: MIST spectrum→fingerprint encoder feeds a **conditional graph
  diffusion model** that generates the molecular graph directly (given
  known/predicted formula fixing the atom set). Reports **2.30% top-1** exact
  match accuracy on MassSpecGym's held-out (generalization) split — small
  numbers, but state-of-the-art at time of publication, showing how hard
  true de novo generalization to unseen structures is.
- **MolForge** (Ucak et al., J. Cheminformatics 2023,
  github.com/knu-lcbc/MolForge) — "reconstruction of lossless molecular
  representations from fingerprints": an autoregressive transformer decoder
  that takes fingerprint on-bit indices → SMILES via beam search. Not
  originally an MS paper, but crucial as a fingerprint-to-structure decoder.
- **"One Small Step..." / MIST+MolForge pipeline** (Neo et al., DSO National
  Labs Singapore, arXiv:2508.04180, 2025) — combines the pretrained **MIST**
  encoder with **MolForge** as decoder (using step-function thresholding of
  fingerprint bit probabilities instead of raw probabilities), plus large-
  scale pretraining of MolForge on >2M compounds (not just the ~17k unique
  structures in MassSpecGym). Result: **28.27% top-1 / 36.11% top-10** exact-
  structure-match accuracy on MassSpecGym — roughly a **10x improvement**
  over DiffMS. Key finding: fingerprint decoding quality was the bottleneck,
  not the spectrum encoder — with ground-truth fingerprints as input, the
  same MolForge decoder reaches 46% top-1 / 59% top-10, meaning there's
  still headroom in the spectrum→fingerprint encoding step too.

  | Model | Top-1 Acc | Top-1 MCES↓ | Top-1 Tanimoto | Top-10 Acc |
  |---|---|---|---|---|
  | SMILES Transformer (MassSpecGym baseline) | 0.00% | 79.39 | 0.03 | 0.00% |
  | MIST + MSNovelist | 0.00% | 45.55 | 0.06 | 4.25% |
  | DiffMS | 2.30% | — | — | — |
  | **MIST + MolForge (pretrained)** | **28.27%** | **14.72** | **0.70** | **36.11%** |

  This table is the single most useful empirical reference point we have —
  it tells us **exact-match de novo generation should not be our primary
  strategy for the bulk of points**; a ranked-25 list with a strong retrieval
  path for Class 1/2 plus a best-effort de novo/candidate-generation tail for
  Class 3 is more realistic, and even modest de novo top-25 recall is
  valuable given MRR@25 rewards any hit in the top 25, not just top-1.

- **MADGEN** (Hassoun Lab, github.com/HassounLab/MADGEN) — "Mass-spec
  Attends to De novo GENeration" — transformer architecture generating
  structures directly from spectra, incorporating molecular + spectral
  information jointly. Newer, worth benchmarking against MIST+MolForge.
- **Test-time-tuned language models** (arXiv:2510.23746, 2025) — uses test-
  time tuning of a pretrained transformer LM for end-to-end de novo
  generation directly from spectra + molecular formula, bypassing
  intermediate fingerprint step. Recent (Oct 2025), worth reading in full
  before committing architecture.
- **MARLIN** (arXiv:2607.04774) — de novo elucidation **without requiring a
  ground-truth formula at any stage** — directly relevant since our test set
  formula must itself be inferred (not given), unlike some academic setups
  that assume formula is known.

## 6. Self-supervised spectrum embeddings / foundation models

- **DreaMS** (Bushuiev et al., Nature Biotechnology 2025,
  doi:10.1038/s41587-025-02663-3, docs: dreams-docs.readthedocs.io) — "Deep
  Representations Empowering the Annotation of Mass Spectra." Self-
  supervised transformer pretrained on **>24 million raw MS/MS spectra**
  (masked peak prediction + chromatographic retention-order prediction as
  pretraining objectives) from their own **GeMS** (GNPS Experimental Mass
  Spectra) corpus mined from MassIVE. Produces spectrum embeddings that
  cluster similar molecules even across instruments/conditions. This is
  explicitly named in the competition's "Other resources" list — a strong
  candidate as a frozen/fine-tuned spectrum encoder backbone, since it's
  pretrained cross-instrument (helps generalize beyond timsTOF specifically)
  and its weights are public (downloadable → attachable as Kaggle
  Dataset/Model, satisfying the no-internet-at-inference constraint).
- **LLM4MS** (Nature Communications Chemistry, late 2025/2026) — uses latent
  knowledge in general-purpose LLMs to derive discriminative spectral
  embeddings for compound ID. Newer, less validated at scale; worth a read
  but not yet a clear production choice.
- **ChemEmbed** (bioRxiv 2025.02.07.637102) — deep learning framework using
  enhanced MS/MS + multidimensional molecular embeddings for metabolite ID.
  Similar space to DreaMS; smaller-scale academic effort.
- **Structure-informed deep generative de novo annotation** (Nature Comms,
  2026, doi:10.1038/s41467-026-72149-6) — very recent (2026): reports top-1
  55.9% on 1,388 NIST MS2 spectra and 68.5% on 1,681 real-biological-sample
  spectra, claims to outperform existing in-silico tools and work across
  ionization modes without retraining. **High priority to read in full** —
  numbers are much higher than the MassSpecGym leaderboard above, which
  could mean either a genuinely better method, or a benchmark/eval protocol
  with more train/test structural overlap (a common inflation source per
  the MSNovelist example above) — verify eval protocol before trusting the
  headline numbers.

## 7. Structure databases relevant to Class 2 retrieval

- **PubChem** — the general/largest chemical structure database; "known
  structure" for Class 2 means present here (or in COCONUT).
- **COCONUT** ("COlleCtion of Open NatUral producTs") — CC0-licensed,
  largest open natural-products-specific structure database.
  COCONUT 2.0 (Nucleic Acids Research 2025, PMC11701633) is a "comprehensive
  overhaul and curation" of the original (~400k+ non-redundant NP structures
  per the original 2020 review). Also carries organism/species/geographic
  provenance metadata, useful if we want to filter candidates by plausible
  biological source. Freely downloadable (Zenodo) — satisfies the
  competition's external-data rules.
- Because Class 2's defining trait is "in PubChem or COCONUT but no public
  spectra," a **retrieval-by-fingerprint-similarity against a merged
  PubChem+COCONUT candidate pool** is the concrete mechanism to target this
  class — this is exactly the SIRIUS/CSI:FingerID design pattern, applied
  with COCONUT as an additional/refined candidate pool beyond generic
  PubChem (which is huge and mostly non-natural-product chemistry, i.e. a
  lot of the retrieval noise for this specific domain).

## 8. Spectral libraries / tooling for data engineering

- **matchms** (github, PyPI) — open Python library for spectral cleaning,
  filtering, similarity computation. Named by organizers explicitly; a
  reasonable choice for the initial data-cleaning module of `src/`.
- **FragHub** — open aggregation/harmonization of major public MS/MS
  libraries; overlaps heavily with train.parquet's sources but "may contain
  some additional spectra either experimental or predicted" — worth a diff
  against our train set once data is loaded, to see if it adds meaningful
  net-new coverage (especially for NP-relevant chemistry).
- **GNPS "suspect"/propagated annotations** — network-propagated (not
  measured) structure labels; explicitly excluded from train.parquet by the
  organizers because inferred ≠ measured. Could still be usable as a *weak-
  label* augmentation source if we're careful to keep it separate from
  "trusted" labels and never use it to validate — flag before using, since
  label noise here is qualitatively different from ordinary spectral noise.

## 9. Open questions / follow-up reading (for next session)

- Read the **DiffMS** paper in full (mentioned but not directly fetched this
  session) — graph diffusion architecture details, how it conditions on
  formula.
- Read the **2026 Nature Communications** "structure-informed deep
  generative" paper in full and scrutinize its eval protocol vs.
  MassSpecGym's — reconcile why its numbers are so much higher.
- Check whether **DreaMS/GeMS pretrained weights** are actually downloadable
  and license-compatible with attaching to a Kaggle notebook (need to verify
  license terms, not just described as "public").
- Investigate whether any **prior Kaggle competitions** dealt with MS/MS →
  structure or NMR/spectra → structure (the task mentioned wanting this
  investigated — no strong hits found yet this session; worth a dedicated
  Kaggle-competitions-search pass next session, e.g. "predict molecule from
  spectrum" on Kaggle's competition list directly rather than general web
  search).
- MADGEN and the Oct-2025 test-time-tuning paper are recent enough that
  they may not yet have public benchmark numbers on MassSpecGym — check for
  updated versions/camera-ready results.

# Literature Review (2026-10-13) — 2D structure from MS/MS: what the field knows, and what it means for us

Written for a reader who is new to the domain. Every claim is tied to a
source in §8. Where a number is contested, the contest is stated. This
supersedes `docs/03-literature-review.md` where they disagree (that file was
written before the MassSpecGym audit and before several 2026 papers).

## 0. The one-paragraph version

Identifying a small molecule from its tandem mass spectrum is a *ranking*
problem wrapped around a *coverage* problem. Everything that works in 2026
does the same four things: (1) narrow the candidate space with the
precursor mass (and ideally the formula), (2) get the true structure into
the candidate list, from a spectral library, a structure database, or a
generator, (3) score each candidate with several independent kinds of
evidence (spectral similarity, predicted fingerprints, *predicted spectra*,
in-silico fragmentation, and a prior for how well-documented the structure
is), and (4) learn to combine those scores. Pure "de novo" generation is
still weak when evaluated honestly (2–11% top-1), so the money is in
coverage plus ranking, with generation as a last resort for the ~39% of the
test set that no database contains.

## 1. The problem, in plain words

A mass spectrometer weighs a charged molecule (the **precursor**), breaks it,
and weighs the pieces (the **MS/MS spectrum**: a list of fragment m/z values
with intensities). The competition asks for the molecule's **2D structure**,
i.e. which atoms are bonded to which, written as SMILES. Stereochemistry and
tautomers are ignored by the scorer (InChIKey14 after RDKit tautomer
canonicalisation, stereo stripped first).

Why it is hard: the precursor mass pins the molecular formula to a few
options, but one formula can have thousands of isomers with similar
fragments; the fragmentation chemistry is only partly understood
(rearrangements); public reference spectra exist for ~60k–300k molecules
while PubChem has >100M; and spectra differ between instruments.

**The protein analogy the host uses.** Enveda's press release calls this
"the next AlphaFold contest". The analogy is loose: AlphaFold predicts 3D
protein *folds* from a known sequence; here the sequence itself (the atom
graph) is unknown. The closer cousin is **de novo peptide sequencing** in
proteomics (§5): reading an amino-acid sequence off an MS/MS spectrum. That
field is 20 years ahead of small-molecule work and its lessons transfer:
transformers that read raw peaks, precursor-mass-constrained decoding,
and above all *rescoring* candidate matches with predicted spectra and a
learned, decoy-calibrated ranker.

## 2. The base papers of this competition

### 2.1 CASMI (2012–2022) and what won

- CASMI was founded by Schymanski and Neumann in 2012 as a blinded contest
  for small-molecule identification. Early rounds showed that **combining
  in-silico fragmentation (MetFrag) with a metabolite-likeness prior helped
  the natural-product challenges**, and that candidate database size is a
  double-edged sword. [1, 2]
- CASMI 2016/2017 were dominated by **SIRIUS + CSI:FingerID** (fragmentation
  trees → predicted fingerprint → database search): 66/198 correct (33%) in
  the 2017 database-search category, more than six times the next entry,
  but with many failures traced to a wrong molecular formula. [3]
- **MetFrag 2.2 (2016)** showed the single largest cheap gain in the field's
  history: adding a *reference-count prior* (PubMed and patent counts from
  PubChem) to fragmentation scoring took top-1 from 43 to 336–420 of ~470
  challenges (71–89%). The learned weights were fragmentation 0.50,
  retention time 0.16, references 0.34. This is the same "popularity prior"
  the current Kaggle leaders rediscovered. [4]

### 2.2 Enveda's own work (the host built the hidden test)

- **MS2Mol (2023, ChemRxiv)**: a BART-style transformer that translates
  MS/MS peaks (tokenised) into SMILES (byte-pair encoded), optional formula
  input, beam search giving 25 candidates. Introduced **EnvedaDark**, 226
  natural products absent from major databases. Reported 62% "meaningful"
  and 21% "close" similarity, few exact matches; the comparison was against
  database tools because no other de novo code was public. [5]
- **MS2Prop (2022)**: predicts drug-likeness properties straight from
  unannotated spectra; shows the host cares about *properties* of unknowns,
  not only exact IDs. [6]
- **GrAFF-MS (ICML 2023, Murphy, Healey, Butler, Kind et al.)**: a forward
  model that predicts a spectrum as a distribution over a fixed vocabulary
  of ~2% of observed fragment formulas; the first fast, high-resolution
  neural spectrum predictor. [7]
- **Multi-scale sinusoidal embeddings (2022)** for high-resolution m/z
  values, the peak encoding now used by Casanovo, DreaMS and our FPNet. [8]
- **PRISM (2024, with Microsoft)**: self-supervised pretraining on >1
  billion unlabeled spectra (masked peaks), used for structure and property
  prediction. Not public. [9]

Takeaway: the host's test set was generated on timsTOF from isolated natural
products and analogs, including compounds "whose structures have not been
previously described". Enveda's own tooling is de-novo-first and
property-aware; their evaluation history favours Tanimoto-similarity
partial credit, but *this* competition scores exact 2D identity only.

### 2.3 MassSpecGym and the 2026 audit (read this before trusting any number)

MassSpecGym (NeurIPS 2024) is the standard benchmark: ~231k spectra, ~29k
molecules, MCES-based splits to limit leakage, three tasks (de novo,
retrieval, simulation). [10] In June 2026 the authors audited 34 papers
built on it and found problems in at least 17 of 26 that reported numbers:
[11]

- **MIST + MolForge's 28–31% top-1 de novo was inflated by a batched
  inference bug and by pretraining on ZINC/UniChem with close analogs of
  test molecules; corrected top-1 is 10.7% (top-10 14.5%).**
- Candidate-set artifacts let spectrum-blind models score 82–99% retrieval:
  ground-truth SMILES were formatted differently from decoys. Uniform
  canonicalisation fixed it (DreaMS+ChemBERTa fell 82% → 6.7%).
- **A ranker that orders candidates by PubChem deposition count reaches
  ~50% hit@1 without looking at the spectrum**, because the answers are
  well-characterised metabolites. The authors call this an open problem;
  for us it is a feature (§6).
- Training formula predictors (MIST-CF) or simulators (ICEBERG) on test
  molecules inflated retrieval by 10–20 points.
- Corrected honest reference points: MIST retrieval hit@1 ≈ 9.6%
  (mass-based candidates), 26.6% with a clean MIST-CF formula filter, 51.5%
  with the true formula; de novo top-1 for DiffMS 2.3%, MADGEN 1.3%,
  MSNovelist/Spec2Mol 0%.

Rules we adopt from the audit: canonicalise every candidate identically;
report a spectrum-blind (popularity-only) baseline next to every model;
never let a pretrained component see held-out molecules or their close
analogs (Tanimoto ≥ 0.7); pin the metric (InChIKey14, stereo stripped).

## 3. Retrieval: classes 1 and 2 (where the points are)

### 3.1 Fingerprint prediction, then database search

- **CSI:FingerID / SIRIUS** (2015–2019): fragmentation tree → kernel SVM
  per fingerprint bit → search PubChem by fingerprint similarity. Still the
  reference tool; ~38% correct on novel structures in CASMI-2016 conditions. [3]
- **MIST (Nature Machine Intelligence 2023)**: a transformer over annotated
  peak *formulas* that predicts fingerprints and, with contrastive training,
  embeds spectra for lookup. Beats CSI:FingerID on 66% of standards. Its
  sibling **MIST-CF** ranks formula+adduct assignments (clean version:
  50.5% top-1 / 73.2% top-5 formula accuracy on MassSpecGym). [12, 13]
- A 2026 theory paper proves a **trade-off between fingerprint accuracy and
  retrieval accuracy**: the Bayes-optimal decision differs by objective, so
  a model trained to predict bits well is not the best ranker and vice
  versa. Practical reading: train or fine-tune the ranking head on
  retrieval loss, not only on bit-wise BCE. [14]
- **In-domain retraining matters more than architecture**: MIST-CF gains
  +8 (NIST) to +34 (MoNA) top-1 points when retrained on the target domain. [15]

### 3.2 Joint embeddings (spectrum and structure in one space)

- **JESTR (Bioinformatics 2025)**: GNN for molecules + MLP for binned
  spectra, InfoNCE contrastive loss, plus a regulariser that pushes a
  spectrum away from its *same-formula* decoys. Rank@1 45.8% on NPLIB1 vs
  MIST 28.0%; 11.9% vs 9.6% on MassSpecGym (formula candidates); beats
  SIRIUS on a leakage-controlled subset (13.7% vs 10.4%). Code public. [16]
- **MSAlign (2026)**: aligns two frozen foundation models (DreaMS for
  spectra, MolDeBERTa for molecules) with a light adapter plus score
  fusion; state of the art on its unified retrieval benchmark; releases
  candidate sets and splits. [17]
- **GLMR (AAAI 2026)**: retrieve candidates contrastively, then *generate*
  refined structures conditioned on candidates + spectrum, then re-rank
  candidates by similarity to the generated structures (>40% relative
  top-1 gain claimed). [18]
- **SpecBridge (2026) was withdrawn** after the authors found a
  preprocessing/evaluation bug; its leaderboard entry (84.7%) is not real. [19]
- Cross-instrument generalisation is a real, separate problem: a 2026
  scaffold-disjoint, cross-instrument study reaches 42% top-1 in 256-way
  retrieval only after explicit domain-generalisation training. [20]

### 3.3 Forward simulators: predict the spectrum of each candidate

This is the most important development for isomer separation.

- **ICEBERG (bioRxiv 2025, ms-pred)**: two-stage GNN (generate fragments by
  bond breaks up to 3 steps, then score intensities), trained on NIST'20
  Orbitrap HCD; candidates = PubChem formula isomers; ranking by spectral
  entropy similarity averaged over collision energies. Top-1 on NIST'20
  [M+H]+ 25.1 → 40.0% after its contrastive fine-tuning with PubChem
  formula decoys; on MassSpecGym 31.9 → 46.0%; on MSnLib against the *full*
  PubChem (100M) 38.0% top-1 / 80.3% top-10, about twice SIRIUS's top-10.
  Limitations: rearrangements approximated by H shifts, Orbitrap only,
  leucine/isoleucine-type isomers indistinguishable. [21]
- **GLACIER (NeurIPS 2026, ms-pred)**: recasts fragmentation as *object
  detection* on the molecular graph: one transformer proposes subgraphs and
  intensities in one pass (multi-breakpoint). Top-1 retrieval 70.0% on
  MassSpecGym (prior SOTA 64.0%) and 52.5% on NIST'20 (prior 33.2%), ~8×
  faster than ICEBERG. Code MIT; MassSpecGym-trained weights public (the
  NIST-trained ones are not, and NIST is forbidden here anyway). [22]
- **FIORA (Nature Communications 2025)** predicts spectra from single
  fragmentation events with local neighbourhoods; **SingleFrag**, **SCARF**,
  **MassFormer**, **3DMolMS**, **CFM-ID** are the other families. In the
  2026 Molecules review, forward models on NIST'20 rank FraGNNet 30.3%,
  ICEBERG 27.9%, MassFormer 20.4%, NEIMS 19.9% top-1. [23, 24]

Why this matters for us: a fingerprint model looks at the *spectrum* and
asks "which bits are on"; a simulator looks at each *candidate* and asks
"would it produce these peaks". The two make different mistakes, and the
simulator sees atom connectivity, which is exactly what separates
same-formula isomers (the dominant error, see 05-community-intel: 98% of
wrong winners share the formula).

### 3.4 Priors that do not look at the spectrum

- **Documentation / popularity**: MetFrag's reference score (2016) and the
  2026 audit's spectrum-blind baseline (~50% hit@1) both show that *which
  structures people have studied* predicts the answer. On the competition's
  250-NP timsTOF panel, FPNet alone 40% top-1 → 93% with
  `0.25·(log1p substances + log1p PubMed)` (community benchmark, docs/08). [4, 11]
- **NP-likeness** (Jayaseelan & Steinbeck 2014): a learned "does this look
  like a natural product" score nearly doubled top-1 in NMR-based
  elucidation ranking (9/34 → 17/35). Cheap to add as a feature. [25]
- **Curated-database membership flags** (ChEBI, LIPID MAPS, NPAtlas) helped
  on public hold-outs and hurt on the leaderboard (community report): the
  hold-out molecules are over-represented in curated databases, the hidden
  test less so. Prefer continuous priors (popularity, NP-likeness) over
  binary membership.

### 3.5 Formula first

- SIRIUS (isotopes + fragmentation trees), **MIST-CF** (learned ranking of
  formula+adduct), and **BUDDY/msbuddy** (Nature Methods 2023: bottom-up
  MS/MS interrogation, works without an MS1 isotope pattern, estimates FDR)
  are the options. On NIST, SIRIUS top-5 0.758; pretrained MIST-CF 0.66–0.69;
  retrained in-domain MIST-CF 0.89 on MoNA. [13, 15, 26]
- Honest benchmarks show the *true* formula is worth ~+25 top-1 points for
  retrieval (26.6% → 51.5%), and formula leakage is the most common way to
  cheat by accident. Our data has only the precursor m/z with 10 known
  adducts; isotopes are partly recoverable from M+1 peaks (community
  measurement: carbon count to ±25% for 41% of molecules, +0.011 MRR). A
  formula predictor trained on *our* train set (leak-free) is the right
  tool; BUDDY is the drop-in alternative.

## 4. De novo generation: class 3 (~39% of the test, worth ~0 so far)

Honest numbers, exact match, after the audit:

| Method | Setting | Top-1 | Top-10 | Note |
|---|---|---|---|---|
| MSNovelist (2022), Spec2Mol (2023) | MassSpecGym | 0.0% | 4% | GNPS-era numbers (45%) were leakage [24, 27] |
| DiffMS (2025) | MassSpecGym, formula given | 2.3% | 4.1% | graph diffusion; robust to *predicted* formula [28] |
| MADGEN (ICLR 2025) | MassSpecGym | 1.3–1.5% | 12% with oracle scaffold | scaffold retrieval is the bottleneck [29] |
| Test-time-tuned LM (2025) | MassSpecGym, formula given | 3.2% | — | +64% Tanimoto vs SOTA [30] |
| MIST + MolForge (2025) | MassSpecGym, corrected | 10.7% | 14.5% | 31% claim was a bug + analog leakage [11, 31] |
| MARLIN (2026) | NPLIB1, **no formula** | ~17% (secondary source) | ~24% | DreaMS encoder + mass-shell block diffusion [32] |
| MS-GPT (2026) | NPLIB1 / MassSpecGym, formula given | 29.8% / 23.9% | 41.1% / 28.7% | posterior *band* of fingerprint queries into SAFE-GPT; not yet audited [33] |
| DualLGD (2026) | NPLIB1 / MassSpecGym | 34.4% / 23.9% | — | dual-stream line-graph diffusion; not yet audited [34] |
| MS-BART (NeurIPS 2025) | MassSpecGym, NPLIB1 | SOTA on 5/12 metrics | — | shared token vocabulary, 4M-molecule fingerprint pretraining [35] |
| MetGenX (Nat. Commun. 2026) | NIST, database-*free* | 21.7% | 35.0% top-3 | template-guided: retrieve analogs by spectrum, generate from their structures [36] |
| MetGenX | NIST, database-restricted (0.52M) | 55.9% | 76.1% top-3 | re-ranked by LightGBM on template similarity; 68.5% top-1 on real samples [36] |
| MS2Mol (Enveda 2023) | EnvedaDark | few exact | — | 21% "close", 62% "meaningful" similarity [5] |

What the field has learned:

1. **Fingerprint → SMILES decoders are bottlenecked by the fingerprint**,
   not the decoder. MS-GPT's diagnosis: training on oracle 0/1 fingerprints
   and decoding a thresholded noisy one is a train/inference mismatch; feed
   a *band* of posterior samples instead. Our MolForge channel had exactly
   this flaw (2% bit-flip augmentation only). [33]
2. **Templates beat blank-slate generation.** MetGenX's whole gain comes
   from retrieving spectrally similar *known* structures and generating
   "structure-to-structure" from them, then re-ranking by similarity to the
   templates. This is analog propagation (our Channel 2) turned into a
   generator, and it is the first de novo method with a believable 20%+
   database-free top-1. [36]
3. **Formula constraints are worth several points** and MARLIN shows a
   mass-shell constraint (every generated candidate must sit on the measured
   mass within ppm) recovers most of that without a formula predictor. [32]
4. **Expectations**: on honest, analog-disjoint test sets exact-match de
   novo is 2–11% top-1 for general methods and ~20% for template-guided
   ones. For MRR@25 on 39% of the test, that is at most +0.05–0.08. Worth
   doing, but after retrieval is saturated.
5. **Diversity of candidate sources** helps top-k more than top-1: the
   review and MADGEN both show oracle-scaffold / template hits concentrate
   in top-10, which MRR@25 rewards. [24, 29]

## 5. What proteomics (peptides and proteins) teaches

Peptide identification from MS/MS is the mature sibling of this problem, and
three of its ideas map directly:

- **Transformers that read peaks, not bins.** Casanovo (2022) and InstaNovo
  (2023–25) encode each (m/z, intensity) pair with sinusoidal embeddings and
  decode residues autoregressively with the precursor mass as a hard
  constraint; InstaNovo+ adds diffusion refinement of whole sequences. Our
  FPNet already uses this peak encoding; the *mass-constrained decoding* is
  what MARLIN borrows for molecules. [37, 38]
- **Rescoring with predicted spectra (Prosit, MS2PIP).** Database search
  gives candidates; a learned forward model predicts each candidate's
  spectrum; similarity to the measured spectrum becomes a rescoring
  feature, lifting identifications by up to 50% (immunopeptidomics) and
  allowing stricter FDR. ICEBERG/GLACIER are exactly Prosit for molecules. [39, 40]
- **Percolator: semi-supervised learning-to-rank with decoys.** Train a
  classifier on target-vs-decoy matches with many orthogonal features (score,
  mass error, matched-intensity fraction, ...), iterate, output calibrated
  posterior error probabilities. It gave +17% (tryptic) to +77%
  (non-tryptic) identifications in 2007 and is universal today. Our GBM
  reranker is Percolator; what we lack are (a) *decoys that look like the
  answer* (same-formula isomers), (b) calibrated confidence for gating
  which channel to trust, and (c) the predicted-spectrum feature. [41]
- **Why "AlphaFold for molecules" is not yet here**: protein structure
  prediction worked because sequence→structure had 200k solved structures
  and billions of evolutionary sequences. Spectrum→structure has ~60k–300k
  labelled molecules, no evolutionary signal, and a measurement that is
  lossy (isomers can be indistinguishable). Self-supervised pretraining on
  hundreds of millions of unlabeled spectra (DreaMS 2025: masked peaks +
  retention order, 201M-spectrum atlas; Enveda's PRISM: >1B spectra) is the
  field's answer to the data gap, and DreaMS is public and fine-tunable. [9, 42]

## 6. Findings → what we should build (ranked by expected MRR per week)

| # | Finding | Action for our pipeline | Expected gain | Cost |
|---|---|---|---|---|
| 1 | Documentation prior is the strongest single ranking signal on natural products (MetFrag 2016; audit 2026; 250-NP panel 40→93%) | Keep it as a reranker feature *and* as the post-rank blend; extend to PubChem tier rows (done); add NP-likeness as a second prior | large, mostly captured | done / small |
| 2 | Same-formula isomers are the error mode; forward simulators separate them (ICEBERG +15 top-1 points, GLACIER 70% on MassSpecGym) | Re-score same-formula groups of our top-60 with GLACIER (MIT, MassSpecGym weights) and ICEBERG; entropy similarity averaged over the molecule's [M+H]+ / [M+Na]+ spectra; feed as reranker features (`sim_pred_max`, `sim_pred_mean`, z-score within formula group) | +0.02–0.05 LB | 1–2 days (offline packages exist) |
| 3 | Retrieval heads trained for retrieval beat fingerprint-accuracy heads (JESTR vs MIST; the 2026 trade-off theorem) | Add a contrastive head on FPNet's encoder: InfoNCE over (spectrum, structure) with same-formula hard negatives, trained on our split; use its cosine as another feature | +0.02–0.04 held-out | 2–3 days + GPU |
| 4 | Leak-free formula knowledge is worth ~25 points of retrieval | Train a formula ranker on our train side (MIST-CF-style, or run msbuddy offline) and add `formula_rank`, `formula_prob` features; use the top formula to tighten the candidate window | +0.02–0.03 | 2 days |
| 5 | Coverage: the audit's corrected MIST reaches 9.6% hit@1 on mass-based candidates and 51% with formula; PubChem (100M) + popularity is what the lineage and ICEBERG use | PubChem tier channel (done); raise top_k per molecule to 40 and let the reranker cut | being measured | done |
| 6 | Template-guided generation (MetGenX) is the only honest 20%+ de novo | For class 3: condition MolForge on the *analog structures'* fingerprints and the mass shell, not only on FPNet bits; re-rank generated candidates by Tanimoto to templates; posterior-band fingerprint sampling (MS-GPT) | +0.02–0.04 if class 3 is really 39% | 1 week |
| 7 | Cross-instrument shift costs 10–30 points; our library is 54% non-timsTOF | Prefer timsTOF representatives in library/analog search (the lineage does), add instrument-family as a reranker feature, fine-tune FPNet on the enveda-180 + np-examples slice last | +0.01–0.02 | small |
| 8 | Evaluation hygiene (audit) | Spectrum-blind popularity-only baseline in every run; canonicalise all candidates; report class-wise recall first; never tune on the 130-molecule public LB | protects private LB | done partly |
| 9 | Calibrated confidence (Percolator PEP; review's "abstention") | Convert reranker scores to calibrated probabilities; gate PubChem/de novo insertion depth on the top-candidate confidence instead of fixed slots | +0.01 | 1 day |

What to **stop** doing: treating MolForge-from-FPNet-bits as the class-3
plan on the strength of the 28% paper (it is 10.7% after correction, on a
benchmark friendlier than ours), and treating raw FPNet dot products as a
fusable score (fixed in docs/08).

## 7. Open questions we will have to settle ourselves

- Does GLACIER trained on MassSpecGym (mostly Orbitrap/QTOF) transfer to
  timsTOF spectra? The lineage's +0.006–0.013 LB suggests "partly". A quick
  test on the 250-NP timsTOF panel decides it.
- How much of the hidden class 2 is in COCONUT ∪ PubChemLite vs only in the
  PubChem tier? Our phase-B recall numbers by source will tell.
- Is the hidden class 3 "novel scaffold" or "known scaffold, new
  decoration"? MetGenX's template approach only helps the second kind.

## 8. Sources

1. Schymanski & Neumann, "CASMI: challenges and solutions", Metabolites 2013 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3901296/
2. Ruttkies et al., CASMI 2012/2013 MetFrag + metabolite-likeness — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3901266
3. Böcker lab, CASMI 2017 results — https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out
4. Ruttkies et al., "MetFrag relaunched", J. Cheminf. 2016 — https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-016-0115-9
5. Butler et al. (Enveda), "MS2Mol", ChemRxiv 2023 — https://chemrxiv.org/engage/chemrxiv/article-details/6492f28ea2c387fa9ab2a465
6. Enveda, "MS2Prop", bioRxiv 2022 — https://www.biorxiv.org/content/10.1101/2022.10.09.511482.full.pdf
7. Murphy et al., "GrAFF-MS", ICML 2023 — https://proceedings.mlr.press/v202/murphy23a.html
8. Voronov et al., "Multi-scale sinusoidal embeddings…", arXiv 2022 — https://arxiv.org/pdf/2207.02980
9. Enveda PRISM announcement, 2024 — https://www.businesswire.com/news/home/20240529298878/en/
10. Bushuiev et al., "MassSpecGym", NeurIPS 2024 — https://github.com/roman-bushuiev/MassSpecGym
11. "MassSpecGym in the Wild: evaluation pitfalls", arXiv 2606.19624 (2026) — https://arxiv.org/abs/2606.19624
12. Goldman et al., "MIST", Nat. Mach. Intell. 2023 — https://www.biorxiv.org/content/10.1101/2022.12.30.522318.full.pdf
13. Goldman et al., "MIST-CF", 2023 — https://chemrxiv.org/engage/chemrxiv/article-details/65edbbba66c1381729ae3ed3
14. "Small molecule retrieval from MS/MS: what are we optimizing for?", arXiv 2602.16507 — https://arxiv.org/abs/2602.16507
15. "Comparative analysis of formula and structure prediction", arXiv 2601.00941 — https://arxiv.org/pdf/2601.00941
16. Kalia et al., "JESTR", Bioinformatics 2025 — https://academic.oup.com/bioinformatics/article/41/7/btaf354/8165426
17. "MSAlign", arXiv 2605.19752 — https://arxiv.org/abs/2605.19752
18. "Breaking the Modality Barrier (GLMR)", AAAI 2026 — https://arxiv.org/abs/2511.06259
19. "SpecBridge" (withdrawn), arXiv 2601.17204 — https://arxiv.org/abs/2601.17204
20. "Contrastive domain generalization for cross-instrument identification", arXiv 2602.00547 — https://arxiv.org/abs/2602.00547
21. Goldman et al., "Neural spectral prediction… (ICEBERG)", bioRxiv 2025 — https://www.biorxiv.org/content/10.1101/2025.05.28.656653.full
22. Wang et al., "GLACIER", NeurIPS 2026, arXiv 2606.29161 — https://arxiv.org/abs/2606.29161 ; code https://github.com/coleygroup/ms-pred
23. Nowatzky et al., "FIORA", Nat. Commun. 2025 — https://www.nature.com/articles/s41467-025-57422-4
24. "De novo structure prediction from tandem MS: algorithms, benchmarks, limitations", Molecules 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12985711/
25. Jayaseelan & Steinbeck, "NP-likeness for candidate ranking", BMC Bioinf. 2014 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4089557/
26. Xing et al., "BUDDY", Nat. Methods 2023; msbuddy docs — https://msbuddy.readthedocs.io/
27. Stravs et al., "MSNovelist", Nat. Methods 2022 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9262714/
28. Bohde et al., "DiffMS", 2025 (via [24])
29. Wang et al., "MADGEN", ICLR 2025 — https://arxiv.org/html/2501.01950v2
30. "Test-time tuned LMs for de novo generation", arXiv 2510.23746 — https://arxiv.org/abs/2510.23746
31. Neo et al., "One small step with fingerprints… (MIST+MolForge)", arXiv 2508.04180 — https://arxiv.org/abs/2508.04180
32. "MARLIN", arXiv 2607.04774 — https://arxiv.org/abs/2607.04774
33. "MS-GPT", arXiv 2607.23607 — https://arxiv.org/abs/2607.23607 ; code https://github.com/VIKI623/MS-GPT
34. "DualLGD", arXiv 2605.07048 — https://arxiv.org/abs/2605.07048
35. Han et al., "MS-BART", NeurIPS 2025 — https://github.com/OpenDFM/MS-BART
36. Wang, Zhang & Zhu, "MetGenX", Nat. Commun. 17:5426 (2026) — https://www.nature.com/articles/s41467-026-72149-6
37. Yilmaz et al., "Casanovo", 2022 — https://www.biorxiv.org/content/10.1101/2022.02.07.479481.full.pdf
38. Eloff et al., "InstaNovo", 2023/2025 — https://www.biorxiv.org/content/10.1101/2023.08.30.555055.full.pdf
39. Gessulat et al., "Prosit", Nat. Methods 2019 — https://www.mls.ls.tum.de/proteomics/news/article/prosit-proteome-wide-prediction-of-peptide-tandem-mass-spectra-by-deep-learning
40. Cormican et al., "inSPIRE", 2022 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9720494/
41. Käll et al., "Percolator", Nat. Methods 2007 — https://noble.gs.washington.edu/papers/kall2007semi-supervised.html
42. Bushuiev et al., "DreaMS", Nat. Biotechnol. 2025 — https://dreams-docs.readthedocs.io/

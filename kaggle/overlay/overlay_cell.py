# ---- Class-3 overlay (aagneye): our analog-derived structure below the base's rank 1 ----
# Runs after the public base has written /kaggle/working/submission.csv. Our pipeline
# (github.com/aagneye/predict-2D-chemical-structures) proposes structures that are in no
# database by applying biosynthetic edits to spectral analogs (Channel 8). For every
# molecule without a close library match, the best such structure is inserted directly
# below the base's rank 1; the base's rank 1 and candidate set are otherwise unchanged.
import glob
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
ROOT = '/kaggle/working'
FULL_RUN = bool(globals().get('FULL_RUN', True))
BASE_PATH = f'{ROOT}/submission.csv'
shutil.copy2(BASE_PATH, f'{ROOT}/submission_base_before_overlay.csv')

# The base installs the same pinned RDKit; install ours only if it is missing.
try:
    import rdkit
    assert rdkit.__version__ == '2026.03.3'
except Exception:
    wheels = glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True)
    assert wheels, 'RDKit 2026.3.3 wheel not found under /kaggle/input'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                    wheels[0]], check=True)
    import rdkit
print('rdkit', rdkit.__version__)

inits = glob.glob('/kaggle/input/**/casmi26-baseline-assets/**/casmi/__init__.py', recursive=True)
assert inits, 'attach aagneye/casmi26-baseline-assets (casmi package not found)'
ASSETS = os.path.dirname(os.path.dirname(os.path.dirname(inits[0])))
sys.path.insert(0, os.path.dirname(os.path.dirname(inits[0])))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from casmi import __version__  # noqa: E402
from casmi.candidates.pool import CandidatePool  # noqa: E402
from casmi.channels.derivation import DerivationConfig  # noqa: E402
from casmi.channels.ranker import Reranker  # noqa: E402
from casmi.chem import inchikey14  # noqa: E402
from casmi.data.loaders import load_query_molecules, load_spectral_library  # noqa: E402
from casmi.models.train import load_checkpoint  # noqa: E402
from casmi.overlay import OverlayConfig, merge_overlay, weak_evidence  # noqa: E402
from casmi.pipeline import run_pipeline  # noqa: E402
from casmi.priors import PopularityPrior  # noqa: E402

print('casmi', __version__, '| assets', ASSETS)
COMP = next(d for d in ('/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra',
                        '/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra')
            if os.path.isfile(d + '/test.parquet'))

pool = CandidatePool.load(f'{ASSETS}/pool.npz')
popularity = PopularityPrior.load(f'{ASSETS}/popularity.npz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
fpnet_model, fpnet_config = load_checkpoint(f'{ASSETS}/fpnet_final.pt', device=DEVICE)
reranker = Reranker.load(f'{ASSETS}/ranker.pkl')
assert 'derived' in reranker.feature_names, 'the overlay needs the Channel 8 (33-feature) ranker'
pubchem_tier, pubchem_config = None, None
tier_hits = glob.glob('/kaggle/input/**/pc_smiles.npy', recursive=True)
pop_hits = glob.glob('/kaggle/input/**/pc_lsid.npy', recursive=True)
if tier_hits:
    from casmi.channels.pubchem import PubChemConfig, PubChemTier
    pubchem_tier = PubChemTier(os.path.dirname(tier_hits[0]),
                               os.path.dirname(pop_hits[0]) if pop_hits else None)
    pubchem_config = PubChemConfig(top_k=25)
print(f'pool {len(pool):,} | popularity {len(popularity):,} | '
      f'pubchem tier {"yes" if pubchem_tier else "no"} | device {DEVICE}')

library = load_spectral_library(f'{COMP}/train.parquet')
molecules = load_query_molecules(f'{COMP}/test.parquet')
if not FULL_RUN:
    molecules = molecules[:12]  # commit smoke on the visible placeholder; the rerun does all
print(f'library {library.n_spectra:,} spectra | query molecules {len(molecules)} | '
      f'setup {time.time() - T0:.0f}s')

result = run_pipeline(
    molecules, library, pool, progress_every=25, fragmentation_channel=True,
    fpnet_model=fpnet_model, fpnet_config=fpnet_config, fpnet_device=DEVICE,
    reranker=reranker, popularity=popularity, popularity_mu=0.15,
    pubchem_tier=pubchem_tier, pubchem_config=pubchem_config,
    derivation_config=DerivationConfig(),
)
print(f'our pipeline done at {time.time() - T0:.0f}s')

# Offline (split v3, 2,244 molecules): one structure below rank 1 when the best library
# similarity is < 0.8 costs classes 1-2 nothing measurable and lifts class 3 from 0 to 0.05.
OVERLAY = OverlayConfig(insert_at=1, max_insert=1, library_max_threshold=0.8)
base = pd.read_csv(BASE_PATH, dtype=str, keep_default_na=False)
diag = {d.molecule_id: d for d in result.diagnostics}
rows, audit = [], []
for mid, text in base.itertuples(index=False):
    original = [s for s in text.split(';') if s]
    d = diag.get(str(mid))
    final, inserted = original, ''
    if d is not None and d.derived_smiles and weak_evidence(d.best_library_similarity, OVERLAY):
        final = merge_overlay(original, d.derived_smiles, OVERLAY,
                              key=lambda s: inchikey14(s) or None)
        if len(final) > 1 and final[1] not in original:
            inserted = final[1]
    rows.append((mid, ';'.join((final + ['CCO'] * 25)[:25])))
    audit.append(dict(molecule_id=mid, processed=d is not None,
                      library_max=d.best_library_similarity if d else None,
                      n_derived=len(d.derived_smiles) if d else 0, inserted=inserted))

out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
sample = pd.read_csv(f'{COMP}/sample_submission.csv', dtype=str, keep_default_na=False)
assert out.molecule_id.tolist() == sample.molecule_id.tolist(), 'molecule order changed'
assert out.smiles.str.split(';').map(len).eq(25).all(), 'every row needs 25 candidates'
out.to_csv(f'{ROOT}/submission.csv', index=False)
pd.DataFrame(audit).to_csv(f'{ROOT}/overlay_audit.csv', index=False)
n_inserted = sum(1 for a in audit if a['inserted'])
print(f'overlay: {n_inserted}/{len(audit)} molecules received a derived structure at rank 2 '
      f'| total {time.time() - T0:.0f}s')

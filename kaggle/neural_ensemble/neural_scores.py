"""Runs after the public pipeline process exits, freeing its GPU memory.

Seed ensemble (aagneye): the public v1 scorer's module (sha256 cb9e4d67...) with one change:
every checkpoint named final_model_seed*.pt under /kaggle/input (same recipe, other seeds)
is scored too, and cos, fp and combined scores are averaged over v1 + the seeds. With no
seed files attached the output equals v1's.
"""
import json,sys,importlib.util,types,hashlib  # noqa: F401 (upstream line, kept verbatim)
from pathlib import Path
import numpy as np,pandas as pd,torch

def load_model(bundle,checkpoint,device):
    # Private module namespace prevents collision with public engines' model/features modules.
    package='_casmi_own';p=types.ModuleType(package);p.__path__=[str(bundle)];sys.modules[package]=p
    for name in ['features','model']:
        spec=importlib.util.spec_from_file_location(package+'.'+name,Path(bundle)/(name+'.py'))
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    features=sys.modules[package+'.features'];cls=sys.modules[package+'.model'].SpectrumCandidateModel
    cfg=json.loads((Path(bundle)/'config.json').read_text());model=cls(cfg).to(device)
    ck=torch.load(Path(bundle)/checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(ck['state_dict']);model.eval()
    return model,features,cfg

def load_seeds(cls,cfg,device):
    models=[]
    for path in sorted(Path('/kaggle/input').rglob('final_model_seed*.pt')):
        model=cls(cfg).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state_dict'])
        models.append(model.eval());print('NEURAL SEED',path,flush=True)
    return models

@torch.inference_mode()
def score(bundle,checkpoint,evidence,queries,output,device='cuda'):
    torch.set_num_threads(4)
    if device=='cuda':assert torch.cuda.is_available(),'GPU required for neural scoring'
    model,feat,cfg=load_model(bundle,checkpoint,device)
    models=[model]+load_seeds(type(model),cfg,device)
    print('NEURAL ENSEMBLE size',len(models),flush=True)
    frame=pd.read_parquet(evidence);q=pd.read_parquet(queries);q['molecule_id']=q.molecule_id.astype(str)
    query={mid:g for mid,g in q.groupby('molecule_id',sort=False)};cache={};parts=[];errors=[]
    for index,(mid,g) in enumerate(frame.groupby('molecule_id',sort=False)):
        g=g.sort_values('baseline_rank',kind='stable').copy()
        g['neural_ok']=False
        if not g.processed.iloc[0]:
            for c in ['neural_cos','neural_fp','neural_score','neural_std','n_spectra','n_adducts']:g[c]=0.
            parts.append(g);continue
        try:
            assert mid in query,mid
            records=[]
            # Use EVERY observed query spectrum; no adduct merging and no new spectrum selection rule.
            for row in query[mid].itertuples(index=False):
                try:
                    pk,n=feat.peaks(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,cfg['max_peaks'])
                    if n<cfg['min_peaks']:continue
                    m,a=feat.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                    records.append((pk,n,m,a))
                except (ValueError,TypeError):continue
            if not records:raise ValueError('No usable spectra after the training preprocessing filters')
            fp=[]
            for s in g.smiles:
                if s not in cache:cache[s]=feat.structure(s,cfg['fp_bits'])[-1]
                fp.append(cache[s])
            F=torch.tensor(np.stack(fp),device=device,dtype=torch.float32)
            cos_sum=0.;fps_sum=0.
            for net in models:
                M=net.encode_molecules(F);cos_all=[];fp_all=[]
                for st in range(0,len(records),64):
                    b=records[st:st+64]
                    x=torch.tensor(np.stack([v[0] for v in b]),device=device)
                    lengths=torch.tensor([v[1] for v in b],device=device,dtype=torch.long)
                    meta=torch.tensor(np.stack([v[2] for v in b]),device=device)
                    adduct=torch.tensor([v[3] for v in b],device=device,dtype=torch.long)
                    Z,P=net.encode_spectra(x,lengths,meta,adduct)
                    cos_all.append((Z@M.T).cpu().numpy());fp_all.append((P@F.T/cfg['fp_bits']).cpu().numpy())
                cos_sum=cos_sum+np.concatenate(cos_all);fps_sum=fps_sum+np.concatenate(fp_all)
            cos=cos_sum/len(models);fps=fps_sum/len(models)
            combined=cos/cfg['temperature']+cfg['fp_score_weight']*fps
            assert np.isfinite(combined).all(),mid
            g['neural_cos']=cos.mean(0);g['neural_fp']=fps.mean(0);g['neural_score']=combined.mean(0)
            g['neural_std']=combined.std(0);g['n_spectra']=len(records);g['n_adducts']=query[mid].adduct.nunique()
            g['neural_ok']=True
            parts.append(g)
        except (ValueError,TypeError) as exc:
            errors.append(dict(molecule_id=mid,error=str(exc)))
            for col in ['neural_cos','neural_fp','neural_score','neural_std','n_spectra','n_adducts']:g[col]=0.
            parts.append(g)
        if (index+1)%25==0:print('NEURAL SCORED',index+1,'molecules',flush=True)
    result=pd.concat(parts,ignore_index=True)
    assert len(result)==len(frame)
    result.to_parquet(output,index=False)
    Path(str(output)+'.errors.json').write_text(json.dumps(errors,indent=2))
    print('NEURAL input failures kept at baseline:',len(errors),flush=True)
    print('NEURAL FEATURES',result.shape,checkpoint,flush=True)

if __name__=='__main__':score(*sys.argv[1:])

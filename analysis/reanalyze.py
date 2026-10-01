"""Reanalysis of the run-level results: writes the summary tables to analysis/tables/; no BO jobs are run."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/tables'; OUT.mkdir(parents=True,exist_ok=True)
files={k:ROOT/f'results/v2a_runs_{k}.csv' for k in ['P1A','P3','B1','B0','P2','S']}
frames={k:pd.read_csv(v) for k,v in files.items()}
main=pd.concat([frames['P1A'],frames['P3']],ignore_index=True)
assert not main.duplicated(['config','method','scenario','seed','round']).any()
assert main.groupby(['config','method','scenario','seed']).size().eq(5).all()
assert main.scenario.between(20,39).all()
assert main.groupby(['config','method','scenario']).seed.nunique().eq(5).all()
labels={'tabpfn_kb':'TabPFN KB','gp_bo_kb':'DS GP KB','gp_qlognei':'DS GP qLogNEI','gp_kb':'Matérn GP KB','tabpfn_lp':'TabPFN LP','lhs':'LHS','gp_lp':'Matérn GP LP','saasbo':'SAAS qLogNEI','turbo':'Fixed TR TS','gitbo_tabpfn':'FD subspace KB'}

def stat(v,tag):
    v=np.asarray(v,float); rng=np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8],16))
    boot=rng.choice(v,(20000,len(v)),replace=True).mean(1)
    w=int((v>0).sum()); nz=int((v!=0).sum())
    return dict(mean=float(v.mean()),median=float(np.median(v)),ci_lo=float(np.quantile(boot,.025)),ci_hi=float(np.quantile(boot,.975)),wins=w,n=len(v),p=float(binomtest(w,nz,.5).pvalue) if nz else 1.)

def holm(rows):
    prev=0
    for i,idx in enumerate(np.argsort([r['p'] for r in rows])):
        prev=max(prev,min(1.,(len(rows)-i)*rows[idx]['p']));rows[idx]['holm']=prev

def tables(df):
    raw=df[df['round']==4].groupby(['scenario','method']).best_true.mean().unstack()
    assert not raw.isna().any().any()
    lo,hi=raw.min(1),raw.max(1)
    return raw,raw.sub(lo,axis=0).div(hi-lo,axis=0),lo,hi

rows=[]; summary=[]; curves=[]; raw_rows=[]
for dim,d in main.groupby('n_sub'):
    raw,z,lo,hi=tables(d)
    for method in raw:
        summary.append(dict(d=int(dim),method=method,score=z[method].mean(),raw_mean=raw[method].mean()))
    for ref in ['gp_bo_kb','gp_qlognei','tabpfn_lp','gp_kb']:
        r=dict(d=int(dim),reference=ref,**stat(z.tabpfn_kb-z[ref],f'main{dim}{ref}'))
        rows.append(r)
        raw_rows.append(dict(d=int(dim),reference=ref,metric='raw biomass',**stat(raw.tabpfn_kb-raw[ref],f'raw{dim}{ref}')))
        raw_rows.append(dict(d=int(dim),reference=ref,metric='relative biomass',**stat(raw.tabpfn_kb/raw[ref]-1,f'rel{dim}{ref}')))
    for (r,m),s in d.groupby(['round','method']):
        x=s.groupby('scenario').best_true.mean(); zt=(x-lo)/(hi-lo)
        curves.append(dict(d=int(dim),round=int(r),method=m,mean=zt.mean(),se=zt.std(ddof=1)/np.sqrt(len(zt))))
# Retrospective families are explicit: five DS-GP-KB contrasts (including primary)
# and ten contrasts to qLogNEI / TabPFN LP. No claim of prospective registration.
holm([r for r in rows if r['reference']=='gp_bo_kb'])
holm([r for r in rows if r['reference'] in ['gp_qlognei','tabpfn_lp']])
holm([r for r in rows if r['reference']=='gp_kb'])
pd.DataFrame(rows).to_csv(OUT/'main_contrasts.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'scores.csv',index=False)
pd.DataFrame(raw_rows).to_csv(OUT/'raw_sensitivity.csv',index=False)
pd.DataFrame(curves).to_csv(OUT/'curves.csv',index=False)

emb=[]; es=[]
for (dim,config),d in frames['B1'].groupby(['n_sub','config']):
    kind='ambient' if config.endswith('_ambient') else 'rotate' if config.endswith('_rotate') else 'intrinsic'
    raw,z,_,_=tables(d)
    emb.append(dict(d=int(dim),embedding=kind,**stat(z.tabpfn_kb-z.gp_bo_kb,f'embed{dim}{kind}')))
    for m in raw:es.append(dict(d=int(dim),embedding=kind,method=m,score=z[m].mean()))
holm(emb)
pd.DataFrame(emb).to_csv(OUT/'embedding_contrasts.csv',index=False)
pd.DataFrame(es).to_csv(OUT/'embedding_scores.csv',index=False)

# Check overlaps, rather than count reused intrinsic runs as independent replication.
key=['config','method','scenario','seed','round']
overlap=frames['B1'].merge(main,on=key,suffixes=('_b','_m'))
inventory=[]
for k,d in frames.items():
    for cfg,g in d.groupby('config'):
        inventory.append(dict(file=k,config=cfg,rows=len(g),campaigns=len(g[key[:-1]].drop_duplicates()),scenarios=g.scenario.nunique(),seeds=g.seed.nunique(),arms=g.method.nunique()))
pd.DataFrame(inventory).to_csv(OUT/'inventory.csv',index=False)
timing=main[main['round']>0].groupby(['n_sub','method','device_actual']).wall_s.agg(['mean','median']).reset_index()
timing.to_csv(OUT/'timing.csv',index=False)
initial=main[main['round']==0].groupby(['config','scenario','seed']).best_true.agg(['min','max'])
manifest={'sources':{k:{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(frames[k])} for k,p in files.items()},'main_campaigns':len(main)//5,'B1_rows_overlapping_main':len(overlap),'overlap_max_best_true_difference':float((overlap.best_true_b-overlap.best_true_m).abs().max()),'initial_max_best_true_difference':float((initial['max']-initial['min']).max()),'main_missing_checkpoint_sha':int(main.ckpt_sha.isna().sum()),'note':'ANALYSIS_v2a pools embedding configurations in final_scores; not used. Main uses P1A/P3 only; B1 stratified by config. Bootstrap 20000 per contrast, stable hash-derived seeds. Secondary Holm families retrospective.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(emb).to_string(index=False));print(manifest)

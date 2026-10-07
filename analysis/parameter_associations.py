"""Post hoc scenario-level associations between growth-model descriptors and the gain of
TabPFN KB (S1 Fig, S3 Table); no optimization or ODE jobs are run.

    uv run python analysis/parameter_associations.py
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import simulator as sim,embed
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
OUT=ROOT/'analysis/tables/parameter_associations';OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'analysis/figures';FIG.mkdir(parents=True,exist_ok=True)
embed.EMBED='intrinsic';sim.ESS_P=.35;sim.W_CONC=.7
paths=[ROOT/f'results/v2a_runs_{p}.csv' for p in ['P1A','P3']]
runs=pd.concat([pd.read_csv(p) for p in paths])
features=['essential_count','mu_max','ks_mean','ks_cv','uptake_cv']
rng=np.random.default_rng(20261005);records=[];rows=[];sens=[]
for dim in [120,80]:
 sim.N_SUB=dim
 last=runs[(runs.n_sub==dim)&(runs['round']==4)]
 assert not last.duplicated(['scenario','method','seed']).any()
 assert last.groupby(['scenario','method']).seed.nunique().eq(5).all()
 m=last.groupby(['scenario','method']).best_true.mean().unstack()
 assert list(m.index)==list(range(20,40))
 vals=[]
 for seed in m.index:
  sc=sim.MonodScenario(int(seed));e=sc.essential;u=sc.w[e]/sc.Y[e]
  vals.append(dict(scenario=seed,essential_count=int(e.sum()),mu_max=sc.mu_max,ks_mean=sc.Ks[e].mean(),ks_cv=sc.Ks[e].std()/sc.Ks[e].mean(),uptake_cv=u.std()/u.mean()))
 x=pd.DataFrame(vals).set_index('scenario')
 y=m.tabpfn_kb/m.gp_bo_kb-1
 x['relative_gain']=y;x['d']=dim;records.append(x.reset_index())
 for col in features:
  xx=x[col].to_numpy();yy=y.to_numpy();rho=float(spearmanr(xx,yy).statistic)
  xr=rankdata(xx);yr=rankdata(yy);xr=(xr-xr.mean())/np.linalg.norm(xr-xr.mean());yr=(yr-yr.mean())/np.linalg.norm(yr-yr.mean())
  perms=np.array([rng.permutation(yr) for _ in range(99999)])@xr
  pv=(1+np.count_nonzero(np.abs(perms)>=abs(rho)-1e-12))/100000
  ids=rng.integers(0,20,(10000,20));bs=[]
  for idx in ids:
   r=spearmanr(xx[idx],yy[idx]).statistic
   if np.isfinite(r):bs.append(r)
  lo,hi=np.percentile(bs,[2.5,97.5]);loo=[spearmanr(np.delete(xx,i),np.delete(yy,i)).statistic for i in range(20)]
  rows.append(dict(d=dim,feature=col,rho=rho,p_perm=pv,ci_lo=lo,ci_hi=hi,loo_min=min(loo),loo_max=max(loo),n=20))
 for metric,yy in [('absolute_gain',m.tabpfn_kb-m.gp_bo_kb),('normalized_gap',(m.tabpfn_kb-m.gp_bo_kb)/(m.max(axis=1)-m.min(axis=1))),('relative_vs_qlognei',m.tabpfn_kb/m.gp_qlognei-1)]:
  sens.append(dict(d=dim,metric=metric,rho=spearmanr(x.essential_count,yy).statistic))
def holm(sub,key):
 prev=0
 for i,r in enumerate(sorted(sub,key=lambda r:r['p_perm'])):
  prev=max(prev,min(1,(len(sub)-i)*r['p_perm']));r[key]=prev
for d in [120,80]:holm([r for r in rows if r['d']==d],'holm_within_dimension')
holm(rows,'holm_all_ten')
pd.DataFrame(rows).to_csv(OUT/'associations.csv',index=False)
pd.concat(records).to_csv(OUT/'scenario_parameters.csv',index=False)
pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps(dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},simulator_sha256=hashlib.sha256((ROOT/'simulator.py').read_bytes()).hexdigest(),seed=20261005,permutations=99999,bootstrap=10000,notes='Post hoc screen of five descriptors at d=120; d=80 sensitivity uses the same scenario seeds and is not independent replication. Parameters reconstructed with the current scenario constructor; historical parameter arrays are not archived here.'),indent=2))
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
f,axs=plt.subplots(1,2,figsize=(7.5,3.5))
for ax,d,letter in zip(axs,[120,80],'AB'):
 z=pd.concat(records).query('d==@d');r=next(r for r in rows if r['d']==d and r['feature']=='essential_count')
 ax.scatter(z.essential_count,z.relative_gain*100,c='#0068A8',s=28,alpha=.85)
 ax.axhline(0,color='#999999',lw=.7)
 ax.set_xlabel('Realized number of essential components');ax.set_ylabel('TabPFN KB relative biomass gain (%)')
 ax.set_title(f'{letter}   {d} components',loc='left',weight='bold')
 ax.text(.04,.96,f"Spearman ρ = {r['rho']:.2f}\nHolm p = {r['holm_all_ten']:.4f}",transform=ax.transAxes,va='top',fontsize=8)
 ax.margins(y=.2)
f.tight_layout();f.savefig(FIG/'S1_Fig.png',dpi=400)
Image.open(FIG/'S1_Fig.png').convert('RGB').save(FIG/'S1_Fig.tif',compression='tiff_lzw',dpi=(400,400));plt.close(f)
print(pd.DataFrame(rows).round(4).to_string(index=False));print(pd.DataFrame(sens).round(4).to_string(index=False))

import sys; sys.path.insert(0,'bench')
import numpy as np, pandas as pd, scipy.sparse as sp
from run_graft_revisit import load
from kindling._native import kindling_core
from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.graph.metadata_smoothing import smoothing_graph
from kindling.item_features import ItemFeatureExtractor
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

train, test, items = load('h-and-m')
rng=np.random.default_rng(0); keep=set(rng.choice(train.entity_id.unique(),50000,replace=False).tolist())
train=train[train.entity_id.isin(keep)].copy(); test=test[test.entity_id.isin(keep)].copy()
item_ids=pd.Index(train.item_id.unique()); i2i={it:i for i,it in enumerate(item_ids)}; n=len(item_ids)
uidx=pd.factorize(train.entity_id)[0].astype(np.int64)
iidx=train.item_id.map(i2i).to_numpy().astype(np.int64)
nu=int(uidx.max())+1; w=np.ones(len(iidx),np.float32)
ts=(train.timestamp.astype('int64')//10**9).to_numpy().astype(np.float64)
counts=np.bincount(iidx,minlength=n).astype(np.float64)
def cooc(kernel):
    if kernel=='pure_count':
        d,ind,ip=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=n,kernel='pure_count')
    else:
        d,ind,ip=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=n,kernel='hybrid_temporal',alpha=1.0,half_life_days=30.0,timestamps=ts)
    d=apply_cooc_transform(np.asarray(d,np.float32),np.asarray(ind,np.int32),np.asarray(ip,np.int32),counts,nu,'wilson')
    return sp.csr_matrix((d,np.asarray(ind,np.int32),np.asarray(ip,np.int32)),shape=(n,n))
feat=ItemFeatureExtractor().fit_transform(items,i2i,n)
F=sp.csr_matrix((feat.data,feat.indices,feat.indptr),shape=(n,feat.n_features))
es=_build_eval_set(train,test,max_users=800,seed=0)
owned={u:set(g.item_id) for u,g in train.groupby('entity_id')}
warm=lambda u:(lambda k:"1-4" if k<=4 else("5-19" if k<20 else"20+"))(len(owned.get(u,())))
def ev(C,tag):
    per={"all":[],"1-4":[],"5-19":[],"20+":[]}
    for u,rel in es.items():
        ow=np.array([i2i[i] for i in owned.get(u,()) if i in i2i])
        if ow.size==0: continue
        rs={i2i[x] for x in rel if x in i2i}
        sc=np.asarray(C[:,ow].sum(axis=1)).ravel(); sc[ow]=-1e9
        top=np.argpartition(-sc,12)[:12]; top=top[np.argsort(-sc[top])]
        per["all"].append(([int(t) for t in top],rs)); per[warm(u)].append(([int(t) for t in top],rs))
    line=f"{tag:<34}"
    for b in ["all","1-4","5-19","20+"]:
        v=per[b]; r=aggregate(v,catalog_size=n,k=12) if v else None
        line+=f" {b}:{r.ndcg_at_k:.4f}/{r.recall_at_k:.4f}" if r else ""
    print(line, flush=True)
print(f"H&M 50k: items={n}  (ndcg/recall@12 by warmth)\n")
Cp=cooc('pure_count'); ev(Cp,"PLAIN cooc")
Cd=cooc('hybrid_temporal'); ev(Cd,"TIME-DECAYED cooc")
M,info=smoothing_graph(F,lambda ei,ej:np.asarray(Cp[ei,ej]).ravel(),n,topk=20,family='logistic')
print(f"\n[smoothing on PLAIN: slope={info.get('slope')} r2={info.get('fit_r2')} applied={info.get('applied')}]")
ev((Cp+M).tocsr() if info['applied'] else Cp,"PLAIN + SMOOTHED")
Md,infod=smoothing_graph(F,lambda ei,ej:np.asarray(Cd[ei,ej]).ravel(),n,topk=20,family='logistic')
print(f"[smoothing on DECAYED: slope={infod.get('slope')} r2={infod.get('fit_r2')} applied={infod.get('applied')}]")
ev((Cd+Md).tocsr() if infod['applied'] else Cd,"DECAYED + SMOOTHED")

# --- reconciliation: bench-style fixed-cap all-items smoothing on PLAIN cooc ---
ei2,ej2,es2 = kindling_core.metadata_knn(
    np.ascontiguousarray(F.data,np.float32), np.ascontiguousarray(F.indices,np.int32),
    np.ascontiguousarray(F.indptr,np.int32), int(F.shape[1]), 20, 0)
ei2=np.asarray(ei2,np.int64); ej2=np.asarray(ej2,np.int64); es2=np.asarray(es2,np.float64)
mx=float(Cp.data.max())
print(f"\n[fixed-cap arms: metadata edges={len(ei2):,}, plain max_obs={mx:.4f}]")
for cap in [0.02,0.05,0.1,0.2]:
    w2=es2*cap*mx
    Mc=sp.csr_matrix((np.concatenate([w2,w2]),(np.concatenate([ei2,ej2]),np.concatenate([ej2,ei2]))),shape=(n,n))
    ev((Cp+Mc).tocsr(), f"PLAIN + cap{cap}")

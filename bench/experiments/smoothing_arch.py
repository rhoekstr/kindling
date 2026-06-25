import sys; sys.path.insert(0,"bench")
import numpy as np, pandas as pd, scipy.sparse as sp
from run_graft_revisit import load
from kindling._native import kindling_core
from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.graph.metadata_smoothing import smoothing_graph
from kindling.item_features import ItemFeatureExtractor
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
train,test,items=load("h-and-m")
rng=np.random.default_rng(0); keep=set(rng.choice(train.entity_id.unique(),50000,replace=False).tolist())
train=train[train.entity_id.isin(keep)].copy(); test=test[test.entity_id.isin(keep)].copy()
ii=pd.Index(train.item_id.unique()); i2i={it:i for i,it in enumerate(ii)}; n=len(ii)
uidx=pd.factorize(train.entity_id)[0].astype(np.int64); iidx=train.item_id.map(i2i).to_numpy().astype(np.int64)
nu=int(uidx.max())+1; w=np.ones(len(iidx),np.float32); counts=np.bincount(iidx,minlength=n).astype(np.float64)
ts=(train.timestamp.astype('int64')//10**9).to_numpy().astype(np.float64)
def wilsonC(kernel):
    if kernel=='pure_count': d,a,p=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=n,kernel='pure_count')
    else: d,a,p=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=n,kernel='hybrid_temporal',alpha=1.0,half_life_days=30.0,timestamps=ts)
    d=apply_cooc_transform(np.asarray(d,np.float32),np.asarray(a,np.int32),np.asarray(p,np.int32),counts,nu,'wilson')
    return sp.csr_matrix((d,np.asarray(a,np.int32),np.asarray(p,np.int32)),shape=(n,n))
Cp=wilsonC('pure_count'); Ct=wilsonC('hybrid_temporal')
feat=ItemFeatureExtractor().fit_transform(items,i2i,n); F=sp.csr_matrix((feat.data,feat.indices,feat.indptr),shape=(n,feat.n_features))
Mp,_=smoothing_graph(F,lambda ei,ej:np.asarray(Cp[ei,ej]).ravel(),n,topk=20,cap=0.1,base_max=float(Cp.data.max()))
Mt,_=smoothing_graph(F,lambda ei,ej:np.asarray(Ct[ei,ej]).ravel(),n,topk=20,cap=0.1,base_max=float(Ct.data.max()))
CMp=(Cp+Mp).tocsr(); CMt=(Ct+Mt).tocsr()
es=_build_eval_set(train,test,max_users=800,seed=0); owned={u:set(g.item_id) for u,g in train.groupby('entity_id')}
COLD=10; ht=lambda x:"coldI" if (x in i2i and counts[i2i[x]]<=COLD) else "warmI"
def z(v): s=v.std(); return (v-v.mean())/s if s>0 else v*0
def evf(score,tag):
    per={"all":[],"coldI":[],"warmI":[]}
    for u,rel in es.items():
        ow=np.array([i2i[i] for i in owned.get(u,()) if i in i2i])
        if ow.size==0: continue
        sc=score(ow).copy(); sc[ow]=-1e9
        top=np.argpartition(-sc,12)[:12]; top=top[np.argsort(-sc[top])]; recs=[int(t) for t in top]
        rs={i2i[x] for x in rel if x in i2i}; per["all"].append((recs,rs))
        for x in rel:
            if x in i2i: per[ht(x)].append((recs,{i2i[x]}))
    o=f"{tag:<34}"
    for b in ["all","coldI","warmI"]:
        r=aggregate(per[b],catalog_size=n,k=12) if per[b] else None
        o+=f" {b}:{r.ndcg_at_k:.4f}/{r.recall_at_k:.4f}" if r else ""
    print(o,flush=True)
print(f"H&M 50k items={n}  [ndcg/recall@12]\n")
evf(lambda ow: np.asarray(Cp[:,ow].sum(1)).ravel(), "plain cooc")
evf(lambda ow: np.asarray(CMp[:,ow].sum(1)).ravel(), "smoothed cooc (base)")
evf(lambda ow: np.asarray(Ct[:,ow].sum(1)).ravel(), "time-decayed cooc")
evf(lambda ow: np.asarray(CMt[:,ow].sum(1)).ravel(), "smoothed time-decayed (time-in-base)")
evf(lambda ow: z(np.asarray(CMp[:,ow].sum(1)).ravel())+0.25*z(np.asarray(Ct[:,ow].sum(1)).ravel()), "smoothed base + time LAYER (0.25)")
evf(lambda ow: z(np.asarray(CMp[:,ow].sum(1)).ravel())+0.5*z(np.asarray(Ct[:,ow].sum(1)).ravel()), "smoothed base + time LAYER (0.5)")
print("\nDONE",flush=True)

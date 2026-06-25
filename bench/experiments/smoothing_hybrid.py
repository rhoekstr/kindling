import sys; sys.path.insert(0,"bench")
import numpy as np, pandas as pd, scipy.sparse as sp
from run_graft_revisit import load
from kindling._native import kindling_core
from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.item_features import ItemFeatureExtractor
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling import Engine

train,test,items=load("h-and-m")
rng=np.random.default_rng(0); keep=set(rng.choice(train.entity_id.unique(),50000,replace=False).tolist())
train=train[train.entity_id.isin(keep)].copy(); test=test[test.entity_id.isin(keep)].copy()
item_ids=pd.Index(train.item_id.unique()); i2i={it:i for i,it in enumerate(item_ids)}; n=len(item_ids)
uidx=pd.factorize(train.entity_id)[0].astype(np.int64); iidx=train.item_id.map(i2i).to_numpy().astype(np.int64)
nu=int(uidx.max())+1; w=np.ones(len(iidx),np.float32); counts=np.bincount(iidx,minlength=n).astype(np.float64)
d,ind,ip=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=n,kernel='pure_count')
d=apply_cooc_transform(np.asarray(d,np.float32),np.asarray(ind,np.int32),np.asarray(ip,np.int32),counts,nu,'wilson')
C=sp.csr_matrix((d,np.asarray(ind,np.int32),np.asarray(ip,np.int32)),shape=(n,n)); mx=float(C.data.max())
feat=ItemFeatureExtractor().fit_transform(items,i2i,n); F=sp.csr_matrix((feat.data,feat.indices,feat.indptr),shape=(n,feat.n_features))
ei,ej,es=kindling_core.metadata_knn(np.ascontiguousarray(F.data,np.float32),np.ascontiguousarray(F.indices,np.int32),np.ascontiguousarray(F.indptr,np.int32),int(F.shape[1]),20,0)
ei,ej,es=np.asarray(ei),np.asarray(ej),np.asarray(es)
cap=0.1; wts=es*cap*mx
M=sp.csr_matrix((np.concatenate([wts,wts]),(np.concatenate([ei,ej]),np.concatenate([ej,ei]))),shape=(n,n)).tocsr()
COLD_T=10; cold_item=(counts<=COLD_T)  # item-warmth mask for routing
es_set=_build_eval_set(train,test,max_users=800,seed=0)
owned={u:set(g.item_id) for u,g in train.groupby('entity_id')}
def held_tier(it): return "coldI" if (it in i2i and counts[i2i[it]]<=COLD_T) else "warmI"
def ev_direct(scorer,tag):
    per={"all":[],"coldI":[],"warmI":[]}
    for u,rel in es_set.items():
        ow=np.array([i2i[i] for i in owned.get(u,()) if i in i2i])
        if ow.size==0: continue
        sc=scorer(ow); sc[ow]=-1e9
        top=np.argpartition(-sc,12)[:12]; top=top[np.argsort(-sc[top])]; recs=[int(t) for t in top]
        rs={i2i[x] for x in rel if x in i2i}
        per["all"].append((recs,rs))
        for x in rel:
            if x in i2i: per[held_tier(x)].append((recs,{i2i[x]}))
    out=f"{tag:<26}"
    for b in ["all","coldI","warmI"]:
        r=aggregate(per[b],catalog_size=n,k=12) if per[b] else None
        out+=f" {b}:{r.ndcg_at_k:.4f}/{r.recall_at_k:.4f}" if r else ""
    print(out,flush=True)
us=lambda ow: np.asarray(C[:,ow].sum(axis=1)).ravel()
CM=(C+M).tocsr()
sm=lambda ow: np.asarray(CM[:,ow].sum(axis=1)).ravel()
def hyb(ow):
    base=np.asarray(C[:,ow].sum(axis=1)).ravel(); mcontrib=np.asarray(M[:,ow].sum(axis=1)).ravel()
    return base + mcontrib*cold_item   # cold items get +M, hot items pure cooc
print(f"H&M 50k items={n} cold_items(<= {COLD_T})={int(cold_item.sum())}  [ndcg/recall@12]\n")
ev_direct(us,"bare cooc")
ev_direct(sm,"cooc + smoothed(all)")
ev_direct(hyb,"cooc + HYBRID(cold-route)")
# full engine for the #1 comparison (same eval_set)
def eng_ev(tag,**kw):
    e=Engine(retrieval_budget=500,random_state=0,open_catalog=False,**kw).fit(train,item_metadata=items)
    per={"all":[],"coldI":[],"warmI":[]}
    for u,rel in es_set.items():
        recs=[r.item_id for r in e.recommend(entity_id=u,n=12)]; ridx=[i2i[r] for r in recs if r in i2i]
        rs={i2i[x] for x in rel if x in i2i}; per["all"].append((ridx,rs))
        for x in rel:
            if x in i2i: per[held_tier(x)].append((ridx,{i2i[x]}))
    out=f"{tag:<26}"
    for b in ["all","coldI","warmI"]:
        r=aggregate(per[b],catalog_size=e._state.n_items,k=12) if per[b] else None
        out+=f" {b}:{r.ndcg_at_k:.4f}/{r.recall_at_k:.4f}" if r else ""
    print(out,flush=True)
print()
eng_ev("FULL engine off",metadata_smoothing="off")
eng_ev("FULL engine + smooth0.1",metadata_smoothing="on",metadata_smoothing_cap=0.1)
print("\nDONE",flush=True)

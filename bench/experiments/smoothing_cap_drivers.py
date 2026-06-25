import numpy as np, scipy.sparse as sp
from kindling._native import kindling_core
from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.item_features import ItemFeatureExtractor
from kindling.benchmarks.metrics import aggregate
import pandas as pd
NI,NU,D=2000,9000,6
def gen(noise,skew,seed=0):
    rng=np.random.default_rng(seed); T=rng.normal(size=(NI,D)); U=rng.normal(size=(NU,D))
    bias=rng.normal(0,skew,NI)  # higher skew -> heavier popularity tail (more cold)
    rows=[]
    for u in range(NU):
        lg=-((U[u]-T)**2).sum(1)+bias; p=np.exp(lg-lg.max()); p/=p.sum()
        k=int(rng.integers(2,14))
        for it in rng.choice(NI,size=k,replace=False,p=p): rows.append((u,int(it)))
    df=pd.DataFrame(rows,columns=['entity_id','item_id'])
    M=T+rng.normal(0,noise,(NI,D)); meta={'item_id':list(range(NI))}
    for j in range(D):
        e=np.quantile(M[:,j],np.linspace(0,1,6)); meta[f'd{j}']=[f'd{j}b{int(np.clip(np.digitize(M[i,j],e[1:-1]),0,4))}' for i in range(NI)]
    return df,pd.DataFrame(meta)
def prep(df,meta):
    rng=np.random.default_rng(1); tr=[]; held={}
    for u,g in df.groupby('entity_id'):
        its=g.item_id.tolist()
        if len(its)>=4: held[u]=its.pop(int(rng.integers(len(its))))
        tr+=[(u,i) for i in its]
    tr=pd.DataFrame(tr,columns=['entity_id','item_id'])
    iidx=tr.item_id.to_numpy().astype(np.int64); uidx=pd.factorize(tr.entity_id)[0].astype(np.int64)
    nu=int(uidx.max())+1; w=np.ones(len(iidx),np.float32); counts=np.bincount(iidx,minlength=NI).astype(np.float64)
    d,ind,ip=kindling_core.build_cooccurrence(uidx,iidx,w,n_users=nu,n_items=NI,kernel='pure_count')
    d=apply_cooc_transform(np.asarray(d,np.float32),np.asarray(ind,np.int32),np.asarray(ip,np.int32),counts,nu,'wilson')
    C=sp.csr_matrix((d,np.asarray(ind,np.int32),np.asarray(ip,np.int32)),shape=(NI,NI))
    feat=ItemFeatureExtractor().fit_transform(meta,{i:i for i in range(NI)},NI)
    F=sp.csr_matrix((feat.data,feat.indices,feat.indptr),shape=(NI,feat.n_features))
    ei,ej,es=kindling_core.metadata_knn(np.ascontiguousarray(F.data,np.float32),np.ascontiguousarray(F.indices,np.int32),np.ascontiguousarray(F.indptr,np.int32),int(F.shape[1]),20,0)
    ei,ej,es=np.asarray(ei),np.asarray(ej),np.asarray(es)
    return C,counts,held,ei,ej,es,float(C.data.max()),tr
def ev(C,counts,held,tr):
    own={u:set(g.item_id) for u,g in tr.groupby('entity_id')}
    wr=lambda u:(lambda k:"cold" if k<=3 else"warm")(len(own.get(u,())))
    per={"all":[]}
    pairs=[]
    for u,h in held.items():
        ow=np.array(list(own.get(u,())))
        if ow.size==0: continue
        sc=np.asarray(C[:,ow].sum(axis=1)).ravel(); sc[ow]=-1e9
        top=np.argpartition(-sc,12)[:12]; top=top[np.argsort(-sc[top])]
        pairs.append(([int(t) for t in top],{h}))
    r=aggregate(pairs,catalog_size=NI,k=12); return r.ndcg_at_k
for noise,skew in [(0.3,1.2),(0.3,3.0),(1.0,1.2),(1.0,3.0)]:
    df,meta=gen(noise,skew); C,counts,held,ei,ej,es,mx,tr=prep(df,meta)
    gini=1-2*np.sum(np.cumsum(np.sort(C.data))/C.data.sum())/len(C.data)
    coldfrac=float((counts<=3).mean())
    base=ev(C,counts,held,tr); row=[]
    for cap in [0.0,0.02,0.05,0.1,0.2,0.4]:
        if cap==0: row.append(base); continue
        w=es*cap*mx; M=sp.csr_matrix((np.concatenate([w,w]),(np.concatenate([ei,ej]),np.concatenate([ej,ei]))),shape=(NI,NI))
        row.append(ev((C+M).tocsr(),counts,held,tr))
    caps=[0,0.02,0.05,0.1,0.2,0.4]; opt=caps[int(np.argmax(row))]
    print(f"noise={noise} skew={skew} | cooc_gini={gini:.2f} coldfrac={coldfrac:.2f} max_obs={mx:.3f} | ndcg by cap {['%.4f'%x for x in row]} | OPT cap={opt}",flush=True)

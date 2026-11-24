import argparse, ctypes, sys, json, time
from pathlib import Path
from functools import lru_cache
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/"output"/HERE.name
NAMES=['bike','bio','community','concrete','homes','star','legislator','skin']
def load(name):
    with np.load(OUTPUT/(name+'.npz')) as f:
        return (*[np.ascontiguousarray(f[k],dtype=float) for k in ('x','s','bounds')],*[int(f[k]) for k in ('n','m','k','depth')])
@lru_cache(None)
def engine():
    lib=ctypes.CDLL(str(OUTPUT/'trees.so')); fn=lib.simulate
    fn.argtypes=[np.ctypeslib.ndpointer(dtype=float,flags='C_CONTIGUOUS')]*4+[ctypes.c_int]*6+[np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')]
    fn.restype=None;return fn
@lru_cache(maxsize=20000)
def region(key):
    path=sorted((int(code) for code in np.frombuffer(key,dtype=np.uint8) if code!=255),key=lambda code:code//2)
    return ','.join(map(str,path))
def values(name,data,phase,rep,queries,verify_repo=None):
    x,score,bounds,n,m,k,depth=data
    rng=np.random.default_rng([2026112305,NAMES.index(name),phase,rep])
    ids=rng.choice(len(x),n+1,replace=False); a=x[ids]; raw=score[ids]
    fit=raw+rng.uniform(0,1e-12,n+1)
    out=np.full((queries,3,depth+1),-1,np.int32)
    engine()(a,fit,raw,bounds,n+1,a.shape[1],m,k,depth,queries,out)
    if verify_repo is not None:
        sys.path.insert(0,str(Path(verify_repo)/'conformal_tree/_utils'))
        from fast_dyadic import DyadicSummaryCache,conformal_threshold
        # Reconstruct each of the three fits independently of the C++ implementation.
        full=DyadicSummaryCache(a,fit,bounds,m,k,depth); ft=full.fit()
        for q in range(queries):
            keep=np.arange(n+1)!=q
            cal=DyadicSummaryCache(a[keep],fit[keep],bounds,m,k,depth)
            for mode,cache,tree in [(0,full,ft),(1,cal,cal.fit(a[q])),(2,cal,cal.fit())]:
                path=tree.route_one(a[q]); idx=cache.indices(path)
                scores=raw[idx[idx!=q]] if mode==0 else raw[keep][idx]
                assert out[q,mode,0]==int(raw[q]<=conformal_threshold(scores,.1))
                assert [c for c in out[q,mode,1:] if c!=-1]==[2*j+b for j,b in path],(name,q,mode)
    paths=out[:,:,1:].astype(np.uint8)
    return [[region(v.tobytes()) for v in paths[:,j]] for j in range(3)],out[:,:,0]
def choose(freq,total):
    f=[{c:n/total for c,n in ct.items()} for ct in freq]
    pool={c for c in f[0] if c and min(f[0][c],f[1].get(c,0))>=.005}
    avg=lambda c:(f[0].get(c,0)+f[1].get(c,0))/2
    chosen=[]
    for target in [.2,.08,.02]:
        if not pool:break
        c=min(pool,key=lambda v:(abs(np.log(avg(v)/target)),v));pool.remove(c);chosen.append(c)
    if pool: chosen.append(min(pool,key=lambda v:(-avg(v),v)))
    if not chosen:chosen=['']
    return chosen
def dataset(task):
    name,reps,pilot,queries,repo=task; data=load(name); start=time.monotonic()
    for i in range(2):values(name,data,0,i,4,repo)
    print(name,'passed independent tree/rank checks',flush=True)
    freq=[Counter(),Counter()]
    for i in range(pilot):
        paths,_=values(name,data,1,i,200)
        for f,path in zip(freq,paths[1:]):f.update(path)
    chosen=choose(freq,pilot*200)
    print(name,'selected',len(chosen),'pilot cells',flush=True)
    # Rows are genuinely independent full samples. Queries within a row are paired.
    counts=np.zeros((reps,2,len(chosen),4),np.int32)
    for i in range(reps):
        paths,cov=values(name,data,2,i,queries)
        full=np.array(paths[0])
        for j in range(2):
            leaves=np.array(paths[j+1])
            for k,c in enumerate(chosen):
                h=leaves==c;f=full==c
                counts[i,j,k]=[h.sum(),(h&~f).sum(),(f&~h).sum(),cov[h,j+1].sum()]
        if (i+1)%500==0:print(name,i+1,'validation draws of full sample',flush=True)
    np.savez_compressed(OUTPUT/(name+'_results.npz'),counts=counts,cells=chosen,queries=queries,pilot=pilot,reps=reps)
    print(name,'complete in',round(time.monotonic()-start),'s',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=HERE.parents[1])
    p.add_argument('--reps',type=int,default=10000);p.add_argument('--pilot',type=int,default=200);p.add_argument('--queries',type=int,default=40)
    p.add_argument('--jobs',type=int,default=6);p.add_argument('--datasets',nargs='+',default=NAMES)
    args=p.parse_args()
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(dataset,[(n,args.reps,args.pilot,args.queries,args.repo) for n in args.datasets]))

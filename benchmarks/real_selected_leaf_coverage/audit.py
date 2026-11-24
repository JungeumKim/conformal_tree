import sys
from pathlib import Path
import numpy as np
from run import NAMES,load,engine,region
from concurrent.futures import ProcessPoolExecutor
def check(task):
    name,repo=task;sys.path.insert(0,str(Path(repo)/'conformal_tree/_utils'))
    from fast_dyadic import DyadicSummaryCache,conformal_threshold
    x,s,bounds,n,m,k,depth=load(name);Q=min(n+1,450);tested=changed=0
    rng=np.random.default_rng([159,NAMES.index(name)])
    for i in range(15):
        ids=rng.choice(len(x),n+1,replace=False);a=x[ids];raw=s[ids];fit=raw+rng.uniform(0,1e-12,n+1)
        out=np.full((Q,3,depth+1),-1,np.int32)
        engine()(a,fit,raw,bounds,n+1,a.shape[1],m,k,depth,Q,out)
        diff=np.where(np.any(out[:,0,1:]!=out[:,1,1:],1) | np.any(out[:,1,1:]!=out[:,2,1:],1))[0]
        if not len(diff) and i:continue
        cache=DyadicSummaryCache(a,fit,bounds,m,k,depth);tree=cache.fit()
        for q in (diff[:2] if len(diff) else [0]):
            keep=np.arange(n+1)!=q
            cal=DyadicSummaryCache(a[keep],fit[keep],bounds,m,k,depth)
            for j,cc,tr in [(0,cache,tree),(1,cal,cal.fit(a[q])),(2,cal,cal.fit())]:
                path=tr.route_one(a[q]);expected=[2*j+b for j,b in path]
                assert expected==out[q,j,1:1+len(expected)].tolist() and all(out[q,j,1+len(expected):]==-1)
                ids=cc.indices(path);v=raw[ids[ids!=q]] if j==0 else raw[keep][ids]
                assert out[q,j,0]==int(raw[q]<=conformal_threshold(v,.1))
            tested+=1;changed+=int(len(diff)>0)
        if changed>=4:break
    return name,tested,changed
if __name__=='__main__':
    with ProcessPoolExecutor(4) as pool:
        for row in pool.map(check,[(n,sys.argv[1]) for n in NAMES]):print(*row,flush=True)

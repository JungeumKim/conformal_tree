"""Paired Monte Carlo for the two paper DGPs. See REPORT.md for the design."""
import argparse
import ctypes
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
NAMES = ["dyadic range", "dyadic SSE", "all cuts range", "all cuts SSE"]
SEED = 20261123


def draw(rng, dataset, shape):
    x = rng.uniform(0, 10, shape)
    if dataset == 1:
        mean, sd = 3*np.sin(4/x+0.2)+1.5, x
    else:
        mean, sd = np.sin(x**-3), 0.1
    return x, rng.normal(mean, sd)


def predictor(dataset, b):
    rng = np.random.default_rng([SEED, 0, dataset, b])
    x, y = draw(rng, dataset, 300)
    scale = np.mean(abs(y))
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                  max_features="sqrt", n_jobs=1,
                                  random_state=int(rng.integers(2**31)))
    model.fit(x[:, None], y/scale)
    # In one dimension the fixed forest is piecewise constant. At each split
    # find the smallest float32 input strictly greater than sklearn's threshold.
    raw = np.unique(np.concatenate([t.tree_.threshold[t.tree_.feature == 0]
                                    for t in model.estimators_]))
    floor = raw.astype("float32")
    starts = np.unique(np.where(floor.astype("float64") > raw, floor,
                                np.nextafter(floor, np.float32(np.inf))))
    values = model.predict(np.r_[np.float32(0), starts][:, None])
    def predict(z):
        return values[np.searchsorted(starts,z.astype("float32"),side="right")]
    check = np.r_[starts,np.nextafter(starts,np.float32(-np.inf)),rng.uniform(0,10,300)]
    np.testing.assert_array_equal(predict(check),model.predict(check[:, None]))
    return scale, predict


def slow_fit(x, scores, query, hide, kmax, method, rate):
    # Independent brute-force calculation with direct slicing and recomputation.
    leaves = [(0., 10., 0)]
    while len(leaves) < kmax:
        winner = None
        for j,(lo,hi,dep) in enumerate(leaves):
            ids = np.flatnonzero((x>lo)&(x<=hi))
            if len(ids)<40 or dep>=8:
                continue
            cuts = (x[ids[:-1]]+x[ids[1:]])/2 if method>=2 else [(lo+hi)/2]
            def loss(ids):
                vis = scores[ids[ids!=query] if hide else ids]
                return np.sum((vis-vis.mean())**2) if method%2 else np.ptp(vis)
            parent = loss(ids)
            for c in cuts:
                l,r = ids[x[ids]<=c],ids[x[ids]>c]
                if min(len(l),len(r))<20 or x[l[-1]]==x[r[0]]:
                    continue
                child = (loss(l)+loss(r)) / (1 if method%2 else 2)
                gain = parent-child
                if parent>(1+rate)*child and (winner is None or gain>winner[0]):
                    winner = gain,j,c
        if winner is None:
            break
        _,j,c = winner
        lo,hi,dep = leaves[j]
        leaves[j:j+1] = [(lo,c,dep+1),(c,hi,dep+1)]
    leaf = next((lo,hi) for lo,hi,_ in leaves if lo<x[query]<=hi)
    return leaf,len(leaves),tuple(v[1] for v in leaves[:-1])


def verify(fn, repo):
    spec = importlib.util.spec_from_file_location("package_fast_dyadic",
               repo/"conformal_tree/_utils/fast_dyadic.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng([SEED, 99])
    for case in range(40):
        n = [80,137,201,499][case%4]
        x,y = draw(rng, 1+case%2, n+1)
        perm = x.argsort()
        x,s = x[perm],abs(y[perm])
        q,k = int(rng.integers(n+1)), (4,8)[case%2]
        out = np.empty((4,8))
        rate=0.05 if case%3 else 0.
        fn(x,s,n+1,q,k,rate,out)
        for hide in (False,True):
            vis = np.arange(n+1)!=q if hide else np.ones(n+1,bool)
            cache = module.DyadicSummaryCache(x[vis,None],s[vis],np.array([[0.,10.]]),20,k,threshold=rate)
            tree = cache.fit(x[q:q+1] if hide else None)
            idx = 5 if hide else 3
            np.testing.assert_array_equal(cache.bounds(tree.route_one(x[q:q+1]))[0],
                                          out[0,idx:idx+2])
            assert tree.n_leaves == out[0,2 if hide else 7]
        for method in range(4):
            f,fl,fc = slow_fit(x,s,q,False,k,method,rate)
            h,hl,hc = slow_fit(x,s,q,True,k,method,rate)
            np.testing.assert_array_equal(out[method],[f!=h,fc!=hc,hl,*f,*h,fl])
    print("40 paired validation cases passed for all four methods",flush=True)


def main(fn,args):
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_checks:
        verify(fn,args.repo)
    sample_sizes = [249,499,999]
    methods = np.empty((2,len(sample_sizes),args.models,args.reps,4,3),dtype=np.uint8)
    zero_threshold = np.empty((2,args.models,args.reps,4,3),dtype=np.uint8)
    for ds in [1,2]:
        for b in range(args.models):
            scale,predict=predictor(ds,b)
            for j,n in enumerate(sample_sizes):
                rng=np.random.default_rng([SEED,1,ds,b,n])
                out=np.empty((4,8))
                for start in range(0,args.reps,48):
                    x,y=draw(rng,ds,(min(48,args.reps-start),n+1))
                    scores=abs(y/scale-predict(x))+rng.uniform(0,1e-12,x.shape)
                    order=np.argsort(x,axis=1)
                    xs=np.take_along_axis(x,order,axis=1)
                    ss=np.take_along_axis(scores,order,axis=1)
                    query=(order==n).argmax(axis=1)
                    for i,q in enumerate(query):
                        fn(xs[i],ss[i],n+1,q,4 if ds==1 else 8,0.05,out)
                        methods[ds-1,j,b,start+i]=out[:,:3]
                        if n==499:
                            fn(xs[i],ss[i],n+1,q,4 if ds==1 else 8,0.,out)
                            zero_threshold[ds-1,b,start+i]=out[:,:3]
            print(f"Data {ds}, completed predictor {b+1}/{args.models}",flush=True)
    np.savez_compressed(args.output/"trials.npz",results=methods,n=sample_sizes,
                        zero_threshold_n499=zero_threshold,
                        metrics=["query_leaf_changed","partition_changed","hidden_leaves"],
                        methods=NAMES,seed=SEED)
    rows=[]
    for ds in [1,2]:
        cap=4 if ds==1 else 8
        for j,n in enumerate(sample_sizes):
            a=methods[ds-1,j].astype(float)
            # Standard errors conditional on the fixed, equally weighted predictor panel.
            def summarize(v):
                se=np.sqrt(v.var(axis=1,ddof=1).mean(axis=0)/(args.models*args.reps))
                return dict(mean=v.mean(axis=(0,1)).tolist(),se=se.tolist())
            rows.append(dict(dataset=ds,n=n,K=cap,bound=(4*cap-4)/(n+1),
                             query=summarize(a[:,:,:,0]),tree=summarize(a[:,:,:,1]),
                             leaves=summarize(a[:,:,:,2]),
                             paired_vs_dyadic_range=summarize(a[:,:,:,0]-a[:,:,0:1,0])))
    (args.output/"summary.json").write_text(json.dumps(dict(models=args.models,
        reps_per_model=args.reps,methods=NAMES,rows=rows),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,default=HERE.parents[1])
    p.add_argument("--models",type=int,default=8)
    p.add_argument("--reps",type=int,default=1500)
    p.add_argument("--skip-checks",action="store_true")
    p.add_argument("--output",type=Path,default=HERE.parent/"output"/HERE.name)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        lib=Path(tmp)/"tree.so"
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(HERE/"tree.cpp"),
                        "-o",str(lib)],check=True)
        fn=ctypes.CDLL(str(lib)).compare
        arr=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        fn.argtypes=[arr,arr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,arr]
        fn.restype=None
        main(fn,args)

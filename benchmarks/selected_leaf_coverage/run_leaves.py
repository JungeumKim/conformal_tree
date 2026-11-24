"""Leaf-specific stability and coverage for two fixed independently trained predictors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import beta

HERE=Path(__file__).resolve().parent
SEED=2026112309
N=500
TARGETS=(.20,.08,.02)
NCELLS=512


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def setup(base,lib,ds):
    bench=load_module("q3_benchmark",Path(base)/"run.py")
    scale,predict=bench.predictor(ds,0)
    fn=ctypes.CDLL(lib).classify_batch
    dbl=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,dbl,dbl,dbl,integer,integer]
    fn.restype=None
    return bench.draw,scale,predict,fn


def cell(bounds):
    left,right=bounds
    w=round(10/(right-left))
    return w+round(left*w/10)


def interval(a):
    w=2**(int(a).bit_length()-1)
    return [10*(a-w)/w,10*(a-w+1)/w]


def simulate(task):
    base,lib,ds,phase,block,reps,repo,ordinary=task
    draw,scale,predict,fn=setup(base,lib,ds)
    rng=np.random.default_rng([SEED,ds,phase,block])
    cap={1:4,2:8}[ds]
    outputs=[]
    if phase==0:
        ref=load_module("q3_reference",Path(repo)/"conformal_tree/_utils/fast_dyadic.py")
    for start in range(0,reps,128):
        batch=min(128,reps-start)
        x,y=draw(rng,ds,(batch,N))
        raw=abs(y/scale-predict(x))
        fitting=raw+rng.uniform(0,1e-12,raw.shape)
        order=np.argsort(x,axis=1)
        q=(order==N-1).argmax(axis=1).astype(np.int32)
        sx,sr,sf=[np.ascontiguousarray(np.take_along_axis(v,order,axis=1)) for v in [x,raw,fitting]]
        out=np.empty((batch,4),dtype=np.int32)
        fn(batch,N,cap,sx,sr,sf,q,out)
        if phase==0:
            for j in range(batch):
                ids=[]
                for hidden in (True,False):
                    vis=np.arange(N)!=q[j] if hidden else np.ones(N,bool)
                    cache=ref.DyadicSummaryCache(sx[j,vis,None],sf[j,vis],np.array([[0.,10.]]),20,cap)
                    test=sx[j,q[j]:q[j]+1]
                    fitted=cache.fit(test if hidden and not ordinary else None)
                    path=fitted.route_one(test)
                    ids.append(cell(cache.bounds(path)[0]))
                    if hidden:
                        leaf_raw=sr[j,vis][cache.indices(path)]
                        covered=int(sr[j,q[j]]<=ref.conformal_threshold(leaf_raw,.1))
                        leaf_count=len(leaf_raw)
                np.testing.assert_array_equal(out[j],[*ids,covered,leaf_count])
        outputs.append(out.astype(np.uint16))
    return np.concatenate(outputs)


def lock_cells(pilot):
    freq=np.bincount(pilot[:,0],minlength=NCELLS)/len(pilot)
    pool=[a for a in range(2,NCELLS) if freq[a]>=.005]
    chosen=[]
    for p in TARGETS:
        a=min(pool,key=lambda i:(abs(math.log(freq[i]/p)),i))
        chosen.append(a)
        pool.remove(a)
    # The mean function of Data 2 oscillates near zero. Select an additional
    # left-edge cell using only this independent pilot and no test outcomes.
    left=[a for a in pool if interval(a)[0]==0]
    if left:
        chosen.append(max(left,key=lambda i:(freq[i],-i)))
    return chosen


def lower(k,total,tail):
    return float(beta.ppf(tail,k,total-k+1)) if k else 0.


def upper(k,total,tail):
    return float(beta.ppf(1-tail,k+1,total-k)) if k<total else 1.


def summarize(data,selected):
    h,f,ok,cals=data.T
    changed=h!=f
    rows=[]
    for a in sorted(set(map(int,np.r_[h,f]))):
        here=h==a
        full=f==a
        den=int(here.sum())
        loss=int((here&~full).sum())
        gain=int((~here&full).sum())
        num=loss+gain
        covers=int(ok[here].sum())
        row=dict(cell=a,interval=interval(a),H_count=den,H_only=loss,F_only=gain,
                 disagreements=num,covered=covers,rho=den/len(h),delta=num/len(h))
        if den:
            ratio=num/den
            row.update(penalty=ratio,plugin=max(0,.9-ratio),coverage=covers/den,
                       coverage_ci=[lower(covers,den,.025),upper(covers,den,.025)],
                       mean_cal_count=float(cals[here].mean()))
        if a in selected and den:
            # Protect two one-sided bounds for each of at most eight pilot-selected cells.
            tail=.05/(2*(len(TARGETS)+1)*2)
            ub=upper(num,len(h),tail)/lower(den,len(h),tail)
            row.update(simultaneous_lower=max(0,.9-ub))
        rows.append(row)
    assert sum(r["disagreements"] for r in rows)==2*changed.sum()
    assert sum(r["H_count"] for r in rows)==len(data)
    return dict(T=len(data),query_changes=int(changed.sum()),overall_coverage=float(ok.mean()),
                selected=selected,cells=rows)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lib=str(Path(tmp)/"leaves.so")
        subprocess.run(["c++",*(["-DORDINARY_TREE"] if args.ordinary else []),"-O3","-std=c++17","-shared","-fPIC","-I",str(args.base),
                        str(HERE/"leaves.cpp"),"-o",lib],check=True)
        for ds in (1,2):
            simulate((args.base,lib,ds,0,0,128,args.repo,args.ordinary))
        print("256 package comparisons passed (both leaves, calibration count and coverage).",flush=True)
        def tasks(ds,phase,total):
            return [(args.base,lib,ds,phase,i,min(10000,total-i*10000),args.repo,args.ordinary)
                    for i in range((total+9999)//10000)]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            selection={}
            for ds in (1,2):
                pilot=np.concatenate(list(pool.map(simulate,tasks(ds,1,args.pilot))))
                selection[ds]=lock_cells(pilot)
                print("Locked pilot leaves",ds,[interval(a) for a in selection[ds]],flush=True)
            (args.out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
            results={}
            for ds in (1,2):
                pieces=[]
                for i,chunk in enumerate(pool.map(simulate,tasks(ds,2,args.reps))):
                    pieces.append(chunk)
                    if (i+1)%10==0:
                        print(f"Data {ds}: {(i+1)*10000:,} validation draws",flush=True)
                a=np.concatenate(pieces)
                results[ds]=summarize(a,selection[ds])
                np.savez_compressed(args.out/f"data{ds}.npz",trials=a,
                                    columns=["ordinary_leaf_id" if args.ordinary else "hidden_leaf_id","full_leaf_id","covered","calibration_count"])
            (args.out/"results.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=HERE.parent/"cut_criterion_stability")
    p.add_argument("--repo",type=Path,default=HERE.parents[1])
    p.add_argument("--jobs",type=int,default=4)
    p.add_argument("--pilot",type=int,default=12000)
    p.add_argument("--reps",type=int,default=400000)
    p.add_argument("--ordinary",action="store_true",help="exclude the query from the tree fit")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.out is None:
        args.out=HERE.parent/"output"/("ordinary_selected_leaf" if args.ordinary else HERE.name)
    main(args)

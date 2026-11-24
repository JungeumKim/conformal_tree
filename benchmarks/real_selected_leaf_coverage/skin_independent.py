"""Run the classification boundary case with one fresh calibration sample per query."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run import OUTPUT, load, values
def chunk(args):
    start,end,cells=args;data=load('skin');counts=np.zeros((2,len(cells),4),np.int64)
    for i in range(start,end):
        paths,cov=values('skin',data,3,i,1)
        f=paths[0][0]
        for j in range(2):
            h=paths[j+1][0]
            for k,c in enumerate(cells):
                if h==c:counts[j,k,0]+=1;counts[j,k,3]+=cov[0,j+1]
                if h!=f:
                    if h==c:counts[j,k,1]+=1
                    if f==c:counts[j,k,2]+=1
    return counts
if __name__=='__main__':
    with np.load(OUTPUT/'skin_results.npz') as z:cells=z['cells'].tolist()
    with ProcessPoolExecutor(6) as pool:
        parts=list(pool.map(chunk,[(i,i+10000,cells) for i in range(0,400000,10000)]))
    np.savez_compressed(OUTPUT/'skin_independent.npz',counts=np.sum(parts,axis=0),cells=cells)

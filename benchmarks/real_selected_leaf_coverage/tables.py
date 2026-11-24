import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from run import NAMES,HERE,OUTPUT
def rule(path,features,bounds):
    box=np.array(bounds,copy=True); limits={}
    for code in map(int,filter(None,path.split(','))):
        j,b=divmod(code,2);v=float(box[j].mean());ends=limits.setdefault(j,[None,None]);ends[1-b]=v;box[j,1-b]=v
    return [{'feature':features[j],'column':j+1,'lower':l,'upper':u} for j,(l,u) in sorted(limits.items())]
def display(r):
    parts=[]
    for e in r:
        f,l,u=[e[k] for k in ('feature','lower','upper')]
        if l is None: parts.append(f'{f} ≤ {u:.6g}')
        elif u is None: parts.append(f'{f} > {l:.6g}')
        else:parts.append(f'{l:.6g} < {f} ≤ {u:.6g}')
    return ', '.join(parts) or 'Whole domain'
def esc(s):return s.replace('_',r'\_').replace('≤',r'\(\le\)').replace(' > ',r' \(>\) ').replace(' < ',r' \(<\) ')
def build(count_dir):
    design_info=json.loads((HERE/"design.json").read_text())
    OUTPUT.mkdir(exist_ok=True,parents=True)
    result={}; rows=[[],[]];keys=[];tex=[]
    design='For each regression dataset we held the first saved benchmark predictor fixed and resampled its calibration and test records. For classification we used the saved model probabilities and scored the true class by one minus its predicted probability. We used the original benchmark leaf caps and minimum child sizes. All methods used the same draws. An independent pilot selected four nonroot cells per dataset using only their selection frequencies, pooled across the two fits.'
    mc='Except for skin disease, we drew 10,000 independent samples of size '+r'\(n+1\)'+' without replacement from the available records. Within each sample we took 40 observations in turn as the test point, giving 400,000 evaluations. We used pointwise 95% Monte Carlo intervals for coverage, with standard errors computed over the 10,000 independent samples. For skin disease, coverage in some pilot-selected cells was close to 100%; we used 400,000 independent calibration and test draws (one query per sample) and exact binomial intervals. The estimates describe repeated splits of these records with the predictor fixed.'
    for name in NAMES:
        z=np.load(count_dir/(name+'_results.npz'));c=z['counts'];B=int(z['reps']);Q=int(z['queries'])
        assert (B,Q)==(10000,40)
        info=design_info[name]
        bounds=np.asarray(info['bounds'])
        result[name]={k:info[k] for k in ('n','m','K','pool')}
        result[name]['cells']=[]
        for j,path in enumerate(z['cells'].tolist()):
            geo=rule(path,info['features'],bounds);label='ABCD'[j];entry=dict(label=label,geometry=geo);result[name]['cells'].append(entry)
            keys.append([name,label,display(geo)])
            for mode,key in enumerate(['hallucination','ordinary']):
                d=c[:,mode,j];h,loss,gain,covered=d.sum(0).tolist()
                assert 0<h<=B*Q and covered<=h
                def estimate(index):
                    num=d[:,index] if isinstance(index,int) else d[:,index].sum(axis=1)
                    v=num.sum()/h
                    se=(num-v*d[:,0]).std(ddof=1)/(np.sqrt(B)*d[:,0].mean())
                    return v,se
                rate,se=estimate(3);pen,ps=estimate([1,2]); rho=h/(B*Q);lower=max(0,.9-pen)
                ci=[max(0,rate-1.959963985*se),min(1,rate+1.959963985*se)]
                if name=='skin':
                    with np.load(count_dir/'skin_independent.npz') as ind:
                        assert list(ind['cells'])==list(z['cells'])
                        h,loss,gain,covered=ind['counts'][mode,j].tolist()
                    rate=covered/h;pen=(loss+gain)/h;rho=h/400000;lower=max(0,.9-pen)
                    ci=[float(beta.ppf(.025,covered,h-covered+1)) if covered else 0.,
                        float(beta.ppf(.975,covered+1,h-covered)) if covered<h else 1.]
                    # Ratio of sample means for independent per-query observations.
                    ps=np.sqrt(((loss+gain)*(1-2*pen)+2*pen*gain+h*pen**2)/(h*h)*400000/399999)
                entry[key]=dict(selections=h,practical_only=loss,full_only=gain,covered=covered,rho=rho,penalty=pen,lower_bound=lower,coverage=rate,coverage_interval=ci,penalty_se=ps,selecting_samples=(h if name=="skin" else int(sum(d[:,0]>0))))
                rows[mode].append([name,label,f'{100*rho:.2f}',f'{100*pen:.2f}',f'{100*lower:.2f}',f'{100*rate:.2f}',f'{100*ci[0]:.2f}–{100*ci[1]:.2f}'])
    md=['# Selected-leaf coverage on the real datasets','',design,'',mc.replace(r'\(n+1\)','n+1'),'','Selection frequency, observed coverage and lower bounds are percentages. Penalties are percentage points. Each lower bound is estimated as max(0, 90% minus the penalty).']
    columns=['Dataset','Cell','Selected (%)','Penalty (pp)','Lower bound (%)','Coverage (%)','95% interval (%)']
    for mode,title in enumerate(['Hallucination tree','Ordinary tree']):
        md+=['','## '+title,'',' | '.join(columns),' | '.join(['---']*len(columns))]+[' | '.join(r) for r in rows[mode]]
        tex+=[r'\clearpage',r'\subsection*{Selected-leaf coverage on the real datasets: '+title.lower()+'}']
        if mode==0:tex += [design,'',mc.replace('%',r'\%'),' ','The estimated lower bound is 90\\% minus the estimated penalty in percentage points, truncated at zero.','']
        else:tex+=['The same cells and draws are used for the ordinary tree.','']
        tex += [r'\begingroup\small\setlength{\tabcolsep}{4pt}',r'\begin{center}\begin{tabular}{llrrrrr}\toprule',r'Dataset & Cell & Selected (\%) & Penalty (pp) & Lower bound (\%) & Coverage (\%) & 95\% interval (\%) \\ \midrule']
        for i,r in enumerate(rows[mode]):
            if i and i%4==0:tex += [r'\addlinespace']
            rr=r.copy()
            if i%4:rr[0]=''
            tex+=[' & '.join(rr).replace('–','--')+r' \\']
        tex += [r'\bottomrule\end{tabular}\end{center}\endgroup']
    md+=['','## Leaf regions','','Conditions within a row all apply. Cutpoints are rounded to six significant digits for display. Exact cutpoints and counts are in results.json.','','Dataset | Cell | Conditions','--- | --- | ---']+[' | '.join(row) for row in keys]
    tex += [r'\clearpage\subsection*{Leaf regions for the real datasets}','All conditions in a row must hold. Cutpoints are rounded to six significant digits.',r'\begingroup\small\begin{longtable}{llp{0.75\linewidth}}\toprule Dataset & Cell & Conditions \\ \midrule\endhead']
    for r in keys:tex+=[' & '.join(map(esc,r))+r' \\']
    tex += [r'\bottomrule\end{longtable}\endgroup',r'\clearpage']
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUTPUT/'REPORT.md').write_text('\n'.join(md)+'\n')
    (OUTPUT/'tables.tex').write_text('\n'.join(tex)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rerun',action='store_true',help='read regenerated rather than archived counts')
    build(OUTPUT if p.parse_args().rerun else HERE)

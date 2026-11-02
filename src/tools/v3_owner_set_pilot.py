"""Small CPU-only expected-F0.5 set-selection pilot; no test submission writes.

Decision-theoretic hypothesis, NOT a claim of calibrated independent labels.
It deliberately evaluates BEFORE owner assignment because auxiliary rivals lack
complete owner lists. Never deploy based on this pilot alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import joblib
import numpy as np

from v3_neural_independent_audit import SCRATCH,lg,summary
from v3_transfer_data import ROOT,PILOT,bootstrap,digest,write


def expected_f05(probabilities):
    """Exact independent-Bernoulli expected score for each sorted prefix.

    Empty/empty is 1. The nonempty score is 5*TP/(true_count+4*k).
    Caller supplies probabilities in descending order.
    """
    p=np.asarray(probabilities,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
    n=len(p);prefix=[np.ones(1)];suffix=[None]*(n+1);suffix[n]=np.ones(1)
    for v in p:prefix.append(np.convolve(prefix[-1],[1-v,v]))
    for i in range(n-1,-1,-1):suffix[i]=np.convolve(suffix[i+1],[1-p[i],p[i]])
    result=np.zeros(n+1);result[0]=prefix[n][0]
    for k in range(1,n+1):
        x=np.arange(k+1)[:,None];y=np.arange(n-k+1)[None,:]
        result[k]=np.sum((prefix[k][:,None]*suffix[k][None,:])*(5*x/(x+y+4*k)))
    return result


def run(audit,output):
    audit,output=Path(audit),Path(output);output.mkdir(parents=True,exist_ok=False)
    report=json.loads((audit/'report.json').read_text());selection=report['selected'];tag=report['tag']
    model=joblib.load(audit/(selection['name']+'.joblib'))
    data={}
    ids=[json.loads(x)['id'] for x in (PILOT/'data/records.jsonl').open()]
    for split in ('tune','dev'):
        rows=pickle.load((SCRATCH/f'{split}.pkl').open('rb'))
        pce=np.load(SCRATCH/tag/f'{split}_pce.npy');ops=np.load(SCRATCH/f'{split}.pkl.ops.npy')
        p=model.predict_proba(np.column_stack((ops,lg([r[4] for r in rows]),lg(pce))).astype(np.float32))[:,1]
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','left','right')}
        pair={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        ix=np.asarray([pair[(r[0],r[1])] for r in rows]);q=a['query_ids'][ix]
        groups=[np.flatnonzero(q==i) for i in range(len(a['reference_ids']))]
        groups=[g[np.argsort(-p[g],kind='stable')] for g in groups]
        data[split]=(a,ix,p,groups)
    def decision(split,rule):
        a,ix,p,groups=data[split];keep=np.zeros(len(p),dtype=bool)
        if rule is None:keep=p>=selection['threshold']
        else:
            pp=1/(1+np.exp(-(lg(p)*rule['scale']+rule['bias'])))
            for group in groups:
                if not len(group):continue
                utility=expected_f05(pp[group]);k=int(np.argmax(utility))
                keep[group[:k]]=True
        full=np.zeros(len(a['y']),dtype=bool);full[ix]=keep
        return summary(a,full)
    refs=data['tune'][0]['reference_ids']
    cal=np.asarray([int(hashlib.sha256(str(o).encode()).hexdigest()[:8],16)%3==0 for o in refs])
    baseline,_=decision('tune',None)
    grid=[dict(rule=None,calibration_macro_f05=float(baseline[cal].mean()))]
    for scale in (.7,1.,1.3):
        for bias in (-1.,-.5,0.,.5,1.):
            rule=dict(scale=scale,bias=bias);s,_=decision('tune',rule)
            grid.append(dict(rule=rule,calibration_macro_f05=float(s[cal].mean())))
    selected=max(grid,key=lambda r:r['calibration_macro_f05'])
    write(output/'selection_before_dev.json',dict(selected=selected,grid=grid))
    old,old_metrics=decision('dev',None);new,new_metrics=decision('dev',selected['rule'])
    result=dict(complete=True,audit_sha256=digest(audit/'report.json'),selected=selected,
        before_owner_baseline=old_metrics,before_owner_new=new_metrics,comparison=bootstrap(old,new),
        safe_to_deploy=False,
        caveats=['Independent Bernoulli probabilities are an unverified modeling assumption.',
            'Selection used only the existing held-out calibration part of tune.',
            'Metrics include every dev owner and retrieval miss; utility model does not represent unseen candidates.',
            'No owner assignment was applied: real auxiliary owner lists must be completed before deployment.',
            'Previously explored development; no French labels.'])
    write(output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))

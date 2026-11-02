"""Bounded V12 corruption-channel correction pilot; tune selection, dev report.

Independent project outputs only. Does not alter any submission or peer cache.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from v3_generator_channel import Channel, Likelihoods, make_dictionary
from v3_owner_assignment import winner
from v3_transfer_data import ROOT, write, digest, bootstrap
from v3_v10_combined_audit import model, logit

C=Path('work/ce')
P=ROOT/'artifacts/improvement_v3/v12_top10_rescue_v1'
OUT=ROOT/'artifacts/improvement_v3/v12_generator_correction_v1'


def run():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(parents=True,exist_ok=False); start=time.monotonic(); hashes={}
    def path(p):
        p=Path(p);h=digest(p);assert hashes.get(str(p),h)==h
        hashes[str(p)]=h;return p
    def load(p):
        with path(p).open('rb') as f:return pickle.load(f)
    def array(p):return np.load(path(p),allow_pickle=False)
    def progress(stage,**kw):
        x=dict(stage=stage,seconds=time.monotonic()-start,**kw)
        write(OUT/'progress.json',x);print(json.dumps(x),flush=True)
    ctx=load(P/'context.pkl');saved=array(P/'fold_scores.npz')['baseline']
    baseline_mask=array(P/'decisions.npz')['baseline']
    allrows=ctx['allrows'];focal=ctx['rows'];ix=ctx['ix'];gold=ctx['gold']
    refs=set(gold);held_targets=set().union(*gold.values())
    dictionary=make_dictionary(load(C/'translit_cnt.pkl'))
    channel=Channel(dictionary); likelihoods=Likelihoods();training_owners=set()
    rng=np.random.default_rng(927);seen=0
    for file in sorted(C.glob('train_*.pkl')):
        part=load(file)
        for j in rng.choice(len(part),min(20000,len(part)),replace=False):
            r=part[j]
            if r[0] in refs or r[1] in held_targets:continue
            _,events=channel.extract(r);likelihoods.add(events,r[2]);training_owners.add(r[0]);seen+=1
        progress('profile_training',file=file.name,rows=seen,owners=len(training_owners))
        del part
    assert not training_owners & refs
    with (OUT/'channel.pkl').open('wb') as f:pickle.dump(dict(dictionary=dictionary,likelihoods=likelihoods),f)
    scalars=[];likelihood_features=[]
    for i,r in enumerate(allrows):
        v,e=channel.extract(r);scalars.append(v);likelihood_features.append(likelihoods.transform(e))
        if i and i%25000==0:progress('features',rows=i,total=len(allrows))
    scalars=np.asarray(scalars,np.float32);lf=np.asarray(likelihood_features,np.float32)
    assert scalars.shape[1]==44 and lf.shape[1]==21,(scalars.shape,lf.shape)
    np.save(OUT/'scalar_features.npy',scalars);np.save(OUT/'likelihood_features.npy',lf)

    # Reconstitute existing matcher inputs, using C8 for original candidates and
    # the small encoder on recovery routes exactly as in V12. No final-fit base
    # stack probabilities are injected as supervised correction-model features.
    base={};train_rows=[]
    def matrix(rr,p,ce):
        x=np.column_stack((array(str(p)+'.ops.npy'),logit([r[4] for r in rr]),
                           logit(ce),array(str(p)+'.nf.npy'),array(str(p)+'.nf2.npy'))).astype(np.float32)
        assert x.shape==(len(rr),33)
        return x
    for sp in ('dev','tune'):
        rr=load(C/f'{sp}_kc.pkl'); xx=matrix(rr,C/f'{sp}_kc.pkl',array(C/f'e5_cont1300k/{sp}_kc_pce.npy'))
        train_rows.extend(rr);base.update((r[:2],x) for r,x in zip(rr,xx))
        rr=load(C/f'{sp}.pkl');xx=matrix(rr,C/f'{sp}.pkl',array(C/f'c8/{sp}_pce.npy'))
        base.update((r[:2],x) for r,x in zip(rr,xx))
    rr=load(C/'rivals.pkl');xx=matrix(rr,C/'rivals.pkl',array(C/'rivals.pkl.c8.npy'))
    for r,x in zip(rr,xx):base.setdefault(r[:2],x)
    eligible=np.r_[np.ones(ctx['nbase'],bool),~ctx['rescue']]
    ai=np.flatnonzero(eligible); all_to_local={i:j for j,i in enumerate(ai)}
    x0=np.stack([base[allrows[i][:2]] for i in ai]);del base,xx,rr
    ax={r[:2]:i for i,r in enumerate(allrows)}
    ti=np.asarray([all_to_local[ax[r[:2]]] for r in train_rows])
    of={r[0]:int(k) for r,k in zip(focal,ctx['fold'])}
    trainfold=np.asarray([of[r[0]] for r in train_rows]);y=np.asarray([r[2] for r in train_rows])
    features={'ops':np.column_stack((x0,scalars[ai])),
              'channel':np.column_stack((x0,scalars[ai],lf[ai]))}

    def decide(p,threshold):
        keep=np.r_[(p[:ctx['nbase']]>=threshold)&ctx['gate'],
                   (p[ctx['nbase']:]>=.9)&~ctx['rescue']]
        for g in ctx['groups']:
            accept=g[keep[g]]
            if len(accept)>1:
                w=winner(p[accept],ctx['text'][accept],**ctx['rule']);keep[accept]=False
                if w is not None:keep[accept[w]]=True
        return keep[ix]
    arms={('baseline',0,.75):baseline_mask.copy()};predictions={}
    for name in features:
        for weight in (.25,.5,1.):
            for threshold in (.7,.75,.8):arms[name,weight,threshold]=np.zeros(len(focal),bool)
        predictions[name]=[]
    reconstructed=np.zeros(len(focal),bool)
    for k in range(5):
        mask=ctx['fold']==k;bp=saved[k];reconstructed[mask]=decide(bp,.75)[mask]
        for name,x in features.items():
            progress('fit',fold=k,head=name,training_pairs=int((trainfold!=k).sum()))
            fit=trainfold!=k
            m=model().fit(x[ti[fit]],y[fit]);q=m.predict_proba(x)[:,1]
            predictions[name].append(q.astype(np.float32))
            for weight in (.25,.5,1.):
                lp=(1-weight)*logit(bp[ai])+weight*logit(q)
                pp=bp.copy();pp[ai]=1/(1+np.exp(-lp))
                # Include confident decisions in the experiment; unchanged rescue
                # rows outside V12 cannot enter through this correction arm.
                for threshold in (.7,.75,.8):arms[name,weight,threshold][mask]=decide(pp,threshold)[mask]
    assert np.array_equal(reconstructed,baseline_mask),'Baseline ownership replay drift'
    def metric(keep,split):
        pred=defaultdict(set)
        for r,k in zip(focal,keep):
            if k:pred[r[0]].add(r[1])
        a=ctx['arrays'][split];owners=list(map(str,a['reference_ids']));tt=a['true_counts']
        tp=np.asarray([len(pred[o]&gold[o]) for o in owners]);nn=np.asarray([len(pred[o]) for o in owners]);den=tt+4*nn
        s=np.divide(5*tp,den,out=np.ones(len(owners)),where=den!=0)
        return s,dict(macro_f05=float(s.mean()),tp=int(tp.sum()),fp=int((nn-tp).sum()),fn=int((tt-tp).sum()),
            singleton_false_merges=int(((tt==0)&(nn>0)).sum()),
            country={str(c):float(s[a['countries']==c].mean()) for c in np.unique(a['countries'])})
    tune={str(k):metric(v,'tune')[1] for k,v in arms.items()}
    selected=max(arms,key=lambda k:(tune[str(k)]['macro_f05'],-k[1]))
    result={}
    for sp in ctx['arrays']:
        bs,bm=metric(baseline_mask,sp);ss,sm=metric(arms[selected],sp);trans=Counter();owners=set(map(str,ctx['arrays'][sp]['reference_ids']))
        for i,r in enumerate(focal):
            if r[0] not in owners:continue
            if arms[selected][i] and not baseline_mask[i]:trans['added_true' if r[2] else 'added_false']+=1
            if baseline_mask[i] and not arms[selected][i]:trans['removed_true' if r[2] else 'removed_false']+=1
        result[sp]=dict(baseline=bm,selected=sm,delta=bootstrap(bs,ss),transitions=dict(trans))
    for p,h in hashes.items():assert digest(p)==h,'input changed: '+p
    np.savez_compressed(OUT/'fold_correction_scores.npz',**{k:np.asarray(v) for k,v in predictions.items()},indices=ai)
    np.savez_compressed(OUT/'decisions.npz',baseline=baseline_mask,selected=arms[selected])
    report=dict(complete=True,selected=selected,tune_grid=tune,splits=result,profile=dict(rows=seen,owners=len(training_owners),
        totals=likelihoods.totals,keys=len(likelihoods.counts),dictionary_keys=len(dictionary)),seconds=time.monotonic()-start,
        inputs_sha256=hashes,caveats=['Exploratory reused dev/tune, no French labels.',
        'Signature likelihoods learned only from separate training pairs; calibration/reporting owners and their true targets excluded.',
        'Existing train-only dictionary reused; not a newly built pristine lexicon.',
        'Same shared owner folds and bounded reverse rivals as audited V12.',
        'This tests a tabular correction, not the unrun GPU architecture or full owner/NONE selector.'])
    write(OUT/'report.json',report);progress('complete',selected=selected,splits=result)


if __name__=='__main__':run()

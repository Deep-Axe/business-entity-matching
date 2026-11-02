"""Frozen additive recovery evaluated against the exact submitted OOF stacker.

Reuses predeclared recovery thresholds and bounded reverse-rival retrieval.
No threshold/model selection, fabricated p100 probabilities, test scoring, or
changes to the public-best output. Refit five owner folds only to score rivals
with the same out-of-fold model as the focal owner.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import importlib.util
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from ber.store import connect
from v3_neural_independent_audit import summary
from v3_neural_test_pipeline import SCRATCH, STRUCT, resources, read, feature_functions, edit_matrix
from v3_owner_assignment import load_text, text_logit, winner
from v3_public0976_error_audit import features, model
from v3_recovery_owner_audit import pick_additions
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

WORK=ROOT/'artifacts/improvement_v3/public0976_recovery_v1'
BASE=ROOT/'artifacts/improvement_v3/public0976_error_audit_v2'
SOURCE=ROOT/'artifacts/improvement_v3/recovery_rivals_v1'
WIDER=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1'


def run():
    WORK.mkdir(parents=True,exist_ok=False);resources();start=time.monotonic();bound={}
    def track(path):
        bound[str(path)]=digest(path);return Path(path)
    def raw(path):return pickle.load(track(path).open('rb'))
    def arr(path):return np.load(track(path),allow_pickle=False)
    def progress(stage,**kw):write(WORK/'progress.json',dict(stage=stage,seconds=time.monotonic()-start,**kw))
    source=read(track(SOURCE/'completion.json'))
    if not source['complete']:raise ValueError('Incomplete bounded recovery rivals')
    for name,h in source['output_sha256'].items():
        if digest(SOURCE/name)!=h:raise ValueError('Frozen recovery proposal source changed')
    proposals=raw(SOURCE/'proposals.pkl');targets={p['row'][1] for p in proposals}
    prior=read(track(BASE/'report.json'))
    # Enforce the exact scored arrays, excluding unused mutable peer scripts/OOF.
    ignored={'src/neural/stack_eval.py',str(SCRATCH/'stack_oof_e5_cont1300k.pkl')}
    for path,h in prior['inputs_sha256'].items():
        if path not in ignored and digest(path)!=h:raise ValueError('Submitted audit dependency changed: '+path)
    decisions=arr(BASE/'decisions.npz');old_keep=decisions['coherent'];old_oof=decisions['oof']
    ids=[__import__('json').loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    rows=[];matrices=[];data={};full_keys=set()
    for split in ('dev','tune'):
        rr=raw(SCRATCH/(split+'.pkl'));lo=len(rows);rows.extend(rr)
        ce=arr(SCRATCH/'e5_cont1300k'/(split+'_pce.npy'))
        freq=np.column_stack((arr(SCRATCH/(split+'.pkl.nf.npy')),arr(SCRATCH/(split+'.pkl.nf2.npy'))))
        matrices.append(features(rr,arr(SCRATCH/(split+'.pkl.ops.npy')),ce,freq))
        with np.load(track(PILOT/'data'/(split+'.npz')),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','baseline_100000','left','right')}
        keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        ix=np.asarray([keys[r[:2]] for r in rr]);full_keys.update(keys)
        data[split]=(a,ix,slice(lo,len(rows)))
    if any(p['row'][:2] in full_keys for p in proposals):raise ValueError('Proposal not actually new')
    M=np.vstack(matrices);Y=np.asarray([r[2] for r in rows]);owners=np.asarray([r[0] for r in rows])
    if old_keep.shape!=(len(rows),):raise ValueError('Submitted decision row count differs')
    # Only candidate targets addressed by recovery are needed for no-stealing.
    selected=[i for i,r in enumerate(rows) if r[1] in targets]
    rivals=[rows[i] for i in selected];rm=[M[i] for i in selected];seen={r[:2] for r in rows}
    rr=raw(SCRATCH/'rivals.pkl')
    matrix=features(rr,arr(SCRATCH/'rivals.pkl.ops.npy'),arr(SCRATCH/'rivals.pkl.e5_cont1300k.npy'),
        np.column_stack((arr(SCRATCH/'rivals.pkl.nf.npy'),arr(SCRATCH/'rivals.pkl.nf2.npy'))))
    for i,r in enumerate(rr):
        if r[:2] in seen or r[:2] in full_keys:continue
        seen.add(r[:2])
        if r[1] in targets and r[4]>=.01:rivals.append(r);rm.append(matrix[i])
    new=[];new_ce=[]
    for root,cepath in [(WIDER,WIDER/'rivals_pce.npy'),
            (SOURCE,ROOT/'artifacts/improvement_v3/recovery_owner_audit_v1/new_rival_pce.npy')]:
        rr=raw(root/'rivals.pkl');ce=arr(cepath)
        if len(rr)!=len(ce):raise ValueError('Extra rival CE alignment differs')
        for r,p in zip(rr,ce):
            if r[:2] in seen or r[:2] in full_keys:continue
            seen.add(r[:2])
            if r[1] in targets and r[4]>=.01:new.append(r);new_ce.append(p)
    if new:
        core_path=track('src/neural/name_unique_lib.py')
        spec=importlib.util.spec_from_file_location('recovery_count_core',core_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);core=lru_cache(maxsize=50000)(module.core)
        left=[(r[3],core(r[5])) for r in new];right=[(r[3],core(r[7])) for r in new];wanted=set(left)|set(right)
        db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192');counts={}
        for table,expected in [('anchors',2206821),('records',10320219)]:
            c=Counter();n=0
            for n,r in enumerate(db.execute('SELECT country,name FROM '+table),1):
                key=(r['country'],core(r['name']))
                if key in wanted:c[key]+=1
                if n%500000==0:resources();progress('counting_text_for_new_rivals',table=table,rows=n)
            if n!=expected:raise ValueError('Full provided-text count coverage differs')
            counts[table]=c
        db.close()
        freq=np.asarray([[np.log1p(counts['anchors'][l]),np.log1p(counts['anchors'][r]),np.log1p(counts['records'][r])]
                         for l,r in zip(left,right)],np.float32)
        rm.extend(features(new,edit_matrix(new,feature_functions()),new_ce,freq));rivals.extend(new)
    RM=np.asarray(rm,np.float32);rule=read(track(STRUCT/'owner_eval/policy.json'))['selected']
    tm=load_text(track(STRUCT/'text_scorer.json'));text=np.asarray([text_logit({'name':r[5],'address':r[6]},
                 {'name':r[7],'address':r[8]},tm) for r in rivals])
    groups=defaultdict(list)
    for i,r in enumerate(rivals):groups[r[1]].append(i)
    additions={route:[] for route in ('native','website','deep','union')}
    for fold,(fit,held) in enumerate(GroupKFold(5).split(M,Y,owners)):
        resources();progress('crossfit_recovery_owner_checks',fold=fold,new_routed_rivals=len(new))
        m=model().fit(M[fit],Y[fit]);p=m.predict_proba(M[held])[:,1]
        if not np.allclose(p,old_oof[held],rtol=0,atol=1e-12):raise ValueError('Exact submitted OOF recipe failed replay')
        rp=m.predict_proba(RM)[:,1] if len(rivals) else np.empty(0);claimed=set()
        for target,ix in groups.items():
            ix=np.asarray(sorted((i for i in ix if rp[i]>=.75 and rivals[i][4]>=.01),key=lambda i:rivals[i][0]))
            if len(ix)==1 or (len(ix)>1 and winner(rp[ix],text[ix],**rule) is not None):claimed.add(target)
        held_owners=set(owners[held])
        for route in additions:
            chosen=pick_additions(proposals,route,claimed,tm,rule)
            additions[route].extend(p for p in chosen if p['row'][0] in held_owners)
    result={}
    for route,added in additions.items():
        result[route]={}
        for split,(a,ix,sl) in data.items():
            keep=np.zeros(len(a['y']),bool);keep[ix]=old_keep[sl];old,metrics=summary(a,keep)
            count=np.bincount(a['query_ids'][keep],minlength=len(a['reference_ids']))
            tp=np.bincount(a['query_ids'][keep&(a['y']==1)],minlength=len(a['reference_ids']))
            qi={str(o):i for i,o in enumerate(a['reference_ids'])};extra=[p for p in added if p['split']==split]
            for item in extra:
                r=item['row'];count[qi[r[0]]]+=1;tp[qi[r[0]]]+=int(r[2])
            den=a['true_counts']+4*count;score=np.divide(5*tp,den,out=np.ones(len(den)),where=den!=0)
            result[route][split]=dict(baseline=metrics,new=dict(macro_f05=float(score.mean()),tp=int(tp.sum()),
                fp=int(count.sum()-tp.sum()),fn=int(a['true_counts'].sum()-tp.sum()),
                countries={str(c):float(score[a['countries']==c].mean()) for c in np.unique(a['countries'])}),
                comparison=bootstrap(old,score),added_tp=sum(int(p['row'][2]) for p in extra),added_fp=sum(1-int(p['row'][2]) for p in extra))
    for path,h in bound.items():
        if digest(path)!=h:raise ValueError('Frozen recovery audit input changed: '+path)
    write(WORK/'report.json',dict(complete=True,variants=result,inputs_sha256=bound,new_routed_rivals=len(new),
        thresholds_retuned=False,ready_for_submission=False,seconds=time.monotonic()-start,
        notes=['Exact submitted baseline OOF replay; five coherent owner models for bounded existing claims.',
               'Recovery cutoffs predeclared on tune; reused development, no French labels or public guarantee.',
               'Additive no-stealing only. Reverse rival coverage is bounded, not exhaustive.',
               'Original baseline OOF autocast and retained recovery FP16 scores; deployment still needs matched-precision calibration.']))
    progress('complete')


if __name__=='__main__':run()

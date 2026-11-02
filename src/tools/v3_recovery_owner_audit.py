"""Frozen full-owner assessment of conservative candidate recovery, no deployment.

Replay the selected structural submission exactly, widen existing-owner evidence,
and add proposals only to targets with no accepted existing owner in that cache.
Route thresholds are already tune-frozen. Do not choose routes from dev results.
"""
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import joblib
import numpy as np

from ber.store import connect
from v3_frozen_widened_audit import decisions,validate_same_edge
from v3_neural_independent_audit import lg,summary
from v3_neural_name_counts import Counts,matrix as structural_matrix
from v3_neural_test_pipeline import SCRATCH,STRUCT,resources,gpu_processes,feature_functions,edit_matrix
from v3_owner_assignment import load_text,text_logit,winner
from v3_transfer_data import ROOT,PILOT,digest,write,bootstrap

WORK=ROOT/'artifacts/improvement_v3/recovery_owner_audit_v1'
SOURCE=ROOT/'artifacts/improvement_v3/recovery_rivals_v1'
PEER=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
AUDIT=ROOT/'artifacts/improvement_v3/neural_cpu_upgrade_v1/same_training_half_extra'
WIDER=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1'
COUNTS=ROOT/'artifacts/improvement_v3/train_s1_structural_counts_v1'


def read(p):return json.loads(Path(p).read_text())


def pick_additions(proposals,route,claimed,text_model,rule):
    groups=defaultdict(list)
    for item in proposals:
        row=item['row'];eligible=[p for p in item['provenance'] if route=='union' or p['route']==route]
        if eligible and row[1] not in claimed:
            groups[row[1]].append((item,max(p['ce'] for p in eligible)))
    result=[]
    for group in groups.values():
        group=sorted(group,key=lambda v:v[0]['row'][0])
        if len(group)==1:result.append(group[0][0]);continue
        p=[v for _,v in group]
        text=[text_logit({'name':r['row'][5],'address':r['row'][6]},
                         {'name':r['row'][7],'address':r['row'][8]},text_model) for r,_ in group]
        ix=winner(p,text,**rule)
        if ix is not None:result.append(group[ix][0])
    return result


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (WORK/'report.json').exists():return
    end=time.monotonic()+4*3600
    def progress(stage,**kw):write(WORK/'progress.json',dict(stage=stage,ready_for_submission=False,**kw))
    while not all((p/'completion.json').exists() for p in (SOURCE,COUNTS)):
        if time.monotonic()>=end:raise TimeoutError('Finite recovery audit wait expired')
        progress('waiting_for_rivals_and_counts');time.sleep(20)
    resources();done=read(SOURCE/'completion.json');report=read(AUDIT/'report.json')
    for name,h in done['output_sha256'].items():
        if digest(SOURCE/name)!=h:raise ValueError('Recovery source changed')
    for name,h in report['inputs_sha256'].items():
        if digest(name)!=h:raise ValueError('Frozen structural inputs changed')
    cd=read(COUNTS/'completion.json')
    if not cd['complete'] or cd['references']!=2206821 or digest(COUNTS/'counts.sqlite3')!=cd['database_sha256']:
        raise ValueError('Incomplete full training-S1 counts')
    raw=SOURCE/'rivals.pkl';cp=WORK/'new_rival_pce.npy';weights=PEER/'reproduction/model.pt'
    identity=dict(source_sha256=digest(SOURCE/'completion.json'),audit_sha256=digest(AUDIT/'report.json'),
        counts_sha256=digest(COUNTS/'completion.json'),checkpoint_sha256=digest(weights),code_sha256=digest(__file__))
    if (WORK/'identity.json').exists():
        if read(WORK/'identity.json')!=identity:raise ValueError('Recovery audit identity changed')
    else:write(WORK/'identity.json',identity)
    if not cp.exists():
        lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
        while True:
            if time.monotonic()>=end:raise TimeoutError('Finite recovery GPU wait expired')
            if not gpu_processes():
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            progress('waiting_for_free_gpu_for_new_rivals');time.sleep(20)
        env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
        try:
            with (WORK/'gpu.log').open('a') as log:
                subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),'tools/v3_ce_half_score.py',
                    '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5','--weights',str(weights),
                    '--input',str(raw),'--output',str(cp)],cwd=ROOT,env=env,stdout=log,stderr=log,check=True,timeout=600)
            write(WORK/'scores.json',dict(identity_sha256=digest(WORK/'identity.json'),scores_sha256=digest(cp)))
        finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    if read(WORK/'scores.json')['scores_sha256']!=digest(cp):raise ValueError('Recovery rival scores changed')
    progress('frozen_owner_replay')
    union=read(WIDER/'union.json')
    for name,h in union['outputs_sha256'].items():
        if digest(WIDER/name)!=h:raise ValueError('Old widened evidence changed')
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    rows=[];ce=[];ops=[];data={};full_keys=set()
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','baseline_100000','left','right')}
        rr=pickle.load((SCRATCH/f'{split}.pkl').open('rb'))
        keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        ix=np.asarray([keys[r[:2]] for r in rr]);expected=np.flatnonzero(a['baseline_100000']>=.01)
        if not np.array_equal(np.sort(ix),expected) or len(set(ix))!=len(ix):raise ValueError('Focal route changed')
        start=len(rows);rows.extend(rr);data[split]=(a,ix,slice(start,len(rows)))
        full_keys.update(keys);ce.extend(np.load(PEER/'half_calibration'/f'{split}_pce.npy',allow_pickle=False))
        ops.extend(np.load(SCRATCH/f'{split}.pkl.ops.npy',allow_pickle=False))
    focal=len(rows);seen={r[:2]:i for i,r in enumerate(rows)}
    old_keys={r[:2] for r in pickle.load((SCRATCH/'rivals.pkl').open('rb'))}
    recovered=pickle.load(raw.open('rb'));recovered_ce=np.load(cp,allow_pickle=False)
    sets=[(pickle.load((WIDER/'rivals.pkl').open('rb')),np.load(WIDER/'rivals_pce.npy'),np.load(WIDER/'rivals_ops.npy')),
          (recovered,recovered_ce,edit_matrix(recovered,feature_functions()))]
    for rr,pp,xx in sets:
        if len(rr)!=len(pp) or xx.shape!=(len(rr),28):raise ValueError('Rival arrays misaligned')
        for r,p,x in zip(rr,pp,xx):
            if r[:2] in seen:validate_same_edge(rows[seen[r[:2]]],r);continue
            if r[:2] in full_keys:continue
            seen[r[:2]]=len(rows);rows.append(r);ce.append(p);ops.append(x)
    counts=Counts(COUNTS/'counts.sqlite3');fragments=read(STRUCT/'feature_pilot_v2/fragments.json')['fragments']
    extra=structural_matrix(rows,counts,fragments);counts.close()
    old_extra=ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1'
    if not np.array_equal(extra[:focal],np.vstack([np.load(old_extra/f'{s}.npy') for s in ('tune','dev')])):
        raise ValueError('Complete S1 counts do not reproduce original focal structural features')
    base=np.asarray([r[4] for r in rows]);ce=np.asarray(ce)
    if not np.isfinite(ce).all() or ((ce<0)|(ce>1)).any():raise ValueError('Invalid CE probabilities')
    matrix=np.column_stack((np.asarray(ops),lg(base),lg(ce),extra)).astype(np.float32)
    selected=report['selected'];model=joblib.load(AUDIT/(selected['name']+'.joblib'))
    p=model.predict_proba(matrix)[:,1];text_model=load_text(STRUCT/'text_scorer.json')
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},text_model) for r in rows])
    rule=read(STRUCT/'owner_eval/policy.json')['selected'];groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(g,key=lambda i:rows[i][0])) for g in groups.values() if len(g)>1]
    old_mask=np.asarray([i<focal or r[:2] in old_keys for i,r in enumerate(rows)])
    original=decisions(p,base,text,groups,selected['threshold'],rule,old_mask)
    existing=decisions(p,base,text,groups,selected['threshold'],rule,np.ones(len(rows),bool))
    def state(split,keep):
        a,ix,sl=data[split];mask=np.zeros(len(a['y']),bool);mask[ix]=keep[sl]
        scores,metrics=summary(a,mask)
        n=len(a['reference_ids']);count=np.bincount(a['query_ids'][mask],minlength=n)
        tp=np.bincount(a['query_ids'][mask&(a['y']==1)],minlength=n)
        return scores,metrics,count,tp
    replay=state('dev',original)[1]
    if abs(replay['macro_f05']-report['models'][selected['name']]['development']['macro_f05'])>1e-12:
        raise ValueError('Frozen structural development failed exact replay')
    proposals=pickle.load((SOURCE/'proposals.pkl').open('rb'))
    if any(item['row'][:2] in full_keys for item in proposals):raise ValueError('Recovery proposal was already a focal candidate')
    # Verify labels only for reporting after all retrieval, CE and existing
    # predictions are fixed; these labels never enter the added-pair selector.
    db=connect(ROOT/'artifacts/train_store_v2')
    for item in proposals:
        r=item['row'];y=int(db.execute('SELECT 1 FROM labels WHERE reference_id=? AND target_id=?',(r[0],r[1])).fetchone() is not None)
        if y!=r[2]:raise ValueError('Recovery reporting label differs from provided truth')
    db.close()
    claimed={rows[i][1] for i in np.flatnonzero(existing)};variants={}
    for route in ('native','website','deep','union'):
        added=pick_additions(proposals,route,claimed,text_model,rule);result={}
        for split in ('tune','dev'):
            old,old_metrics,count,tp=state(split,existing);a=data[split][0]
            qi={str(o):i for i,o in enumerate(a['reference_ids'])};accepted=[x for x in added if x['split']==split]
            for item in accepted:
                r=item['row'];j=qi[r[0]];count[j]+=1;tp[j]+=int(r[2])
            den=a['true_counts']+4*count;scores=np.divide(5*tp,den,out=np.ones(len(den)),where=den!=0)
            metrics=dict(macro_f05=float(scores.mean()),tp=int(tp.sum()),fp=int(count.sum()-tp.sum()),fn=int(a['true_counts'].sum()-tp.sum()),
                countries={str(c):float(scores[a['countries']==c].mean()) for c in np.unique(a['countries'])})
            result[split]=dict(baseline=old_metrics,new=metrics,comparison=bootstrap(old,scores),
                added_tp=sum(int(x['row'][2]) for x in accepted),added_fp=sum(1-int(x['row'][2]) for x in accepted))
        variants[route]=result
    write(WORK/'report.json',dict(complete=True,original_replay=replay,variants=variants,
        selected_existing=selected,identity_sha256=digest(WORK/'identity.json'),rival_pairs=len(rows)-focal,
        full_owner_metrics=True,exhaustive_rival_coverage=False,fitted_or_retuned=False,ready_for_submission=False,
        note='Frozen tune-only recovery cutoffs, no stealing accepted targets; bounded wider reverse rivals, not all S1 owners. Routes reported, not selected by dev. Reused US/India data; no French/public guarantee.'))
    progress('complete')


if __name__=='__main__':run()

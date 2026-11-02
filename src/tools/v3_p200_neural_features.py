"""Test the 200k CPU score alongside the frozen neural matcher.

Wait for the existing 200k owner evaluation, reuse its genuine forward scores,
and recompute only missing owner profiles. No fake context, GPU work, modified
release, or submission writes. Comparisons use tune selection, not dev fitting.
"""
from collections import defaultdict
from pathlib import Path
import pickle
import re
import sqlite3
import subprocess
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.deadline_retrieval import retrievers
from ber.direct_features import make_arrays
from ber.disk_index import TargetStore
from ber.store import connect, lookup, record_row
from v2_cpu_features import Weighting
from v2_run_pipeline import enhanced_matrix, NAMES, code_hashes
from v3_neural_test_pipeline import SCRATCH, read, resources, lg
from v3_transfer_data import ROOT, PILOT, digest, write


def features(work):
    root=work/'features'; root.mkdir(exist_ok=True)
    if (root/'manifest.json').exists():
        return
    owner=ROOT/'artifacts/improvement_v3/owner200k_v1/evaluation'
    models={n:ROOT/f'artifacts/improvement_v2/cpu/fit_gate0_{n}k_v1/enhanced.cbm' for n in (100,200)}
    identity=dict(code_sha256=digest(__file__),model_sha256={str(n):digest(p) for n,p in models.items()},
        pipeline_code=code_hashes(),rivals_sha256=digest(SCRATCH/'rivals.pkl'),
        owner_policy_sha256=digest(owner/'policy.json'),route='Unchanged p100k>=.01')
    if (work/'identity.json').exists():
        if read(work/'identity.json')!=identity:
            raise ValueError('200k-neural feature identity changed')
    else:
        write(work/'identity.json',identity)
    db=sqlite3.connect((owner/'rivals.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-8192')
    import json
    metadata=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    if metadata['model']!=digest(models[200]):
        raise ValueError('Rival database is not bound to the 200k model')
    raw=pickle.load((SCRATCH/'rivals.pkl').open('rb'))
    values=np.full(len(raw),np.nan); missing=defaultdict(list)
    for i,row in enumerate(raw):
        result=db.execute('SELECT p FROM claims WHERE owner=? AND target=?',row[:2]).fetchone()
        if result is None:
            missing[row[0]].append(i)
        else:
            values[i]=result[0]
    db.close(); reused=int(np.isfinite(values).sum())
    caches=work/'profiles'; caches.mkdir(exist_ok=True)
    source=connect(ROOT/'artifacts/train_store_v2'); source.execute('PRAGMA cache_size=-8192')
    base=extra=target_store=weight=None; classifiers={}
    for n,path in models.items():
        m=CatBoostClassifier(thread_count=1);m.load_model(str(path))
        if m.feature_names_!=NAMES:
            raise ValueError('CPU feature schema changed')
        classifiers[n]=m
    recomputed=reused_profiles=0
    for count,(o,indices) in enumerate(sorted(missing.items()),1):
        resources()
        if not re.fullmatch(r'S1-\d+',o):
            raise ValueError('Unexpected owner ID')
        path=caches/f'{o}.npz'
        previous=ROOT/'artifacts/improvement_v3/empty_target_rivals_v1/profiles'/path.name
        if path.exists() or previous.exists():
            with np.load(path if path.exists() else previous,allow_pickle=False) as z:
                ids=z['target_ids'];X=z['X']
            # Independently score cached full-context X with both frozen models.
            p1=classifiers[100].predict_proba(X,thread_count=1)[:,1]
            p2=classifiers[200].predict_proba(X,thread_count=1)[:,1]
            reused_profiles+=1
        else:
            if base is None:
                base,extra=retrievers(ROOT/'artifacts/train_index_v2',ROOT/'artifacts/training_lexicon_v1',0,None,
                    store=ROOT/'artifacts/train_store_v2')
                target_store=TargetStore(ROOT/'artifacts/train_store_v2');weight=Weighting(ROOT/'artifacts/train_index_v2')
            left=lookup(source,'anchors','entity_id',[o])[o]
            a,_=make_arrays([record_row(left)],base,extra,target_store,None,20)
            targets=lookup(source,'records','entity_id',a['target_ids'].tolist())
            X=enhanced_matrix(a,[record_row(left)],targets,weight);ids=a['target_ids']
            if not len(X):
                raise ValueError('Previously genuine rival has an empty forward list')
            p1=classifiers[100].predict_proba(X,thread_count=1)[:,1]
            p2=classifiers[200].predict_proba(X,thread_count=1)[:,1]
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,target_ids=ids,X=X,p100=p1,p200=p2);temp.replace(path);recomputed+=1
        index={str(t):i for i,t in enumerate(ids)}
        for i in indices:
            if raw[i][1] not in index:
                raise ValueError('Cannot invent context for an unretrieved rival')
            j=index[raw[i][1]]
            if abs(float(p1[j])-float(raw[i][4]))>1e-8:
                raise ValueError('Genuine forward p100k replay changed')
            values[i]=p2[j]
        write(work/'progress.json',dict(stage='missing_200k_rivals',profiles_done=count,profiles_total=len(missing)))
    source.close()
    if target_store is not None:
        target_store.close()
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Incomplete 200k rival probabilities')
    bound={}
    def save(split,path,p):
        rows=pickle.load(path.open('rb'))
        if len(rows)!=len(p) or not np.isfinite(p).all():
            raise ValueError('200k probabilities misaligned')
        np.save(root/f'{split}.npy',lg(p).astype(np.float32).reshape(-1,1))
        bound[split]=dict(path=str(path),sha256=digest(path),pairs=len(rows),features_sha256=digest(root/f'{split}.npy'))
    save('rivals',SCRATCH/'rivals.pkl',values)
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(z['left'],z['right']))}
            rows=pickle.load((SCRATCH/f'{split}.pkl').open('rb'));ix=np.asarray([keys[r[:2]] for r in rows])
            if not np.allclose(z['baseline_100000'][ix],[r[4] for r in rows],atol=1e-8,rtol=0):
                raise ValueError('Focal probability alignment changed')
            save(split,SCRATCH/f'{split}.pkl',z['baseline_200000'][ix])
    fresh=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
    receipt=read(fresh/'completion.json')
    for name in ('train5k.pkl','train5k_p200k.npy'):
        if digest(fresh/name)!=receipt['output_sha256'][name]:
            raise ValueError('Fresh 200k context changed')
    save('train5k',fresh/'train5k.pkl',np.load(fresh/'train5k_p200k.npy',allow_pickle=False))
    manifest=dict(complete=True,names=['genuine_p200k_logit'],inputs=bound,identity_sha256=digest(work/'identity.json'),
        original_rival_pairs=len(raw),database_scores_reused=reused,recomputed_owner_profiles=recomputed,
        reused_empty_target_profiles=reused_profiles,ready_for_submission=False,
        note='Auxiliary labels unused; genuine complete forward contexts; same p100k neural route')
    write(root/'manifest.json',manifest)


def run():
    work=ROOT/'artifacts/improvement_v3/p200_neural_pilot_v1';work.mkdir(parents=True,exist_ok=True)
    owner=ROOT/'artifacts/improvement_v3/owner200k_v1/evaluation/report.json'
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    end=time.monotonic()+12*3600
    def waiting(stage):
        if time.monotonic()>=end:raise TimeoutError('Finite 200k-neural pilot window expired')
        write(work/'progress.json',dict(stage=stage,ready_for_submission=False));time.sleep(30)
    while not owner.exists():waiting('waiting_for_completed_200k_owner_evaluation')
    features(work)
    while not (peer/'evaluation/report.json').exists():waiting('waiting_for_precision_matched_neural_scores')
    if not (work/'evaluation/report.json').exists():
        resources()
        with (work/'audit.log').open('a') as log:
            subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
                '--ce-root',str(peer/'half_calibration'),'--rival-scores',str(peer/'half_calibration/rivals.npy'),
                '--extra-root',str(work/'features'),'--output',str(work/'evaluation')],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
    result=read(work/'evaluation/report.json')
    write(work/'progress.json',dict(stage='complete',selected=result['selected'],
        development=result['models'][result['selected']['name']]['development'],ready_for_submission=False))


if __name__=='__main__':run()

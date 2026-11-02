"""Widen reverse-owner evidence for all routed empty-address tune/dev targets.

Ranking never reads ownership labels. Scores use genuine full forward retrieval
and complete 76-feature contexts. Save full new-owner profiles for future reuse;
never invent missing BM25/context features for an injected reverse pair.
"""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import sqlite3
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.deadline_retrieval import retrievers
from ber.direct_features import make_arrays
from ber.disk_index import TargetStore
from ber.store import connect,lookup,record_row
from v2_cpu_features import Weighting
from v2_owner_competition import Competition
from v2_run_pipeline import enhanced_matrix,NAMES,code_hashes
from v3_neural_test_pipeline import SCRATCH,resources
from v3_owner_assignment import load_text,text_logit
from v3_transfer_data import ROOT,PILOT,digest,write


OUT=ROOT/'artifacts/improvement_v3/empty_target_rivals_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=True);profiles=OUT/'profiles';profiles.mkdir(exist_ok=True)
    db=sqlite3.connect(OUT/'work.sqlite3');db.execute('PRAGMA cache_size=-8192')
    db.executescript('''CREATE TABLE IF NOT EXISTS targets(target TEXT PRIMARY KEY,done INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS selected(target TEXT,owner TEXT,PRIMARY KEY(target,owner));
        CREATE TABLE IF NOT EXISTS scored(owner TEXT PRIMARY KEY,pairs INTEGER);
        CREATE TABLE IF NOT EXISTS claims(target TEXT,owner TEXT,p100 REAL,p200 REAL,PRIMARY KEY(target,owner));''')
    model_paths={n:ROOT/f'artifacts/improvement_v2/cpu/fit_gate0_{n}k_v1/enhanced.cbm' for n in (100,200)}
    identity=dict(models={n:digest(p) for n,p in model_paths.items()},code=code_hashes(),runner_sha256=digest(__file__),
        split_sha256={s:digest(SCRATCH/f'{s}.pkl') for s in ('tune','dev')},
        route='All p100k>=.01 routed pairs with empty target address; no label selection',top_text_rivals=3)
    identity=json.loads(json.dumps(identity))
    if (OUT/'identity.json').exists():
        if json.loads((OUT/'identity.json').read_text())!=identity:raise ValueError('Empty-target experiment identity changed')
    else:write(OUT/'identity.json',identity)
    seeds=defaultdict(set)
    for split in ('tune','dev'):
        for r in pickle.load((SCRATCH/f'{split}.pkl').open('rb')):
            if not r[8].strip() or r[8].strip().upper() in ('NULL','<NULL>'):seeds[r[1]].add(r[0])
    with db:
        db.executemany('INSERT OR IGNORE INTO targets(target) VALUES(?)',[(t,) for t in seeds])
        db.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',[(t,o) for t,owners in seeds.items() for o in owners])
    source=connect(ROOT/'artifacts/train_store_v2');source.execute('PRAGMA cache_size=-8192')
    text_model=load_text(ROOT/'artifacts/improvement_v3/structural_v1/text_scorer.json')
    reverse=Competition(ROOT/'artifacts/train_store_v2',ROOT/'artifacts/improvement_v2/owner/train_s1_index_v1',top_k=10)
    start=time.monotonic()
    for j,(target,) in enumerate(db.execute('SELECT target FROM targets WHERE done=0 ORDER BY target').fetchall()):
        _,rivals=reverse.rivals(target);owners=lookup(source,'anchors','entity_id',list(rivals))
        row=lookup(source,'records','entity_id',[target])[target]
        chosen=sorted(((text_logit(owners[o],row,text_model),o) for o in rivals),reverse=True)[:3]
        with db:
            db.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',[(target,o) for _,o in chosen])
            db.execute('UPDATE targets SET done=1 WHERE target=?',(target,))
        if j%100==0:
            resources();write(OUT/'progress.json',dict(stage='reverse_discovery',targets_done=db.execute('SELECT count(*) FROM targets WHERE done=1').fetchone()[0],targets=len(seeds)))
    reverse.db.close();del reverse
    # Existing focal caches already contain genuine full forward probabilities.
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    focal=set()
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('left','right','reference_ids','query_ids','baseline_100000','baseline_200000')}
        with db:
            for i,(l,r) in enumerate(zip(a['left'],a['right'])):
                target=ids[int(r)];owner=ids[int(l)]
                if target in seeds:db.execute('INSERT OR REPLACE INTO claims VALUES(?,?,?,?)',(target,owner,float(a['baseline_100000'][i]),float(a['baseline_200000'][i])))
            for o in map(str,a['reference_ids']):db.execute('INSERT OR IGNORE INTO scored VALUES(?,?)',(o,-1));focal.add(o)
    del ids,a
    models={}
    for n,p in model_paths.items():
        m=CatBoostClassifier(thread_count=1);m.load_model(str(p));assert m.feature_names_==NAMES;models[n]=m
    base,extra=retrievers(ROOT/'artifacts/train_index_v2',ROOT/'artifacts/training_lexicon_v1',0,None,store=ROOT/'artifacts/train_store_v2')
    targets=TargetStore(ROOT/'artifacts/train_store_v2');weight=Weighting(ROOT/'artifacts/train_index_v2')
    pending=[r[0] for r in db.execute('SELECT DISTINCT c.owner FROM selected c LEFT JOIN scored s ON c.owner=s.owner WHERE s.owner IS NULL ORDER BY c.owner')]
    for j,owner in enumerate(pending):
        resources()
        if not re.fullmatch(r'S1-\d+',owner):raise ValueError('Unexpected owner ID format')
        row=lookup(source,'anchors','entity_id',[owner])[owner]
        a,_=make_arrays([record_row(row)],base,extra,targets,None,20)
        target_rows=lookup(source,'records','entity_id',a['target_ids'].tolist())
        X=enhanced_matrix(a,[record_row(row)],target_rows,weight)
        scores={n:m.predict_proba(X,thread_count=1)[:,1] if len(X) else np.empty(0) for n,m in models.items()}
        path=profiles/f'{owner}.npz';tmp=profiles/f'{owner}.tmp.npz'
        np.savez_compressed(tmp,reference_ids=a['reference_ids'],target_ids=a['target_ids'],X=X,
                            p100=scores[100],p200=scores[200]);tmp.replace(path)
        with db:
            for i,t in enumerate(a['target_ids']):
                if str(t) in seeds:db.execute('INSERT OR REPLACE INTO claims VALUES(?,?,?,?)',(str(t),owner,float(scores[100][i]),float(scores[200][i])))
            db.execute('INSERT INTO scored VALUES(?,?)',(owner,len(X)))
        if (j+1)%25==0:
            write(OUT/'progress.json',dict(stage='genuine_forward_profiles',new_profiles_done=j+1,new_profiles_total=len(pending),
                cached_focal_profiles=len(focal),seconds=time.monotonic()-start));print('profiles',j+1,len(pending),flush=True)
    claims=list(db.execute('SELECT target,owner,p100,p200 FROM claims ORDER BY target,owner'))
    owner_rows=lookup(source,'anchors','entity_id',list({r[1] for r in claims}));target_rows=lookup(source,'records','entity_id',list(seeds))
    rows=[]
    for t,o,p1,p2 in claims:
        l,r=owner_rows[o],target_rows[t]
        rows.append((o,t,-1,l['country'],p1,l['name'],l['address'],r['name'],r['address'],text_logit(l,r,text_model)))
    dest=OUT/'extra_rivals.pkl'
    with dest.open('wb') as f:pickle.dump(rows,f)
    np.save(OUT/'extra_rivals_p200.npy',np.asarray([r[3] for r in claims]))
    write(OUT/'completion.json',dict(complete=True,targets=len(seeds),claims=len(rows),
        owner_profiles=db.execute('SELECT count(*) FROM scored').fetchone()[0],new_profiles_this_run=len(pending),
        rows_sha256=digest(dest),identity_sha256=digest(OUT/'identity.json'),
        labels_read_for_ranking_or_auxiliary_claims=False,ready_for_submission=False,
        note='Bounded reverse top10/channel, text top3 plus all seed owners. Forward-unretrieved reverse pairs are not injected. No neural scoring or new decision policy yet.'))
    source.close();targets.close();db.close()


if __name__=='__main__':run()

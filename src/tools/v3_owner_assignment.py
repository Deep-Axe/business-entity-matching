"""Streaming candidate-score inversion; isolated, deletion-only owner assignment.

Partial ingestion is permitted, partial submissions are not. No production
controller/release is modified. Policy must be selected using tune owners only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from ber.data import read_tsv, write_tsv
from ber.store import lookup, record_row, connect
from v1_conflict_resolution import pair_features, prep_name, prep_address
from v3_transfer_data import ROOT, PILOT, digest, load, write

OUT = ROOT/'artifacts/improvement_v3/structural_v1'
MODEL = ROOT/'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm'
TEXT_COLUMNS = ['name_set','name_sort','name_exact','legal_agree','addr_set','addr_jaccard',
                'target_addr_missing','house_in_target','numbers_conflict','numbers_missing']


def text_features(left, right):
    return pair_features(prep_name(left['name']),prep_address(left['address']),
                         prep_name(right['name']),prep_address(right['address']))


def train_text(output=OUT/'text_scorer.json'):
    from sklearn.linear_model import LogisticRegression
    if Path(output).exists(): raise FileExistsError(output)
    m=json.loads((PILOT/'data/manifest.json').read_text())
    assert digest(PILOT/'data/train.npz')==m['splits']['train']['data_sha256']
    assert digest(PILOT/'data/records.jsonl')==m['record_sha256']
    rows=[json.loads(x) for x in (PILOT/'data/records.jsonl').open()]
    a=load('train')
    assert all(rows[int(i)]['fold']=='train' for i in np.unique(np.r_[a['left'],a['right']]))
    X=np.array([text_features(rows[int(l)],rows[int(r)]) for l,r in zip(a['left'],a['right'])],dtype=np.float32)
    model=LogisticRegression(C=1.,max_iter=500,random_state=20260926).fit(X,a['y'])
    result=dict(complete=True,columns=TEXT_COLUMNS,coefficients=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]),training_owners=len(a['reference_ids']),
                training_pairs=len(a['y']),training_fold='train',
                feature_code_sha256=digest(ROOT/'tools/v1_conflict_resolution.py'),
                data_sha256=m['splits']['train']['data_sha256'],
                label_selection='Only transfer-pilot train-fold pairs; no tune/dev labels or old v1 weights',
                score_semantics='Ranking logit on sampled hard negatives; NOT a calibrated posterior',
                base100k_model_sha256=digest(MODEL))
    write(output,result); return result


def load_text(path):
    result=json.loads(Path(path).read_text())
    if not result.get('complete') or result['columns']!=TEXT_COLUMNS or result['training_fold']!='train':
        raise ValueError('Unverified text scorer')
    if result['feature_code_sha256']!=digest(ROOT/'tools/v1_conflict_resolution.py'):
        raise ValueError('Text scorer implementation changed')
    return result


def text_logit(left,right,model):
    return float(np.dot(text_features(left,right),model['coefficients'])+model['intercept'])


def combined(p,text,alpha):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return np.log(p/(1-p))+alpha*np.asarray(text,dtype=float)


def winner(probabilities,text,alpha=0.,floor=0.,margin=0.):
    """Only multi-claimant groups are resolved; exact ties always abstain."""
    if len(probabilities)<2: raise ValueError('Only conflicting claimants belong here')
    score=combined(probabilities,text,alpha)
    if not np.isfinite(score).all(): raise ValueError('Non-finite owner score')
    order=np.argsort(-score,kind='stable'); first,second=order[:2]
    keep_floor=-math.inf if floor<=0 else math.log(floor/(1-floor))
    if score[first]<keep_floor or score[first]-score[second]<=max(0.,margin): return None
    return int(first)


def open_claims(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(directory/'claims.sqlite3')
    db.execute('PRAGMA cache_size=-16384');db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS partitions(partition INTEGER PRIMARY KEY,identity TEXT NOT NULL,refs INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS owners(owner TEXT PRIMARY KEY,country TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS claims(target TEXT NOT NULL,owner TEXT NOT NULL,p REAL NOT NULL,
                                        PRIMARY KEY(target,owner));
      CREATE INDEX IF NOT EXISTS claims_owner ON claims(owner);
    ''')
    return db


def ingest(controller,output=OUT/'test100k',model=MODEL,threshold=.76):
    model=Path(model)
    if not 0 < threshold < 1: raise ValueError('Invalid acceptance threshold')
    controller=Path(controller);queue=json.loads((controller/'queue.json').read_text())
    if not queue.get('config_sha256'): raise ValueError('Missing frozen controller binding')
    db=open_claims(output); identity=json.dumps({'controller_config':queue['config_sha256'],
                                               'model_sha256':digest(model),'threshold':threshold},sort_keys=True)
    old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if old and old[0]!=identity: raise ValueError('Claim cache belongs to another release')
    db.execute("INSERT OR IGNORE INTO meta VALUES('identity',?)",(identity,));db.commit()
    started=time.monotonic()
    for key,entry in sorted(queue['complete'].items(),key=lambda x:int(x[0])):
        part=int(key);directory=Path(entry['directory']);config=json.loads((directory/'config.json').read_text())
        manifest=json.loads((directory/'manifest.json').read_text())
        if not manifest.get('complete') or not manifest.get('partition_complete') or manifest['references']!=entry['references']:
            raise ValueError('Partition is not committed complete')
        if config['model_sha256']!=digest(model) or config['threshold']!=threshold or config['partition']!=part:
            raise ValueError('Unexpected partition model/threshold/identity')
        signature=digest(directory/'manifest.json')+digest(directory/'config.json')
        existing=db.execute('SELECT identity FROM partitions WHERE partition=?',(part,)).fetchone()
        if existing:
            if existing[0]!=signature: raise ValueError('An ingested partition changed')
            continue
        refs=0
        with db:
            for shard in sorted((directory/'shards').glob('[0-9]*')):
                done=json.loads((shard/'done.json').read_text())
                if done['config_sha256']!=digest(directory/'config.json'):
                    raise ValueError('Shard configuration identity mismatch')
                for name in ('pair_ids.npz','probabilities.npy'):
                    if digest(shard/name)!=done['output_sha256'][name]: raise ValueError('Shard checksum failed')
                with np.load(shard/'pair_ids.npz',allow_pickle=False) as z:
                    a={k:z[k] for k in ('reference_ids','countries','query_ids','target_ids')}
                p=np.load(shard/'probabilities.npy',allow_pickle=False)
                if len(a['reference_ids'])!=done['references'] or len(a['countries'])!=len(a['reference_ids']):
                    raise ValueError('Owner/country alignment mismatch')
                ref_hash=hashlib.sha256('\n'.join(map(str,a['reference_ids'])).encode()).hexdigest()
                if ref_hash!=done['reference_ids_sha256']:raise ValueError('Reference order changed')
                if len(p)!=done['candidate_pairs'] or p.shape!=a['query_ids'].shape or not np.isfinite(p).all():
                    raise ValueError('Invalid probability alignment')
                if ((p<0)|(p>1)).any() or len(a['target_ids'])!=len(p): raise ValueError('Invalid probabilities')
                if len(p) and (a['query_ids'].min()<0 or a['query_ids'].max()>=len(a['reference_ids'])):
                    raise ValueError('Invalid owner indices')
                db.executemany('INSERT INTO owners VALUES(?,?)',[(str(r),str(c)) for r,c in zip(a['reference_ids'],a['countries'])])
                db.executemany('INSERT INTO claims VALUES(?,?,?)',
                    [(str(a['target_ids'][i]),str(a['reference_ids'][a['query_ids'][i]]),float(p[i])) for i in np.flatnonzero(p>=threshold)])
                refs+=len(a['reference_ids'])
            if refs!=entry['references']: raise ValueError('Partition reference count mismatch')
            db.execute('INSERT INTO partitions VALUES(?,?,?)',(part,signature,refs))
        write(Path(output)/'progress.json',dict(complete=False,ingested_partitions=db.execute('SELECT count(*) FROM partitions').fetchone()[0],
              owners=db.execute('SELECT count(*) FROM owners').fetchone()[0],last_partition=part,seconds=time.monotonic()-started))
        print(json.dumps({'stage':'claims_ingested','partition':part,'refs':refs}),flush=True)
    n,refs=db.execute('SELECT count(*),sum(refs) FROM partitions').fetchone()
    result=dict(complete_ingestion=n==128 and refs==1732544,ingested_partitions=n,owners=refs,
                accepted_pairs=db.execute('SELECT count(*) FROM claims').fetchone()[0],
                partial_submission_allowed=False,controller_config_sha256=queue['config_sha256'])
    write(Path(output)/'progress.json',result);db.close();return result


def apply_policy(cache,policy,text_scorer,baseline_output,output,model=MODEL):
    cache,output,baseline_output=map(Path,(cache,output,baseline_output))
    if output.exists(): raise FileExistsError(output)
    rule=json.loads(Path(policy).read_text());text_model=load_text(text_scorer)
    if not rule.get('frozen_on_tune') or not rule.get('development_reported'):
        raise ValueError('Policy must be frozen on tune and independently reported on dev')
    if rule['text_scorer_sha256']!=digest(text_scorer) or rule['model_sha256']!=digest(model):
        raise ValueError('Policy model binding mismatch')
    db=open_claims(cache)
    identity=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    if identity['model_sha256']!=digest(model) or identity['threshold']!=rule.get('threshold',.76):
        raise ValueError('Claim cache and frozen policy differ')
    n,refs=db.execute('SELECT count(*),sum(refs) FROM partitions').fetchone()
    if n!=128 or refs!=1732544 or db.execute('SELECT count(*) FROM owners').fetchone()[0]!=1732544:
        raise ValueError('Refusing partial test output')
    source=connect(ROOT/'artifacts/test_store_v2');source.execute('PRAGMA cache_size=-8192')
    output.mkdir(parents=True)
    # A separate decisions database allows crash-safe reruns into a NEW output.
    decisions=sqlite3.connect(output/'decisions.sqlite3')
    decisions.execute('CREATE TABLE keep(target TEXT PRIMARY KEY,owner TEXT)')
    changed=abstained=0
    for (target,) in db.execute('SELECT target FROM claims GROUP BY target HAVING count(*)>1'):
        edges=list(db.execute('SELECT owner,p FROM claims WHERE target=? ORDER BY owner',(target,)))
        owners=lookup(source,'anchors','entity_id',[x[0] for x in edges]);row=lookup(source,'records','entity_id',[target])[target]
        t=[text_logit(owners[r],row,text_model) for r,_ in edges]
        choice=winner([p for _,p in edges],t,**rule['selected']) if rule['enabled'] else None
        if not rule['enabled']: continue
        decisions.execute('INSERT INTO keep VALUES(?,?)',(target,None if choice is None else edges[choice][0]))
        changed+=len(edges)-(choice is not None);abstained+=choice is None
        if changed%1000==0: decisions.commit()
    decisions.commit()
    references=ROOT/'6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv'
    def rows():
        for row in read_tsv(references):
            owner=row['entity_id'];kept=[]
            known=db.execute('SELECT country FROM owners WHERE owner=?',(owner,)).fetchone()
            if known is None or known[0]!=row['country']:raise ValueError('Test owner coverage/country mismatch')
            for (target,) in db.execute('SELECT target FROM claims WHERE owner=? ORDER BY target',(owner,)):
                decision=decisions.execute('SELECT owner FROM keep WHERE target=?',(target,)).fetchone()
                if decision is None or decision[0]==owner: kept.append(target)
            yield {'source1_entity_id':owner,'matched_entity_ids':','.join(kept)}
    write_tsv(output/'matching_results.tsv',rows(),['source1_entity_id','matched_entity_ids'])
    # Reuse the independently assembled full candidate output, never partial shards.
    assembly=json.loads((baseline_output/'assembly.json').read_text())
    if not assembly.get('complete_submission') or assembly['references']!=1732544:
        raise ValueError('Full baseline candidate assembly required')
    candidate=baseline_output/'candidate_pairs.tsv'
    if digest(candidate)!=assembly['output_sha256']['candidate_pairs.tsv']: raise ValueError('Candidate output changed')
    os.link(candidate,output/'candidate_pairs.tsv')
    result=dict(complete=True,submission_ready=False,requires_official_validation=True,references=refs,
                removed_edges=changed,abstained_targets=abstained,policy_sha256=digest(policy),
                matching_sha256=digest(output/'matching_results.tsv'),unchanged_candidates=True)
    write(output/'report.json',result);source.close();db.close();decisions.close();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train-text');p.add_argument('--output',type=Path,default=OUT/'text_scorer.json')
    p=sub.add_parser('ingest');p.add_argument('--controller',type=Path,default=ROOT/'artifacts/improvement_v2/inference/submission_controller_v1');p.add_argument('--output',type=Path,default=OUT/'test100k')
    p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--threshold',type=float,default=.76)
    p=sub.add_parser('apply');p.add_argument('--cache',type=Path,default=OUT/'test100k');p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--text-scorer',type=Path,default=OUT/'text_scorer.json');p.add_argument('--baseline-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=MODEL)
    args=vars(parser.parse_args());command=args.pop('command')
    print(json.dumps({'train-text':train_text,'ingest':ingest,'apply':apply_policy}[command](**args),indent=2))

"""Prepare genuinely new left-owner training data for the neural/CPU stacker.

Only train-fold S1s, excluding the 200k first-stage/CE training selection.
The reserved audit is not scored. Selection reads no ownership labels; later
feature generation explicitly reads training labels. CPU only, resumable shards.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from ber.data import FIELDS,read_tsv,write_tsv
from ber.store import connect,lookup
from v2_prepare_inputs import reservoir
from v3_transfer_data import ROOT,digest,write

OUT=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
TRAIN=ROOT/'artifacts/improvement_v2/selection/inputs_v1/train_200k/references.tsv'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'selection.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if digest(TRAIN)!=saved['excluded_200k_sha256']:raise ValueError('First-stage training selection changed')
        for name,value in saved['output_sha256'].items():
            if digest(OUT/name)!=value:raise ValueError('Selected references changed')
        return
    expected=json.loads((ROOT/'artifacts/improvement_v2/cpu/cloud200k_provenance_v1/prepared_manifest.json').read_text())['source_config']['references_sha256']
    if digest(TRAIN)!=expected:raise ValueError('Cannot prove excluded 200k training-owner identity')
    excluded={r['entity_id'] for r in read_tsv(TRAIN)}
    if len(excluded)!=200000:raise ValueError('Wrong training exclusion size')
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    selected,eligible=reservoir(db.execute("SELECT entity_id,name,address,country FROM anchors WHERE fold='train' ORDER BY entity_id"),
                                30000,202609267,excluded)
    db.close()
    if {r[0] for r in selected}&excluded:raise ValueError('First-stage owner overlap')
    def save(name,rows):write_tsv(OUT/name,(dict(zip(FIELDS,r)) for r in rows),FIELDS)
    save('references_30k.tsv',selected);save('references_5k.tsv',selected[:5000])
    write(manifest,dict(complete=True,training_fold='train',eligible_owners=eligible,selected_owners=30000,
        first_local_pilot_owners=5000,excluded_200k_sha256=expected,
        country_counts=dict(Counter(r[3] for r in selected)),selection_read_labels=False,reserved_audit_scored=False,
        caveat='Left-owner disjoint from first-stage and CE training; gallery text/alias lexicon may have seen these records. Not a pristine evaluation set.',
        output_sha256={n:digest(OUT/n) for n in ('references_30k.tsv','references_5k.tsv')}))


def features():
    source=OUT/'features_5k';model=ROOT/'artifacts/improvement_v2/cpu/fit_gate0_100k_v1'
    args=['tools/v2_run_pipeline.py','--index','artifacts/train_index_v2','--store','artifacts/train_store_v2',
          '--references',OUT/'references_5k.tsv','--lexicon','artifacts/training_lexicon_v1',
          '--model',model/'enhanced.cbm','--model-report',model/'enhanced_report.json',
          '--training-manifest','artifacts/improvement_v2/cpu/prepared_gate0_train_100k_v1/manifest.json',
          '--output',source,'--threshold','.76','--weighting-policy','forward_gallery_index_idf_v1',
          '--gate-terms','0','--workers','1','--labeled','--emit-matrix','--resume']
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def export():
    from catboost import CatBoostClassifier
    source=OUT/'features_5k';manifest=json.loads((source/'manifest.json').read_text())
    if not manifest['complete'] or manifest['references']!=5000:raise ValueError('Incomplete fresh training cache')
    model=CatBoostClassifier(thread_count=1);model.load_model(str(ROOT/'artifacts/improvement_v2/cpu/fit_gate0_200k_v1/enhanced.cbm'))
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    rows=[];p200=[];refs=[];truth=[];country=[];all_pairs=all_positives=route_positives=0
    for shard in sorted((source/'shards').glob('[0-9]*')):
        done=json.loads((shard/'done.json').read_text())
        for name,value in done['output_sha256'].items():
            if digest(shard/name)!=value:raise ValueError('Fresh shard changed')
        with np.load(shard/'pair_ids.npz',allow_pickle=False) as z:a=dict(z)
        p=np.load(shard/'probabilities.npy');X=np.load(shard/'X76.npy')
        ix=np.flatnonzero(p>=.01);p2=model.predict_proba(X[ix],thread_count=1)[:,1]
        anchors=lookup(db,'anchors','entity_id',a['reference_ids'].tolist())
        if any(r['fold']!='train' for r in anchors.values()):raise ValueError('Fresh selection escaped training fold')
        targets=lookup(db,'records','entity_id',a['target_ids'][ix].tolist())
        for i,pb in zip(ix,p2):
            owner=str(a['reference_ids'][a['query_ids'][i]]);target=str(a['target_ids'][i]);l,r=anchors[owner],targets[target]
            rows.append((owner,target,int(a['y'][i]),str(l['country']),float(p[i]),l['name'] or '',l['address'] or '',r['name'] or '',r['address'] or ''))
            p200.append(float(pb))
        refs.extend(map(str,a['reference_ids']));truth.extend(map(int,a['true_counts']));country.extend(map(str,a['countries']))
        all_pairs+=len(p);all_positives+=int(a['y'].sum());route_positives+=int(a['y'][ix].sum())
    db.close()
    with (OUT/'train5k.pkl').open('wb') as handle:pickle.dump(rows,handle)
    np.save(OUT/'train5k_p200k.npy',np.asarray(p200))
    np.savez_compressed(OUT/'train5k_owners.npz',reference_ids=np.asarray(refs),true_counts=np.asarray(truth),countries=np.asarray(country))
    write(OUT/'completion.json',dict(complete=True,owners=len(refs),all_candidate_pairs=all_pairs,
        all_retrieved_positives=all_positives,total_true_matches=sum(truth),routed_pairs=len(rows),routed_positives=route_positives,
        route='p100k>=0.01; no positive injection',cpu_models_out_of_sample_by_left_owner=True,
        reserved_audit_scored=False,ready_for_submission=False,training_only=True,
        output_sha256={n:digest(OUT/n) for n in ('train5k.pkl','train5k_p200k.npy','train5k_owners.npz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['all','select','export']);args=p.parse_args()
    prepare()
    if args.command=='all':features();export()
    elif args.command=='export':export()

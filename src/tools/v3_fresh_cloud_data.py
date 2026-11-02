"""Prepare an isolated 25k-owner cloud plan; export compact routed training data.

Neither subcommand launches an EC2 instance or a job. The plan is only eligible
after the current 200k queue is completely finished and the 5k fit merits growth.
"""
import argparse
from pathlib import Path
import pickle

import numpy as np

from ber.data import FIELDS, read_tsv, write_tsv
from v2_remote_job import validate_spec
from v3_transfer_data import ROOT, digest, write

WORK=ROOT/'artifacts/improvement_v3/fresh_cloud25k_v1'


def partition_owners(rows,part,parts=8,batch=200):
    return [r for i,r in enumerate(rows) if (i//batch)%parts==part]


def prepare():
    source=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'; selection=__import__('json').loads((source/'selection.json').read_text())
    for name,expected in selection['output_sha256'].items():
        if digest(source/name)!=expected:raise ValueError('Fresh owner selection changed')
    rows=list(read_tsv(source/'references_30k.tsv')); prior=list(read_tsv(source/'references_5k.tsv'))
    if len(rows)!=30000 or len(prior)!=5000 or rows[:5000]!=prior:raise ValueError('Expected exact nested training selection')
    selected=rows[5000:]
    if len({r['entity_id'] for r in rows})!=30000:raise ValueError('Duplicate fresh training owner')
    excluded_path=ROOT/'artifacts/improvement_v2/selection/inputs_v1/train_200k/references.tsv'
    if digest(excluded_path)!=selection['excluded_200k_sha256']:raise ValueError('200k exclusion identity changed')
    excluded={r['entity_id'] for r in read_tsv(excluded_path)}
    if excluded&{r['entity_id'] for r in selected}:raise ValueError('Fresh owners overlap first-stage training')
    WORK.mkdir(parents=True,exist_ok=True); refs=WORK/'references.tsv'
    if refs.exists():
        if list(read_tsv(refs))!=selected:raise ValueError('Existing cloud owner plan changed')
    else:write_tsv(refs,selected,FIELDS)
    specs=[]
    for part in range(8):
        out=f'artifacts/improvement_v3/fresh25k_features_p{part}_v1'
        args=['.venv/bin/python','tools/v2_run_pipeline.py','--index','artifacts/train_index_v2',
            '--store','artifacts/train_store_v2','--references',str(refs.relative_to(ROOT)),
            '--lexicon','artifacts/training_lexicon_v1','--model','artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm',
            '--model-report','artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_report.json',
            '--training-manifest','artifacts/improvement_v2/cpu/prepared_gate0_train_100k_v1/manifest.json',
            '--output',out,'--threshold','0.76','--weighting-policy','forward_gallery_index_idf_v1',
            '--gate-terms','0','--workers','2','--partition',str(part),'--partitions','8','--labeled','--emit-matrix','--resume']
        spec=dict(job=f'fresh25k_train_p{part}_v1',output=out,argv=args)
        validate_spec(spec);specs.append(dict(partition=part,expected_references=len(partition_owners(selected,part)),spec=spec))
    models={str(n):digest(ROOT/f'artifacts/improvement_v2/cpu/fit_gate0_{n}k_v1/enhanced.cbm') for n in (100,200)}
    value=dict(prepared=True,jobs_launched=False,owners=25000,partitions=8,workers_per_node=2,
        references_sha256=digest(refs),selection_sha256=digest(source/'selection.json'),models_sha256=models,
        runner_sha256=digest(ROOT/'tools/v2_run_pipeline.py'),exporter_sha256=digest(__file__),specs=specs,
        no_new_instances=True,no_plan_upgrade=True,launch_requirements=[
            'Exact 128/128 complete 200k queue with no assigned workers',
            'Fresh 5k-owner neural fit improves calibration; inspect country and dev tradeoffs',
            'Live budget and absolute timer still within approved USD120 cap',
            'Idle selected nodes, minimum 4GiB available RAM and 6GiB disk; local 8GiB disk reserve'],
        note='Prepared only. Existing fleet may be reused after review; no GPU or test outputs changed.')
    old=WORK/'plan.json'
    if old.exists() and __import__('json').loads(old.read_text())!=value:raise ValueError('Frozen cloud plan changed')
    write(old,value)


def export(plan,part,source,output,model200):
    from catboost import CatBoostClassifier
    from ber.store import connect,lookup
    from v2_run_pipeline import NAMES
    import json
    plan,source,output,model200=map(Path,(plan,source,output,model200));cfg=json.loads(plan.read_text())
    spec=cfg['specs'][part]
    if spec['partition']!=part:raise ValueError('Wrong cloud partition specification')
    manifest=json.loads((source/'manifest.json').read_text());config=json.loads((source/'config.json').read_text())
    expected={'reference_count':25000,'references_sha256':cfg['references_sha256'],'partition':part,'partitions':8,
        'model_sha256':cfg['models_sha256']['100'],'labels_available':True,'emit_matrix':True,
        'runner_code_sha256':cfg['runner_sha256'],'gate_terms':0}
    for key,want in expected.items():
        if config[key]!=want:raise ValueError('Wrong fresh cloud recipe: '+key)
    if not manifest['complete'] or not manifest['partition_complete'] or manifest['references']!=spec['expected_references']:
        raise ValueError('Incomplete fresh owner partition')
    if digest(model200)!=cfg['models_sha256']['200'] or digest(__file__)!=cfg['exporter_sha256']:
        raise ValueError('Cloud model/export code identity changed')
    output.mkdir(parents=True,exist_ok=True);done_path=output/'completion.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        if done['plan_sha256']!=digest(plan):raise ValueError('Existing compact export plan changed')
        for name,value in done['output_sha256'].items():
            if digest(output/name)!=value:raise ValueError('Retained compact export changed')
        return
    model=CatBoostClassifier(thread_count=1);model.load_model(str(model200))
    if model.feature_names_!=NAMES:raise ValueError('Wrong 200k model feature schema')
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    rows=[];features=[];prob200=[];owners=[];truth=[];countries=[];full_pairs=full_positive=0;seen=set()
    for shard in sorted((source/'shards').glob('[0-9]*')):
        receipt=json.loads((shard/'done.json').read_text())
        if receipt['config_sha256']!=digest(source/'config.json'):raise ValueError('Shard config changed')
        for name in ('X76.npy','pair_ids.npz','probabilities.npy'):
            if digest(shard/name)!=receipt['output_sha256'][name]:raise ValueError('Fresh shard data changed')
        with np.load(shard/'pair_ids.npz',allow_pickle=False) as z:a=dict(z)
        p=np.load(shard/'probabilities.npy',allow_pickle=False);X=np.load(shard/'X76.npy',allow_pickle=False)
        ix=np.flatnonzero(p>=.01);p2=model.predict_proba(X[ix],thread_count=1)[:,1] if len(ix) else np.empty(0)
        left=lookup(db,'anchors','entity_id',a['reference_ids'].tolist())
        if any(r['fold']!='train' for r in left.values()):raise ValueError('Cloud training owner escaped train fold')
        right=lookup(db,'records','entity_id',a['target_ids'][ix].tolist())
        for i,value in zip(ix,p2):
            o=str(a['reference_ids'][a['query_ids'][i]]);t=str(a['target_ids'][i]);l,r=left[o],right[t]
            rows.append((o,t,int(a['y'][i]),l['country'],float(p[i]),l['name'] or '',l['address'] or '',r['name'] or '',r['address'] or ''))
            prob200.append(float(value))
        features.append(X[ix]);current=list(map(str,a['reference_ids']))
        if seen&set(current):raise ValueError('Duplicate exported training owner')
        seen.update(current);owners.extend(current);truth.extend(map(int,a['true_counts']));countries.extend(map(str,a['countries']))
        full_pairs+=len(p);full_positive+=int(a['y'].sum())
    db.close()
    if len(owners)!=spec['expected_references']:raise ValueError('Compact owner coverage differs')
    with (output/'train.pkl').open('wb') as handle:pickle.dump(rows,handle)
    np.save(output/'p200.npy',np.asarray(prob200));np.save(output/'X76.npy',np.vstack(features))
    np.savez_compressed(output/'owners.npz',reference_ids=np.asarray(owners),true_counts=np.asarray(truth),countries=np.asarray(countries))
    write(done_path,dict(complete=True,training_only=True,partition=part,owners=len(owners),routed_pairs=len(rows),
        full_pairs=full_pairs,full_retrieved_positives=full_positive,total_true_matches=sum(truth),
        routed_positives=sum(r[2] for r in rows),plan_sha256=digest(plan),source_manifest_sha256=digest(source/'manifest.json'),
        output_sha256={n:digest(output/n) for n in ('train.pkl','p200.npy','X76.npy','owners.npz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');ex=sub.add_parser('export')
    for name in ('plan','source','output','model200'):ex.add_argument('--'+name,type=Path,required=True)
    ex.add_argument('--part',type=int,choices=range(8),required=True)
    args=vars(p.parse_args());cmd=args.pop('command');prepare() if cmd=='prepare' else export(**args)

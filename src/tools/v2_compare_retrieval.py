"""Compare matched full-gallery tune/dev caches using the frozen baseline matcher."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from ber.context_features import model_matrix
from ber.metrics import retrieval_metrics,score_arrays,tune_threshold
from ber.data import write_json
from ber.data import read_tsv
from ber.shards import file_hash

def read(features,model):
    root=Path(features)
    if not json.loads((root/'manifest.json').read_text())['complete']:
        raise ValueError('Incomplete feature cache')
    config=json.loads((root/'config.json').read_text())
    if not config['labels_available'] or config['gallery_fold'] is not None:
        raise ValueError('Need labeled full-gallery evaluation, not a training fold')
    directories=sorted((root/'shards').glob('[0-9]*'))
    expected=(config['reference_count']+config['query_batch_size']-1)//config['query_batch_size']
    if [p.name for p in directories]!=[f'{i:06d}' for i in range(expected)]:
        raise ValueError('Missing or unexpected feature shard')
    parts={};offset=0
    for directory in directories:
        path=directory/'features.npz'
        done=json.loads((directory/'done.json').read_text())
        for filename,sha in done['output_sha256'].items():
            if file_hash(directory/filename)!=sha:raise ValueError('Source shard checksum mismatch')
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        ids=[r['entity_id'] for r in read_tsv(directory/'references.tsv')]
        if ids!=a['reference_ids'].tolist() or len(ids)!=done['references']:
            raise ValueError('Reference identity/coverage mismatch')
        n=len(a['y'])
        if n!=done['candidate_pairs'] or any(len(a[k])!=n for k in ('query_ids','target_ids')):
            raise ValueError('Pair dimension mismatch')
        if n and (a['query_ids'].min()<0 or a['query_ids'].max()>=len(ids)):
            raise ValueError('Query index out of range')
        if not np.isin(a['y'],[0,1]).all() or (a['true_counts']<0).any():
            raise ValueError('Invalid evaluation labels')
        p=model.predict_proba(model_matrix(a,model.feature_names_),thread_count=1)[:,1]
        values={k:a[k] for k in ('query_ids','y','true_counts','reference_ids','countries','target_ids')}
        values['query_ids']=values['query_ids']+offset
        values['probabilities']=p
        offset+=len(a['reference_ids'])
        for k,v in values.items():parts.setdefault(k,[]).append(v)
    a={k:np.concatenate(v) for k,v in parts.items()}
    if offset!=config['reference_count']:raise ValueError('Cache coverage mismatch')
    if len(set(a['reference_ids']))!=offset:raise ValueError('Duplicate references')
    return a,config

def clean(score):return {k:v for k,v in score.items() if k!='entity_scores'}

def run(spec,model,output):
    spec=json.loads(Path(spec).read_text());output=Path(output)
    if list(spec)[0]!='gate1':raise ValueError('First recipe must be gate1 baseline')
    output.mkdir(parents=True,exist_ok=False)
    classifier=CatBoostClassifier(thread_count=1);classifier.load_model(model)
    results=[];baseline={};baseline_configs={};started=time.perf_counter()
    for name,paths in spec.items():
        arrays={};configs={}
        for split in ('tune','dev'):arrays[split],configs[split]=read(paths[split],classifier)
        recipe_keys=('gate_terms','channels','channel_order','top_k_per_channel','feature_names',
                     'index_config_sha256','lexicon_sha256','feature_code_sha256',
                     'direct_feature_code_sha256','retrieval_code_sha256')
        if any(configs['tune'][k]!=configs['dev'][k] for k in recipe_keys):
            raise ValueError('Tune/development recipe mismatch')
        for split in ('tune','dev'):
            if split not in baseline_configs:baseline_configs[split]=configs[split]
            for k in recipe_keys:
                if k!='gate_terms' and configs[split][k]!=baseline_configs[split][k]:
                    raise ValueError('Retrieval comparison changes more than the gate: '+k)
        if set(arrays['tune']['reference_ids']) & set(arrays['dev']['reference_ids']):raise ValueError('Split leakage')
        t=arrays['tune'];threshold,score=tune_threshold(t['query_ids'],t['y'],t['probabilities'],t['true_counts'])
        item=dict(recipe=name,threshold=threshold,tune_macro_f05=score,gate_terms=configs['tune']['gate_terms'],
                  source_config_sha256={s:file_hash(Path(paths[s])/'config.json') for s in ('tune','dev')},
                  orchestration_runner_sha256={s:configs[s]['runner_code_sha256'] for s in ('tune','dev')},splits={})
        for split,a in arrays.items():
            accepted=score_arrays(a['query_ids'],a['y'],a['probabilities'],a['true_counts'],threshold)
            fixed=score_arrays(a['query_ids'],a['y'],a['probabilities'],a['true_counts'],.74)
            pairs=set(zip(a['query_ids'][a['y']==1].tolist(),a['target_ids'][a['y']==1].tolist()))
            if split not in baseline:
                baseline[split]=dict(ids=a['reference_ids'],pairs=pairs,scores=fixed['entity_scores'],truth=a['true_counts'])
            b=baseline[split]
            if not np.array_equal(b['ids'],a['reference_ids']):raise ValueError('Recipes use different/order references')
            if not np.array_equal(b['truth'],a['true_counts']):raise ValueError('Recipes use different ground truth')
            delta=accepted['entity_scores']-b['scores'];rng=np.random.default_rng(20260925)
            draws=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)]
            item['splits'][split]=dict(retrieval=retrieval_metrics(a['query_ids'],a['y'],a['true_counts']),
                frozen_threshold=clean(fixed),retuned=clean(accepted),
                new_positive_pairs_vs_gate1=len(pairs-b['pairs']),lost_positive_pairs_vs_gate1=len(b['pairs']-pairs),
                delta_vs_frozen_gate1=float(delta.mean()),paired_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                by_country={str(c):dict(references=int((a['countries']==c).sum()),macro_f05=float(accepted['entity_scores'][a['countries']==c].mean())) for c in np.unique(a['countries'])})
            np.savez_compressed(output/f'{name}_{split}.npz',**a)
        results.append(item);print(json.dumps(item),flush=True)
    selected=max(results,key=lambda r:r['tune_macro_f05'])
    report=dict(results=results,selected_on_tune=selected['recipe'],model_sha256=file_hash(model),seconds=time.perf_counter()-started,
                comparability='Feature/retrieval/index/lexicon recipe hashes match except gate; historical orchestration runner hashes differ and are reported explicitly.',
                caveat='Frozen gate1 matcher; changed retrieval can require retraining. Development never selects threshold. No final audit used.')
    write_json(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('spec','model','output'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))

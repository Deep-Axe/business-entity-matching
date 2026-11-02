"""Tune a bounded fusion grid on tuning only, then report fixed development results."""
import argparse
import json
from pathlib import Path
import numpy as np
from v2_cascade_cache import digest,write_new_json
from v2_cascade_train import entity_scores,fuse,score


def load_scored(prepared,scored):
    prepared,scored = Path(prepared),Path(scored)
    source = json.loads((prepared/'manifest.json').read_text())
    pred = json.loads((scored/'manifest.json').read_text())
    if not source.get('complete') or not pred.get('complete'):
        raise ValueError('Cascade preparation/scoring incomplete')
    if pred['prepared_manifest_sha256'] != digest(prepared/'manifest.json'):
        raise ValueError('Scored inputs differ from prepared manifest')
    expected = {x['shard']:x for x in source['shards']}
    if len(pred['shards']) != len(expected) or {x['shard'] for x in pred['shards']} != set(expected):
        raise ValueError('Neural shard coverage differs from prepared cache')
    values = {k:[] for k in ('query_ids','reference_ids','target_ids','y','true_counts','country','tree','neural','routed')}
    qoffset = offset = 0
    for item in pred['shards']:
        original = expected[item['shard']]
        if any(item.get(k) != v for k,v in original.items()):
            raise ValueError('Scored shard identity differs from prepared input')
        p = prepared/'shards'/item['shard']/'arrays.npz'
        n = scored/'shards'/item['shard']/'neural.npy'
        if digest(p) != item['arrays_sha256'] or digest(n) != item['neural_sha256']:
            raise ValueError('Prepared arrays or neural predictions changed')
        with np.load(p,allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        neural = np.load(n,allow_pickle=False)
        if item['global_pair_start'] != offset or item['global_query_start'] != qoffset:
            raise ValueError('Noncontiguous aggregate alignment')
        route = np.zeros(len(a['tree']),dtype=bool)
        route[a['routed_indices']] = True
        if neural.shape != route.shape or not np.isfinite(neural[route]).all() or not np.isnan(neural[~route]).all():
            raise ValueError('Invalid/missing neural score coverage')
        if not np.isin(a['y'],[0,1]).all() or (a['true_counts']<0).any():
            raise ValueError('Calibration/evaluation requires provided labels')
        for key in ('reference_ids','target_ids','y','true_counts','tree'):
            values[key].append(a[key])
        values['query_ids'].append(a['query_ids']+qoffset)
        values['country'].append(a['countries'])
        values['neural'].append(neural)
        values['routed'].append(route)
        qoffset += len(a['reference_ids'])
        offset += len(a['tree'])
    if qoffset != source['references'] or offset != source['pairs']:
        raise ValueError('Full cache coverage mismatch')
    return {k:np.concatenate(v) for k,v in values.items()},source,pred


def paired_delta(a,new_probability,new_threshold,old_probability,old_threshold):
    d = entity_scores(a,new_probability,new_threshold)-entity_scores(a,old_probability,old_threshold)
    rng = np.random.default_rng(20260926)
    boot = np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(3000)])
    return {'mean':float(d.mean()),'paired_owner_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
            'improved_owners':int((d>0).sum()),'worsened_owners':int((d<0).sum()),
            'conditional_on_selected_policy':True}


def detailed_score(a,p,t):
    result = score(a,p,t)
    chosen = p>=t
    tp = int((chosen & (a['y']==1)).sum())
    fp = int((chosen & (a['y']==0)).sum())
    result.update(tp=tp,fp=fp,precision=tp/max(1,tp+fp),recall=tp/max(1,int(a['true_counts'].sum())))
    counts = np.bincount(a['query_ids'],weights=chosen,minlength=len(a['true_counts']))
    singleton = a['true_counts']==0
    result['singleton_false_merge_rate'] = float((counts[singleton]>0).mean()) if singleton.any() else None
    return result


def run(tune_prepared,tune_scored,dev_prepared,dev_scored,enhanced_report,baseline_data,output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for split in ('tune','dev'):
        if (output.parent/f'{split}_selected_probabilities.npy').exists():
            raise FileExistsError(output.parent/f'{split}_selected_probabilities.npy')
    enhanced = json.loads(Path(enhanced_report).read_text())
    original_manifest = json.loads((Path(baseline_data)/'manifest.json').read_text())
    data,provenance,original = {},{},{}
    for split,prepared,scored in (('tune',tune_prepared,tune_scored),('dev',dev_prepared,dev_scored)):
        data[split],source,pred = load_scored(prepared,scored)
        if source['probability_binding']['model_sha256'] != enhanced['model_sha256']:
            raise ValueError('Enhanced model/report identity mismatch')
        original[split] = dict(np.load(Path(baseline_data)/f'{split}.npz',allow_pickle=False))
        for key in ('reference_ids','target_ids','query_ids','y','true_counts','country'):
            if not np.array_equal(data[split][key],original[split][key]):
                raise ValueError(f'Baseline/enhanced row identity or labels differ: {split}/{key}')
        provenance[split] = {'prepared_manifest_sha256':digest(Path(prepared)/'manifest.json'),
            'scored_manifest_sha256':digest(Path(scored)/'manifest.json'),
            'tree_probability_binding':source['probability_binding'],
            'neural_checkpoint_sha256':pred['checkpoint_sha256'],'routing':source['routing'],
            'routed_pairs':int(data[split]['routed'].sum()),'candidate_pairs':len(data[split]['tree']),
            'routing_fraction':float(data[split]['routed'].mean()),'neural_seconds':pred['seconds']}
        provenance[split]['original_array_sha256'] = digest(Path(baseline_data)/f'{split}.npz')
    if provenance['tune']['neural_checkpoint_sha256'] != provenance['dev']['neural_checkpoint_sha256']:
        raise ValueError('Different neural checkpoints across tuning/development')
    if set(data['tune']['reference_ids']) & set(data['dev']['reference_ids']):
        raise ValueError('Tuning/development owner overlap')
    if any(provenance[s]['routing'] != {'lower':.01,'upper':.99} for s in provenance):
        raise ValueError('Expected fixed .01-.99 route for this controlled combination')
    rows = []
    thresholds = np.unique(np.r_[np.arange(.4,.981,.02),enhanced['threshold'],.74,.99])
    for weight in (0.,.1,.25,.5,.75,1.):
        mixed = fuse(data['tune']['tree'],data['tune']['neural'],data['tune']['routed'],weight)
        for threshold in thresholds:
            rows.append({'neural_weight':weight,'threshold':float(threshold),
                'macro_f05':float(entity_scores(data['tune'],mixed,threshold).mean())})
    best = max(rows,key=lambda r:(r['macro_f05'],-r['neural_weight']))
    scores = {}
    for split,a in data.items():
        mixed = fuse(a['tree'],a['neural'],a['routed'],best['neural_weight'])
        scores[split] = {'original_tree':detailed_score(a,original[split]['tree'],.74),
            'enhanced_tree':detailed_score(a,a['tree'],enhanced['threshold']),
            'enhanced_plus_neural':detailed_score(a,mixed,best['threshold']),
            'delta_vs_enhanced':paired_delta(a,mixed,best['threshold'],a['tree'],enhanced['threshold']),
            'delta_vs_original':paired_delta(a,mixed,best['threshold'],original[split]['tree'],.74)}
        with (output.parent/f'{split}_selected_probabilities.npy').open('xb') as handle:
            np.save(handle,mixed)
        scores[split]['selected_probability_sha256'] = digest(output.parent/f'{split}_selected_probabilities.npy')
    result = {'complete':True,'no_neural_retraining':True,'final_audit_used':False,
        'selection':'Only full tuning labels selected weight/threshold; fixed policy applied to development.',
        'gate_terms':1,'enhanced_tree_threshold':enhanced['threshold'],
        'enhanced_model_sha256':enhanced['model_sha256'],'original_model_sha256':original_manifest['model_sha256'],
        'provenance':provenance,'selected_on_tune':best,'scores':scores,'tuning_grid':rows,
        'runner_sha256':digest(__file__),
        'limitations':['No France provided-label evaluation.','Results conditional on current gate-1 candidates.',
                       'Development is not the final independent audit or hidden leaderboard.']}
    write_new_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tuning_grid','provenance')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tune-prepared','tune-scored','dev-prepared','dev-scored','enhanced-report','baseline-data','output'):
        parser.add_argument('--'+name,required=True)
    run(**vars(parser.parse_args()))

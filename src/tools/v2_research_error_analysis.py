"""Read-only development error taxonomy; never touches the reserved final audit.

Counterfactuals use development labels only to describe headroom, NOT as
deployable policies or measured improvements from a new technique.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from ber.data import write_json
from ber.metrics import score_arrays,retrieval_metrics
from ber.shards import file_hash,gold_for
from ber.store import connect,lookup,record_row
from v2_cascade_cache import validate_inputs

BASE=Path('artifacts/improvement_v2')

def owner_scores(tp,pred,true):
    denominator=.25*true+pred
    return np.divide(1.25*tp,denominator,out=np.ones(len(true),dtype=float),where=denominator!=0)

def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    source=BASE/'cloud/fleet_v1/collected/v2_gate0_dev'
    prepared=BASE/'cpu/prepared_gate0_full_dev_v1'
    probability=BASE/'cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.npy'
    config,matrix,binding,_,_,p=validate_inputs(source,probability,prepared/'manifest.json',
        'artifacts/train_store_v2','artifacts/train_index_v2')
    if matrix['references']!=3000 or binding['model_sha256']!='aa236b4c971c83b7226edfe9e39d4f80b14bcee4a0188f7ed46cb996f6e62a49':
        raise ValueError('Expected frozen100k and existing3000development references only')
    a={key:np.load(prepared/(key+'.npy'),mmap_mode='r',allow_pickle=False)
       for key in ('X','y','query_ids','true_counts','reference_ids','countries')}
    names=matrix['names'];q=a['query_ids'];y=a['y'];true=a['true_counts'];n=len(true)
    targets=[]
    for item in matrix['shards']:
        with np.load(source/'shards'/item['shard']/'features.npz',allow_pickle=False) as z:
            if (not np.array_equal(z['y'],y[item['pair_start']:item['pair_end']])
                or not np.array_equal(z['query_ids']+item['query_start'],q[item['pair_start']:item['pair_end']])
                or not np.array_equal(z['reference_ids'],a['reference_ids'][item['query_start']:item['query_end']])):
                raise ValueError('Matrix/source alignment differs')
            targets.extend(map(str,z['target_ids']))
    targets=np.asarray(targets);accepted=p>=.76
    count=lambda mask:np.bincount(q[mask],minlength=n)
    pred=count(accepted);tp=count(accepted&(y==1));retrieved=count(y==1)
    fp=pred-tp;fn=true-tp;missing=true-retrieved;rejected=retrieved-tp
    if (missing<0).any():raise ValueError('Invalid true/retrieval counts')
    score=owner_scores(tp,pred,true);loss=1-score;oracle=owner_scores(retrieved,retrieved,true)
    categories={
        'correct_empty':(true==0)&(pred==0),
        'false_merge_on_singleton':(true==0)&(pred>0),
        'positive_owner_all_gold_missing_from_retrieval':(true>0)&(retrieved==0),
        'positive_owner_retrieved_gold_but_predicted_empty':(true>0)&(retrieved>0)&(pred==0),
        'nonempty_fp_only':(true>0)&(retrieved>0)&(pred>0)&(fp>0)&(fn==0),
        'nonempty_fn_only':(true>0)&(retrieved>0)&(pred>0)&(fp==0)&(fn>0),
        'nonempty_fp_and_fn':(true>0)&(retrieved>0)&(pred>0)&(fp>0)&(fn>0),
        'perfect_nonempty':(true>0)&(retrieved>0)&(pred>0)&(fp==0)&(fn==0)}
    if not np.all(sum(categories.values())==1):raise ValueError('Taxonomy not disjoint/exhaustive')
    col=lambda name:np.asarray(a['X'][:,names.index(name)])
    masks={'true_positive':accepted&(y==1),'false_positive':accepted&(y==0),
        'retrieved_false_negative':~accepted&(y==1)}
    slices={
        'either_address_missing':(col('reference_address_missing')>.5)|(col('target_address_missing')>.5),
        'either_name_nonascii':(col('reference_name_nonascii')>.5)|(col('target_name_nonascii')>.5),
        'near_name_weak_address':(np.maximum(col('name_ratio'),col('ascii_name_ratio'))>=.9)&(col('address_ratio')<.5),
        'short_number_disjoint':col('v2_short_number_disjoint')>.5,
        'long_number_disjoint':col('v2_long_number_disjoint')>.5,
        'distinctive_name_exact_token_coverage_below_half':col('v2_ascii_name_reference_coverage')<.5,
        'fuzzy_name_high_exact_tokens_low':(col('ascii_name_ratio')>=.8)&(col('v2_ascii_name_reference_coverage')<.5),
        's2_target':col('target_is_s2')>.5}
    interesting=['name_ratio','ascii_name_ratio','address_ratio','v2_ascii_name_reference_coverage',
        'v2_address_reference_coverage','v2_number_jaccard','retrieval_channels','context_log_near_names']
    country={}
    for c in np.unique(a['countries']):
        owners=a['countries']==c;pairmask=owners[q]
        country[str(c)]=dict(references=int(owners.sum()),macro_f05=float(score[owners].mean()),
            loss_sum=float(loss[owners].sum()),tp=int(tp[owners].sum()),fp=int(fp[owners].sum()),
            retrieval_misses=int(missing[owners].sum()),scored_rejected_positives=int(rejected[owners].sum()),
            positive_predicted_empty=int(((true>0)&(pred==0)&owners).sum()),
            singleton_false_merges=int(((true==0)&(pred>0)&owners).sum()))
    band_edges=[0,.01,.1,.3,.5,.7,.76,.9,.99,1.0000000001]
    bands=[]
    for lower,upper in zip(band_edges,band_edges[1:]):
        selection=(p>=lower)&(p<upper);k=int(selection.sum())
        bands.append(dict(lower=lower,upper=upper,pairs=k,positive_pairs=int(y[selection].sum()),
            empirical_positive_fraction=float(y[selection].mean()) if k else None,
            mean_probability=float(p[selection].mean()) if k else None,
            owners=int(len(np.unique(q[selection])))))
    report=dict(complete=True,scope='Existing3000development only; exploratorydiagnosis, notfinalaudit',
        references=n,candidate_pairs=len(p),gold_pairs=int(true.sum()),threshold=.76,
        macro_f05=float(score.mean()),tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum()),
        retrieval_missed_gold_pairs=int(missing.sum()),retrieved_rejected_gold_pairs=int(rejected.sum()),
        retrieval=retrieval_metrics(q,y,true),
        diagnostic_label_oracles_not_deployable=dict(
            perfect_classifier_with_current_candidates=float(oracle.mean()),
            remove_all_false_positives_only=float(owner_scores(tp,tp,true).mean()),
            recover_all_retrieved_false_negatives_keep_current_fp=float(owner_scores(retrieved,retrieved+fp,true).mean()),
            retrieval_loss=float((1-oracle).mean()),classifier_gap=float((oracle-score).mean()),
            classifier_fraction_of_current_score_gap=float((oracle-score).sum()/loss.sum())),
        owner_categories={key:dict(references=int(mask.sum()),loss_sum=float(loss[mask].sum()),
            share_of_total_loss=float(loss[mask].sum()/loss.sum())) for key,mask in categories.items()},
        countries=country,
        pair_slices={name:{key:dict(pairs=int((mask&selection).sum()),
            fraction=float((mask&selection).sum()/max(1,mask.sum()))) for key,selection in slices.items()}
            for name,mask in masks.items()},
        feature_quantiles={name:{f:np.quantile(col(f)[mask],[.1,.5,.9]).tolist() for f in interesting}
            for name,mask in masks.items()},probability_bands=bands,
        provenance=dict(model_sha256=binding['model_sha256'],probabilities_sha256=file_hash(probability),
            matrix_manifest_sha256=file_hash(prepared/'manifest.json'),source_config_sha256=file_hash(source/'config.json')),
        final_audit_accessed=False,test_labels_used=False,production_modified=False,
        warnings=['Slices overlap and are descriptive associations, not causal explanations.',
            'Oracle scores use true development labels; they cannot be deployed or claimed as model gains.',
            'Development-guided ideas need fresh confirmation; do not mine development errors as training examples.'])
    output.mkdir(parents=True);write_json(output/'report.json',report)
    # Inspect all error pairs using supplied data, preserving a local review table.
    db=connect('artifacts/train_store_v2')
    owners=lookup(db,'anchors','entity_id',list(map(str,a['reference_ids'])))
    needed=set(targets[masks['false_positive']|masks['retrieved_false_negative']])
    gold=gold_for(db,list(map(str,a['reference_ids'])))
    missing_pairs=[]
    for i,eid in enumerate(a['reference_ids']):
        candidates=set(targets[q==i]);absent=gold[str(eid)]-candidates
        if len(absent)!=missing[i]:raise ValueError('Missinggold pair reconstruction differs')
        for tid in sorted(absent):missing_pairs.append((i,tid));needed.add(tid)
    raw=lookup(db,'records','entity_id',list(needed))
    with (output/'error_pairs.jsonl').open('x') as handle:
        def emit(i,tid,kind,prob,features=None):
            handle.write(json.dumps(dict(reference_id=str(a['reference_ids'][i]),target_id=tid,kind=kind,
                probability=prob,reference=record_row(owners[str(a['reference_ids'][i])]),
                target=record_row(raw[tid]),features=features),ensure_ascii=False)+'\n')
        for kind in ('false_positive','retrieved_false_negative'):
            for j in np.flatnonzero(masks[kind]):
                emit(int(q[j]),str(targets[j]),kind,float(p[j]),{f:float(col(f)[j]) for f in interesting})
        for i,tid in missing_pairs:emit(i,tid,'retrieval_miss',None)
    db.close()
    print(json.dumps({k:report[k] for k in ('references','macro_f05','tp','fp','fn',
        'retrieval_missed_gold_pairs','retrieved_rejected_gold_pairs','diagnostic_label_oracles_not_deployable',
        'owner_categories','countries')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))

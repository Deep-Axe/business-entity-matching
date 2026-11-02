"""Read-only replay and error attribution for the frozen 100k owner policy.

No tuning or inference changes. Ground truth is read only for existing labeled
development owners. Test evidence is structural, never claimed as test accuracy.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

import numpy as np
from rapidfuzz import fuzz

from ber.store import connect, lookup
from v3_transfer_data import ROOT, PILOT, digest, write
from v3_structural_features import folded, leet_view, house_number, street_component, single_digit_change

BASE = ROOT/'artifacts/improvement_v3/structural_v1'


def score(tp, fp, truth):
    tp, fp, truth = map(lambda x: np.asarray(x, dtype=float), (tp, fp, truth))
    den = truth + 4*(tp+fp)
    return np.divide(5*tp, den, out=np.ones_like(den), where=den != 0)


def attribution(tp, fp, truth):
    """Two-group Shapley split: average FN-first and FP-first oracle repairs."""
    current = score(tp, fp, truth)
    fix_fn = score(truth, fp, truth)
    fix_fp = score(tp, np.zeros_like(fp), truth)
    fn_loss = ((fix_fn-current)+(1-fix_fp))/2
    fp_loss = ((fix_fp-current)+(1-fix_fn))/2
    if not np.allclose(fn_loss+fp_loss, 1-current):
        raise ValueError('Loss attribution must add to actual macro score gap')
    return current, fn_loss, fp_loss


def choose(edges, rule):
    """Independent scalar implementation; input includes accepted claims only."""
    if len(edges) < 2:
        return edges[0][0] if edges else None, None
    ranked = []
    for owner, p, text in edges:
        p = min(1-1e-7, max(1e-7, float(p)))
        ranked.append((math.log(p/(1-p))+rule['alpha']*text, owner))
    ranked.sort(reverse=True)
    margin = ranked[0][0]-ranked[1][0]
    floor = -math.inf if rule['floor'] <= 0 else math.log(rule['floor']/(1-rule['floor']))
    return (None if ranked[0][0] < floor or margin <= max(0, rule['margin']) else ranked[0][1]), margin


def missing(address):
    return not address.strip() or address.strip().upper() in ('NULL', '<NULL>')


def legacy_category(left, right):
    """Reproduce the supplied heuristic table; not verified generator classes."""
    if missing(right['address']):
        return 'target_address_empty'
    if any(ord(c) > 0x900 for c in right['name']):
        return 'native_script_name_legacy'
    if fuzz.token_set_ratio(left['name'].lower(), right['name'].lower()) < 50:
        return 'unrelated_name_legacy'
    a, b = re.findall(r'\d+', left['address']), re.findall(r'\d+', right['address'])
    if not a or not b:
        relation = 'no_numbers'
    elif a[0] in b:
        relation = 'equal'
    else:
        relation = None
        for y in b:
            x = a[0]
            if len(y) == len(x)-1 and (x[1:] == y or x[:-1] == y):
                relation = 'digit_deleted'; break
            if len(y) == len(x) and sum(c != d for c,d in zip(x,y)) == 1:
                relation = 'one_digit_changed' if abs(int(x)-int(y)) >= 10 or len(x) == 1 else 'last_digit_changed'; break
        if relation is None:
            relation = 'small_offset' if any(abs(int(a[0])-int(y)) <= 25 for y in b) else 'different'
    return 'house:'+relation


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    manifest = json.loads((PILOT/'data/manifest.json').read_text())
    data = PILOT/'data/dev.npz'
    if digest(data) != manifest['splits']['dev']['data_sha256']:
        raise ValueError('Development cache identity changed')
    records_path = PILOT/'data/records.jsonl'
    if digest(records_path) != manifest['record_sha256']:
        raise ValueError('Record cache identity changed')
    with np.load(data, allow_pickle=False) as z:
        a = {k:z[k] for k in ['query_ids','true_counts','reference_ids','countries','y','left','right','baseline_100000']}
    required = set(map(int, np.r_[a['left'], a['right']]))
    records = {}
    with records_path.open() as f:
        for i, line in enumerate(f):
            if i in required:
                records[i] = json.loads(line)
    by_id = {r['id']:r for r in records.values()}
    refs = list(map(str, a['reference_ids']))
    owner_index = {r:i for i,r in enumerate(refs)}
    pair_index = {(refs[int(q)], records[int(r)]['id']):i for i,(q,r) in enumerate(zip(a['query_ids'],a['right']))}
    if len(pair_index) != len(a['y']):
        raise ValueError('Duplicate cached candidate pair')
    source = connect(ROOT/'artifacts/train_store_v2')
    gold = {owner:{r[0] for r in source.execute('SELECT target_id FROM labels WHERE reference_id=?',(owner,))} for owner in refs}
    if not np.array_equal([len(gold[o]) for o in refs], a['true_counts']):
        raise ValueError('Ground truth/cache count mismatch')
    absent = set().union(*gold.values())-by_id.keys()
    for eid, r in lookup(source, 'records', 'entity_id', list(absent)).items():
        by_id[eid] = dict(id=eid,name=r['name'],address=r['address'],country=r['country'])
    for (owner, target), i in pair_index.items():
        if int(target in gold[owner]) != int(a['y'][i]):
            raise ValueError('Cached labels disagree with actual label table')
    raw = {o:set() for o in refs}
    for (owner, target), i in pair_index.items():
        if a['baseline_100000'][i] >= .76:
            raw[owner].add(target)
    after = {o:set(v) for o,v in raw.items()}
    rival_path = BASE/'owner_eval/rivals.sqlite3'
    db = sqlite3.connect(rival_path.resolve().as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-8192')
    policy_path = BASE/'owner_eval/policy.json'
    policy = json.loads(policy_path.read_text())
    rule = policy['selected']
    accepted = defaultdict(list)
    for target, owner, p, text in db.execute('SELECT target,owner,p,text FROM claims WHERE p>=.76 ORDER BY target,owner'):
        accepted[target].append((owner,p,text))
    removed = []
    for target, edges in accepted.items():
        if len(edges)<2:
            continue
        chosen, margin = choose(edges, rule)
        for owner,p,text in edges:
            if owner in after and target in raw[owner] and owner != chosen:
                after[owner].remove(target)
                removed.append(dict(owner=owner,target=target,true_match=target in gold[owner],
                                    own_probability=p,own_text_logit=text,winner=chosen,margin=margin,
                                    reason='near_tie_abstention' if chosen is None else 'another_owner_won',
                                    competing_claims=[dict(owner=o,probability=s,text_logit=t) for o,s,t in edges]))
    metrics, arrays = {}, {}
    for name, predictions in [('raw100k',raw),('owner100k',after)]:
        tp = np.array([len(predictions[o]&gold[o]) for o in refs])
        fp = np.array([len(predictions[o]-gold[o]) for o in refs])
        s, fn_loss, fp_loss = attribution(tp,fp,a['true_counts'])
        fn = a['true_counts']-tp
        arrays[name] = dict(tp=tp,fp=fp,fn=fn,scores=s,fn_loss=fn_loss,fp_loss=fp_loss)
        metrics[name] = dict(macro_f05=float(s.mean()),tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum()),
            errors=int(fp.sum()+fn.sum()),loss=1-float(s.mean()),fn_loss=float(fn_loss.mean()),fp_loss=float(fp_loss.mean()),
            fn_share_of_loss=float(fn_loss.sum()/(fn_loss.sum()+fp_loss.sum())),
            positive_owners_predicted_empty=int(((a['true_counts']>0)&((tp+fp)==0)).sum()),
            false_merges_on_unmatched_owners=int(((a['true_counts']==0)&(fp>0)).sum()),
            countries={c:dict(owners=int((a['countries']==c).sum()),macro_f05=float(s[a['countries']==c].mean()),
                             tp=int(tp[a['countries']==c].sum()),fp=int(fp[a['countries']==c].sum()),fn=int(fn[a['countries']==c].sum()))
                       for c in map(str,np.unique(a['countries']))})
    old_report = json.loads((BASE/'owner_eval/report.json').read_text())['splits']['dev']
    if (abs(metrics['raw100k']['macro_f05']-old_report['baseline_macro_f05'])>1e-12 or
            abs(metrics['owner100k']['macro_f05']-old_report['macro_f05'])>1e-12 or
            sum(x['true_match'] for x in removed)!=old_report['removed_true'] or
            sum(not x['true_match'] for x in removed)!=old_report['removed_false']):
        raise ValueError('Independent replay differs from frozen policy evaluation')
    print(json.dumps({'stage':'replay_verified','metrics':metrics}),flush=True)

    @lru_cache(maxsize=None)
    def view(eid):
        r=by_id[eid]
        return dict(name=folded(r['name']),leet=leet_view(r['name']),
                    house=house_number(r['address'],r['country']),street=street_component(r['address'],r['country']),
                    native=any('\u0900'<=c<='\u0dff' and unicodedata.category(c)[0] in 'LM' for c in r['name']))

    @lru_cache(maxsize=None)
    def patterns(owner,target):
        l,r=by_id[owner],by_id[target];lv,rv=view(owner),view(target)
        low=fuzz.token_set_ratio(lv['name'],rv['name'])<50
        both=bool(lv['house'] and rv['house'])
        deleted=both and len(lv['house'])==len(rv['house'])+1 and single_digit_change(lv['house'],rv['house'])
        street=bool(lv['street'] and rv['street']) and fuzz.token_sort_ratio(lv['street'],rv['street'])>=90
        return dict(target_address_empty=missing(r['address']),target_indic_name=rv['native'],
                    low_romanized_name_similarity=low,
                    low_name_same_house_and_street=low and both and lv['house']==rv['house'] and street,
                    house_digit_deletion=bool(deleted),house_digit_deletion_same_street=bool(deleted and street),
                    small_house_offset=bool(both and 0<abs(int(lv['house'])-int(rv['house']))<=25 and not deleted),
                    leet_ratio_gain_10=fuzz.ratio(lv['leet'],rv['leet'])-fuzz.ratio(lv['name'],rv['name'])>=10)

    errors, categories, overlapping = [], defaultdict(lambda:defaultdict(Counter)), defaultdict(lambda:defaultdict(Counter))
    error_sets = {}
    for name,predictions in [('raw100k',raw),('owner100k',after)]:
        error_sets[name] = {(o,t,'FN') for o in refs for t in gold[o]-predictions[o]} | {(o,t,'FP') for o in refs for t in predictions[o]-gold[o]}
        for o in refs:
            for t in gold[o] | predictions[o]:
                category=legacy_category(by_id[o],by_id[t]);b=categories[category][name]
                label=t in gold[o];selected=t in predictions[o];qi=owner_index[o]
                if label:b['gold']+=1
                kind='TP' if label and selected else 'FN' if label else 'FP'
                b[kind]+=1
                if kind=='FN':
                    b['retrieved_fn' if (o,t) in pair_index else 'retrieval_miss']+=1
                if kind in ('FN','FP'):
                    loss=arrays[name][kind.lower()+'_loss'][qi]/max(1,arrays[name][kind.lower()][qi])/len(refs)
                    b['allocated_score_loss']+=float(loss)
                for pattern,yes in patterns(o,t).items():
                    if yes:
                        ob=overlapping[pattern][name];ob[kind]+=1
                        if label:ob['gold']+=1
                        if kind=='FN':ob['retrieved_fn' if (o,t) in pair_index else 'retrieval_miss']+=1
    for cat,versions in categories.items():
        for name,predictions in [('raw100k',raw),('owner100k',after)]:
            tp,fp=arrays[name]['tp'].copy(),arrays[name]['fp'].copy()
            for owner,target,kind in error_sets[name]:
                if legacy_category(by_id[owner],by_id[target])==cat:
                    if kind=='FN':tp[owner_index[owner]]+=1
                    else:fp[owner_index[owner]]-=1
            versions[name]['oracle_fix_category_gain']=float(score(tp,fp,a['true_counts']).mean()-metrics[name]['macro_f05'])
    # Cross-check the earlier error file without relying on its classifications.
    prior_errors = ROOT/'artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl'
    previous = [json.loads(line) for line in prior_errors.open()]
    expected={(r['reference_id'],r['target_id'],'FP' if r['kind']=='false_positive' else 'FN') for r in previous}
    if expected != error_sets['raw100k']:
        raise ValueError('Independent raw errors disagree with prior 779-error inventory')

    remaining_fp_reasons=Counter();remaining_fp_details=[]
    for owner,target,kind in sorted(error_sets['owner100k']):
        if kind!='FP':continue
        truth=source.execute('SELECT reference_id FROM labels WHERE target_id=?',(target,)).fetchone()
        reason='unowned_decoy' if truth is None else 'other_true_owner'
        extra={}
        if truth:
            rival=db.execute('SELECT p,text FROM claims WHERE target=? AND owner=?',(target,truth[0])).fetchone()
            if rival:
                reason+='__accepted' if rival[0]>=.76 else '__scored_below_acceptance'
                extra=dict(true_owner=truth[0],true_owner_probability=rival[0],true_owner_text_logit=rival[1])
            else:
                selected=db.execute('SELECT 1 FROM chosen WHERE target=? AND owner=?',(target,truth[0])).fetchone()
                reason+='__shortlisted_but_not_forward_retrieved' if selected else '__not_in_shortlist'
                extra=dict(true_owner=truth[0])
        remaining_fp_reasons[reason]+=1
        remaining_fp_details.append(dict(owner=owner,target=target,reason=reason,category=legacy_category(by_id[owner],by_id[target]),
                                        remaining_accepted_rivals=len(accepted.get(target,[])),**extra))
    removed_summary=defaultdict(Counter)
    for item in removed:
        o,t=item['owner'],item['target'];cat=legacy_category(by_id[o],by_id[t]);q=owner_index[o]
        item.update(category=cat,country=by_id[o]['country'],reference=by_id[o],target_record=by_id[t],
                    original_true_count=int(a['true_counts'][q]),owner_score_before=float(arrays['raw100k']['scores'][q]),
                    owner_score_after=float(arrays['owner100k']['scores'][q]),observable_patterns=patterns(o,t))
        removed_summary['removed_true' if item['true_match'] else 'removed_false'][cat]+=1
    lost_by_reason=Counter(x['reason'] for x in removed if x['true_match'])
    changed=[]
    for i,o in enumerate(refs):
        delta=arrays['owner100k']['scores'][i]-arrays['raw100k']['scores'][i]
        if abs(delta)>1e-14:
            changed.append(dict(owner=o,country=str(a['countries'][i]),delta=float(delta),
                before=float(arrays['raw100k']['scores'][i]),after=float(arrays['owner100k']['scores'][i]),
                truth_count=int(a['true_counts'][i]),removed_true=len((raw[o]-after[o])&gold[o]),
                removed_false=len((raw[o]-after[o])-gold[o])))
    # Counterfactual ceilings diagnose priority; they are not deployable scores.
    post=arrays['owner100k'];base_score=metrics['owner100k']['macro_f05']
    retrieved_true=np.bincount(a['query_ids'][a['y']==1],minlength=len(refs))
    miss=a['true_counts']-retrieved_true
    oracles=dict(remove_all_remaining_fp=float(score(post['tp'],np.zeros_like(post['fp']),a['true_counts']).mean()),
                 recover_all_remaining_fn=float(score(a['true_counts'],post['fp'],a['true_counts']).mean()),
                 recover_retrieved_fn_only=float(score(retrieved_true,post['fp'],a['true_counts']).mean()),
                 recover_retrieval_misses_only=float(score(post['tp']+miss,post['fp'],a['true_counts']).mean()),
                 perfect_classifier_on_saved_candidates=float(score(retrieved_true,np.zeros_like(post['fp']),a['true_counts']).mean()))
    # Route base rates include every retrieved candidate in explicitly stated
    # probability bands, unlike an error-only positive/negative count.
    route_stats=defaultdict(lambda:defaultdict(Counter))
    for (o,t),i in pair_index.items():
        prob=float(a['baseline_100000'][i])
        if not .05<=prob<.76:continue
        band='0.05_to_0.2' if prob<.2 else '0.2_to_0.5' if prob<.5 else '0.5_to_0.76'
        p=patterns(o,t)
        for key,yes in p.items():
            if yes:route_stats[key][band]['true' if a['y'][i] else 'false']+=1
    for o,t,k in sorted(error_sets['owner100k']):
        errors.append(dict(owner=o,target=t,kind=k,retrieved=(o,t) in pair_index,
                           category=legacy_category(by_id[o],by_id[t]),patterns=patterns(o,t),
                           probability=float(a['baseline_100000'][pair_index[o,t]]) if (o,t) in pair_index else None,
                           newly_lost_to_assignment=(o,t,k) not in error_sets['raw100k'],reference=by_id[o],target_record=by_id[t]))
    report=dict(complete=True,scope='Existing 3000 US/India development owners; no new tuning, no French/test labels',
        independent_replay_verified=True,ground_truth_rechecked=True,original_779_errors_reproduced=True,
        accepted_rival_threshold=.76,policy=rule,policy_sha256=digest(policy_path),
        data_sha256=digest(data),analysis_code_sha256=digest(Path(__file__)),
        metrics=metrics,categories=dict(categories),overlapping_patterns=dict(overlapping),
        removed_summary=dict(removed_summary),lost_true_by_decision_reason=lost_by_reason,
        remaining_false_positive_reasons=remaining_fp_reasons,
        changes=dict(improved=sum(x['delta']>0 for x in changed),worsened=sum(x['delta']<0 for x in changed),
                     gross_gain=sum(max(x['delta'],0) for x in changed)/len(refs),
                     gross_loss=-sum(min(x['delta'],0) for x in changed)/len(refs)),
        oracle_scores=oracles,oracle_gains={k:v-base_score for k,v in oracles.items()},
        rejected_candidate_routes=dict(route_stats),
        attribution_method='Two-group Shapley FN/FP loss, split within owner in proportion to error counts; additive bookkeeping, not uniquely causal classes',
        category_method='Legacy hierarchical categories reproduced for comparison; native range and first-number heuristic are imprecise, low name similarity does not establish DBA',
        caveats=['Previously reused development; descriptive diagnosis, not fresh validation',
                 'Bounded top3 reverse-discovered rival neighborhoods; some auxiliary rivals occur in first-stage training',
                 'Oracle category gains and overlapping-pattern counts are not additive',
                 'Error-only pattern counts do not establish safe acceptance rules',
                 'Full test country removal counts do not measure country accuracy'],seconds=time.monotonic()-started)
    output.mkdir(parents=True)
    write(output/'report.json',report)
    write(output/'changed_owners.json',sorted(changed,key=lambda x:x['delta']))
    write(output/'removed_claims.json',removed)
    write(output/'remaining_false_positives.json',remaining_fp_details)
    with (output/'remaining_errors.jsonl').open('w') as f:
        for item in errors:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    db.close();source.close()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))

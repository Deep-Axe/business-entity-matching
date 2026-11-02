"""Export exact labeled v6 errors and diagnostic country-level oracle ceilings.

No training, test labeling, external lookup or submission modification. Oracle
repairs use labels only to quantify headroom; never deploy these decisions.
"""
import csv
import json
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from ber.store import connect, lookup
from v3_owner_error_analysis import score
from v3_structural_features import core_name, folded
from v3_transfer_data import ROOT, PILOT, digest, write

OUT = ROOT / 'artifacts/improvement_v3/v6_country_errors_v1'
AUDIT = ROOT / 'artifacts/improvement_v3/v6_band_error_audit_v1'
PEER = Path('work/ce')


def minimum_oracle_repairs(tp, fp, truth, target=.997):
    """Exact multiple-choice knapsack: minimal label-known error repairs.

Each owner may gain true links and lose false links. Enumerate all achievable
scores for a given edit count, then maximize total score at each global count.
This is an optimistic lower bound, not an algorithm with unknown labels.
"""
    tp, fp, truth = (np.asarray(v, dtype=int) for v in (tp, fp, truth))
    base = score(tp, fp, truth)
    need = len(tp) * target - base.sum()
    if need <= 1e-10:
        return dict(minimum_repairs=0, score_at_minimum=float(base.mean()))
    dp = np.array([0.0])
    for t, f, g, old in zip(tp, fp, truth, base):
        miss = g - t
        if miss + f == 0:
            continue
        options = np.full(miss + f + 1, -np.inf)
        for add in range(miss + 1):
            for remove in range(f + 1):
                value = float(score(t + add, f - remove, g)) - old
                options[add + remove] = max(options[add + remove], value)
        combined = np.full(len(dp) + len(options) - 1, -np.inf)
        for cost, gain in enumerate(options):
            combined[cost:cost + len(dp)] = np.maximum(
                combined[cost:cost + len(dp)], dp + gain)
        dp = combined
    found = np.flatnonzero(dp >= need - 1e-10)
    if not len(found):
        return dict(minimum_repairs=None)
    k = int(found[0])
    return dict(minimum_repairs=k, score_at_minimum=float((base.sum()+dp[k])/len(tp)),
                note='Exact optimistic bound with label-known repairs, no newly introduced errors, and no retrieval/ambiguity restrictions.')


def tsv(path, rows, fields):
    with path.open('x', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, '') for key in fields})


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((AUDIT/'completion.json').read_text())
    assert digest(AUDIT/'development_errors.json') == receipt['errors_sha256']
    original = json.loads((AUDIT/'report.json').read_text())
    for p in (PEER/'dev.pkl', PILOT/'data/dev.npz'):
        assert digest(p) == original['inputs_sha256'][str(p)]
    errors = json.loads((AUDIT/'development_errors.json').read_text())
    with (PEER/'dev.pkl').open('rb') as fh:
        rows = pickle.load(fh)
    with np.load(AUDIT/'decisions.npz') as z:
        keep = z['band_cC_keep'][:len(rows)].copy()
    with np.load(PILOT/'data/dev.npz') as z:
        a = {k:z[k] for k in ('reference_ids','query_ids','true_counts','countries','y')}
    refs = list(map(str, a['reference_ids'])); oi = {r:i for i,r in enumerate(refs)}
    predicted = {r:set() for r in refs}
    routed_positives = np.zeros(len(refs), int)
    for row, accepted in zip(rows, keep):
        routed_positives[oi[row[0]]] += row[2]
        if accepted:
            predicted[row[0]].add(row[1])
    db = connect(ROOT/'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-8192')
    gold = {r:{x[0] for x in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(r,))} for r in refs}
    truth = np.array([len(gold[r]) for r in refs])
    assert np.array_equal(truth, a['true_counts'])
    tp = np.array([len(predicted[r]&gold[r]) for r in refs])
    fp = np.array([len(predicted[r]-gold[r]) for r in refs])
    actual_errors = {(r,t,'FN') for r in refs for t in gold[r]-predicted[r]}
    actual_errors |= {(r,t,'FP') for r in refs for t in predicted[r]-gold[r]}
    assert actual_errors == {(e['owner'],e['target'],e['kind']) for e in errors}
    current = score(tp,fp,truth)
    assert abs(current.mean()-original['metrics']['dev']['band_cC']['macro_f05']) < 1e-12
    retrieved = np.bincount(a['query_ids'][a['y']==1], minlength=len(refs))

    # Relevant exact/core families only, discovered from the full S1 gallery.
    # This is not exhaustive fuzzy rivalry and cannot prove unidentifiability.
    core = lru_cache(maxsize=10000)(core_name)
    wanted = {(e['country'],core(e['reference']['name'])) for e in errors}
    families = defaultdict(list)
    for scanned, row in enumerate(db.execute('SELECT entity_id,country,name,address FROM anchors'),1):
        key = (row['country'],core(row['name']))
        if key in wanted:
            families[key].append(dict(row))
        if scanned % 500000 == 0:
            write(OUT/'progress.json', dict(stage='scanning_name_families',scanned=scanned))
    assert scanned == 2206821
    cases=[]; groups=defaultdict(list)
    for e in errors:
        i=oi[e['owner']]; ref=e['reference']; target=e['target_record']
        rivals=[r for r in families[(e['country'],core(ref['name']))] if r['entity_id']!=e['owner']]
        exact=[r for r in rivals if r['name']==ref['name']]
        norm=[r for r in rivals if folded(r['name'])==folded(ref['name'])]
        empty='empty_target_address' in e['patterns']
        bucket=('empty_exact_name_twin' if exact else 'empty_core_name_twin' if rivals else 'empty_no_core_name_twin') if empty else 'nonempty_address'
        owner_rows=list(db.execute('SELECT reference_id FROM labels WHERE target_id=?',(e['target'],)))
        assert len(owner_rows)<=1
        true_owner=owner_rows[0][0] if owner_rows else ''
        wrong_owner=lookup(db,'anchors','entity_id',[true_owner]).get(true_owner) if true_owner and e['kind']=='FP' else None
        changed=float(score(tp[i]+(e['kind']=='FN'),fp[i]-(e['kind']=='FP'),truth[i]))
        country_n=int((a['countries']==e['country']).sum())
        intervention={
            'retrieval_missing':'retrieve_pair_then_classify; do not label-inject candidates',
            'neural_route_excluded':'validate a rescue route and score its negatives',
            'stacker_classification':'hard-positive/hard-negative training and calibrated matching',
            'owner_assignment_removed':'inspect competitor/abstention context; wrong winner not established',
        }[e['stage']]
        if empty and exact:
            intervention+='; exact-name ambiguity: do not force a winner without additional provided-record evidence'
        if e['kind']=='FP':
            intervention+='; '+('wrong business owner' if true_owner else 'unmatched decoy')
        case=dict(country=e['country'],kind=e['kind'],stage=e['stage'],bucket=bucket,
            source1_entity_id=e['owner'],target_entity_id=e['target'],
            source1_name=ref['name'],source1_address=ref['address'],
            target_name=target['name'],target_address=target['address'],
            patterns=','.join(e['patterns']),score=e['v6_score'],
            exact_name_other_s1s=len(exact),folded_name_other_s1s=len(norm),core_name_other_s1s=len(rivals),
            true_target_owner=true_owner, true_owner_name=wrong_owner['name'] if wrong_owner else '',
            true_owner_address=wrong_owner['address'] if wrong_owner else '',
            owner_true_links=int(truth[i]),owner_tp=int(tp[i]),owner_fp=int(fp[i]),owner_fn=int(truth[i]-tp[i]),
            owner_f05=float(current[i]),individual_oracle_gain=changed-float(current[i]),
            combined_macro_oracle_gain=(changed-float(current[i]))/len(refs),
            country_macro_oracle_gain=(changed-float(current[i]))/country_n,
            possible_experiment=intervention)
        cases.append(case)
        groups['stage/'+e['stage']].append((i,e['kind']))
        groups['bucket/'+bucket].append((i,e['kind']))
    db.close()
    report=dict(complete=True,scope='Labeled US/India development; not hidden-test errors',
        targets=.997,countries={},groups={},full_s1_scan=scanned)
    for country in ('all','US','India'):
        mask=np.ones(len(refs),bool) if country=='all' else a['countries']==country
        stats=dict(owners=int(mask.sum()),tp=int(tp[mask].sum()),fp=int(fp[mask].sum()),fn=int((truth-tp)[mask].sum()),
            current=float(current[mask].mean()),remove_all_fp=float(score(tp,np.zeros_like(fp),truth)[mask].mean()),
            perfect_current_neural_candidates=float(score(routed_positives,np.zeros_like(fp),truth)[mask].mean()),
            perfect_current_retrieval=float(score(retrieved,np.zeros_like(fp),truth)[mask].mean()),
            minimum_error_repairs_to_0997=minimum_oracle_repairs(tp[mask],fp[mask],truth[mask]),
            minimum_retrieval_repairs_after_perfect_classifier=minimum_oracle_repairs(retrieved[mask],np.zeros(int(mask.sum()),int),truth[mask]))
        ce=[c for c in cases if country=='all' or c['country']==country]
        stats['error_stage_counts']={key:dict(Counter(c['kind'] for c in ce if c['stage']==key)) for key in sorted({c['stage'] for c in ce})}
        stats['empty_address_name_groups']={key:dict(Counter(c['kind'] for c in ce if c['bucket']==key)) for key in sorted({c['bucket'] for c in ce})}
        report['countries'][country]=stats
        report['groups'][country]={}
        for name, changes in groups.items():
            t=tp.copy();f=fp.copy()
            for idx,kind in changes:
                if kind=='FN':t[idx]+=1
                else:f[idx]-=1
            improved=score(t,f,truth)
            report['groups'][country][name]=dict(oracle_score=float(improved[mask].mean()),
                oracle_gain=float((improved-current)[mask].mean()))
        # A useful conditional ceiling if all exact-name-twin missing-address FN
        # are left unmatched, but every other mistake is magically repaired.
        unresolved=np.zeros(len(refs),int)
        for case in cases:
            if case['kind']=='FN' and case['bucket']=='empty_exact_name_twin':
                unresolved[oi[case['source1_entity_id']]]+=1
        stats['oracle_everything_except_exact_name_twin_fn']=float(score(truth-unresolved,np.zeros_like(fp),truth)[mask].mean())
        if country!='all':
            ordered=sorted(ce,key=lambda c:(-c['country_macro_oracle_gain'],c['source1_entity_id'],c['target_entity_id']))
            tsv(OUT/f'{country}_errors.tsv',ordered,list(cases[0]))
            owner_rows=[]
            for i,r in enumerate(refs):
                if mask[i] and current[i]<1:
                    owner_rows.append(dict(source1_entity_id=r,f05=float(current[i]),
                        gold_ids=','.join(sorted(gold[r])),predicted_ids=','.join(sorted(predicted[r])),
                        missing_ids=','.join(sorted(gold[r]-predicted[r])),false_ids=','.join(sorted(predicted[r]-gold[r]))))
            tsv(OUT/f'{country}_owners.tsv',sorted(owner_rows,key=lambda r:r['f05']),
                ['source1_entity_id','f05','gold_ids','predicted_ids','missing_ids','false_ids'])
    report['limitations']=[
        'Reused owner-grouped OOF development; no fresh generalization or test-country estimate.',
        'Pattern groups overlap stage groups. Per-pair gains do not add under macro F0.5.',
        'Core-name and exact-name collisions are observable proxies, not a proof of impossibility.',
        'Owner removals may be abstentions; these are not eight proven wrong-owner assignments.',
        'Oracle repairs have perfect labels and introduce no new errors. They are not deployable rules.',
        'Diagnostic record IDs and dev labels must not become training rules or submission patches.']
    write(OUT/'report.json',report)
    write(OUT/'completion.json',dict(complete=True,report_sha256=digest(OUT/'report.json'),
        exported={p.name:digest(p) for p in OUT.glob('*.tsv')},
        input_errors_sha256=digest(AUDIT/'development_errors.json'),
        input_decisions_sha256=digest(AUDIT/'decisions.npz'),code_sha256=digest(__file__)))
    print(json.dumps(report['countries'],indent=2),flush=True)


if __name__=='__main__':
    run()

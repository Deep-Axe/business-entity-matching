"""Bounded read-only data audit; never changes model, retrieval, or submissions.

Profiles all candidates in a reproducible 4k-owner sample from the extra 100k
training owners, plus all French S1 names and sampled saved French candidates.
French output diagnostics have no ground truth and are never accuracy claims.
"""
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from catboost import CatBoostClassifier
from rapidfuzz import fuzz

from ber.store import connect, lookup, record_row
from v2_cpu_features import Weighting
from v2_run_pipeline import enhanced_matrix
from v3_structural_features import (folded, core_name, address_view, house_number,
                                    street_component, single_digit_change, LEGAL)
from v3_owner_error_analysis import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/improvement_v3/fundamental_audit_v1'


def save(name, data):
    p = OUT / name
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def view(row):
    name, address, country = row['name'], row['address'], row['country']
    n = folded(name)
    return dict(name=n, core=core_name(name), address=address_view(address, country),
                house=house_number(address, country), street=street_component(address, country),
                legal=' '.join(w for w in n.split() if w in LEGAL), country=country)


def relation(a, b):
    if not a or not b: return 'unknown'
    if a == b: return 'equal'
    if single_digit_change(a, b):
        return 'target_digit_deleted' if len(a) > len(b) else 'target_digit_inserted'
    delta = abs(int(a) - int(b))
    hamming = sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else None
    if delta <= 30:
        return 'small_shift_one_digit' if hamming == 1 else 'small_shift_multiple_digits'
    if hamming == 1: return 'large_shift_one_digit'
    return 'other_conflict'


def evidence(l, r):
    street = fuzz.token_sort_ratio(l['street'], r['street']) if l['street'] and r['street'] else 0.
    name = fuzz.token_set_ratio(l['name'], r['name']) if l['name'] and r['name'] else 0.
    rel = relation(l['house'], r['house'])
    return rel, street, name, bool(l['core'] and l['core'] == r['core']), bool(
        l['legal'] and r['legal'] and l['legal'] != r['legal'])


def record_stats(counter, label, accepted):
    counter['pairs'] += 1
    counter['positive' if label else 'negative'] += 1
    counter[('TP' if label else 'FP') if accepted else ('FN' if label else 'TN')] += 1


def audit_fresh():
    if (OUT / 'fresh_labeled.json').exists(): return
    source = ROOT / 'artifacts/improvement_v2/training/gate0_train_200k_v1/shards'
    shards = random.Random(20260926).sample(list(range(500, 1000)), 20)
    db = connect(ROOT / 'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-16384')
    weighting = Weighting(ROOT / 'artifacts/train_index_v2')
    model = CatBoostClassifier(thread_count=1)
    model.load_model(str(ROOT / 'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm'))
    totals, groups, samples = Counter(), defaultdict(Counter), defaultdict(list)
    owner_empty = []; rule_tp = []; rule_fp = []; raw_tp = []; raw_fp = []; truths = []
    started = time.monotonic()
    for shard in sorted(shards):
        path = source / f'{shard:06d}' / 'features.npz'
        with np.load(path, allow_pickle=False) as z: a = {k: z[k] for k in z.files}
        anchors = lookup(db, 'anchors', 'entity_id', a['reference_ids'].tolist())
        targets = lookup(db, 'records', 'entity_id', np.unique(a['target_ids']).tolist())
        rows = [record_row(anchors[str(e)]) for e in a['reference_ids']]
        X = enhanced_matrix(a, rows, targets, weighting)
        p = model.predict_proba(X, thread_count=1)[:, 1]
        left = {k: view(v) for k, v in anchors.items()}
        right = {k: view(v) for k, v in targets.items()}
        accepted = p >= .76; veto = np.zeros(len(p), bool)
        totals.update(owners=len(a['reference_ids']), pairs=len(p), positives=int(a['y'].sum()))
        for i, (q, eid) in enumerate(zip(a['query_ids'], a['target_ids'])):
            owner = str(a['reference_ids'][q]); target = str(eid)
            l, r = left[owner], right[target]
            rel, street, name, core_equal, legal_swap = evidence(l, r)
            label, chosen = bool(a['y'][i]), bool(accepted[i])
            keys = [f'number/{rel}', f'country/{l["country"]}/number/{rel}']
            if street >= 90:
                keys.append(f'same_street/number/{rel}')
                if name >= 85: keys.append(f'same_street_high_name/number/{rel}')
                if core_equal and legal_swap: keys.append(f'same_street_core_legal_swap/number/{rel}')
            if not r['address']: keys.append('target_address_empty')
            if rel == 'equal' and street >= 90 and name < 50: keys.append('low_name_same_house_street')
            if r['address'] and rel == 'unknown': keys.append('number_parser_unknown_nonempty')
            if street >= 90 and rel == 'small_shift_multiple_digits':
                veto[i] = True
                if label or chosen:
                    key = 'small_multidigit_true' if label else 'small_multidigit_false_accepted'
                    if len(samples[key]) < 18:
                        samples[key].append(dict(owner=owner, target=target, probability=float(p[i]),
                            reference=dict(anchors[owner]), target_record=dict(targets[target])))
            for key in keys: record_stats(groups[key], label, chosen)
        q = a['query_ids']; n = len(a['reference_ids']); y = a['y'].astype(bool)
        tp = np.bincount(q, weights=accepted & y, minlength=n)
        fp = np.bincount(q, weights=accepted & ~y, minlength=n)
        rtp = np.bincount(q, weights=accepted & ~veto & y, minlength=n)
        rfp = np.bincount(q, weights=accepted & ~veto & ~y, minlength=n)
        raw_tp.extend(tp); raw_fp.extend(fp); rule_tp.extend(rtp); rule_fp.extend(rfp); truths.extend(a['true_counts'])
        for qi in np.flatnonzero(tp + fp == 0):
            ids = np.flatnonzero(q == qi)
            order = ids[np.argsort(p[ids])[::-1][:3]]
            owner_empty.append(dict(owner=str(a['reference_ids'][qi]), country=str(a['countries'][qi]),
                true_count=int(a['true_counts'][qi]), retrieved_positives=int(y[ids].sum()),
                top=[dict(target=str(a['target_ids'][i]), p=float(p[i]), true=bool(y[i])) for i in order]))
        print(json.dumps({'stage': 'fresh_labeled', 'owners': totals['owners'],
                          'seconds': time.monotonic() - started}), flush=True)
    report = dict(complete=True, scope='4k owners from training positions 100k–200k; out of sample for 100k classifier, not 200k. Existing training lexicon may have seen these labels.',
        sample_shards=sorted(shards), totals=dict(totals), groups=dict(groups),
        raw100k_macro=float(score(raw_tp, raw_fp, truths).mean()),
        diagnostic_veto=dict(description='Reject parsed multi-digit house shifts <=30 with street similarity >=90; diagnostic, NOT promoted.',
                            macro=float(score(rule_tp, rule_fp, truths).mean()),
                            removed_true=int(sum(raw_tp) - sum(rule_tp)), removed_false=int(sum(raw_fp) - sum(rule_fp))),
        predicted_empty=owner_empty, examples=dict(samples), seconds=time.monotonic() - started)
    save('fresh_labeled.json', report)
    db.close()


def audit_french():
    if (OUT / 'french.json').exists(): return
    started = time.monotonic()
    db = connect(ROOT / 'artifacts/test_store_v2'); db.execute('PRAGMA cache_size=-16384')
    anchors, groups, exact, addresses = {}, defaultdict(list), Counter(), Counter()
    for row in db.execute("SELECT entity_id,country,name,address FROM anchors WHERE country='France'"):
        eid = row['entity_id']; v = view(row)
        anchors[eid] = (dict(row), v)
        if v['core']: groups[v['core']].append(eid)
        exact[v['name']] += 1
        addresses[v['address']] += 1
    sizes = np.array([len(groups[v['core']]) if v['core'] else 0 for _, v in anchors.values()])
    profile = dict(owners=len(anchors), unique_core_names=len(groups),
        owners_sharing_core_name=int((sizes > 1).sum()), owners_in_core_groups_10plus=int((sizes >= 10).sum()),
        largest_core_group=int(sizes.max()),
        owners_sharing_exact_name=sum(n for n in exact.values() if n > 1),
        owners_sharing_exact_normalized_address=sum(n for n in addresses.values() if n > 1),
        house_parser_known=sum(bool(v['house']) for _, v in anchors.values()),
        top_core_groups=[{'core': k, 'owners': len(ids), 'examples': [anchors[e][0] for e in ids[:4]]}
                         for k, ids in sorted(groups.items(), key=lambda x: -len(x[1]))[:12]])
    save('french_s1_profile.json', profile)
    print(json.dumps({'stage': 'french_s1_profile', **{k:v for k,v in profile.items() if k != 'top_core_groups'}}), flush=True)
    queue = json.loads((ROOT / 'artifacts/improvement_v2/inference/submission_controller_v1/queue.json').read_text())
    paths = sorted(s for item in queue['complete'].values() for s in (Path(item['directory'])/'shards').iterdir() if s.is_dir())
    selected = random.Random(20260926).sample(paths, 96)
    chosen_owners = set(); rows = []
    for path in selected:
        with np.load(path/'pair_ids.npz', allow_pickle=False) as z:
            ids = z['reference_ids']; q = z['query_ids']; tids = z['target_ids']; countries = z['countries']
            p = np.load(path/'probabilities.npy')
            for i in np.flatnonzero(countries == 'France'): chosen_owners.add(str(ids[i]))
            mask = countries[q] == 'France'
            rows.extend((str(ids[q[i]]), str(tids[i]), float(p[i])) for i in np.flatnonzero(mask))
    selected_matches = {}
    submission = ROOT / 'artifacts/duf-fel_full_output_v3_100k_owner_v1/matching_results.tsv'
    with submission.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['source1_entity_id'] in chosen_owners:
                selected_matches[row['source1_entity_id']] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(selected_matches) != chosen_owners: raise ValueError('Actual submission sample coverage mismatch')
    targets = lookup(db, 'records', 'entity_id', list({t for _, t, _ in rows}))
    views = {eid: view(r) for eid, r in targets.items()}

    @lru_cache(maxsize=150000)
    def competitors(core, target):
        rv = views[target]
        members = groups.get(core, []) if core else []
        # Top two distinct owners permit leave-one-owner-out comparisons.
        ranked = sorted(((fuzz.token_sort_ratio(anchors[e][1]['address'], rv['address'])
                          if rv['address'] else 0., e) for e in members), reverse=True)[:2]
        exact_location = []
        if rv['house'] and rv['street']:
            for eid in members:
                v = anchors[eid][1]
                if v['house'] == rv['house'] and v['street'] and fuzz.token_sort_ratio(v['street'], rv['street']) >= 95:
                    exact_location.append(eid)
        return ranked, exact_location

    counts = defaultdict(Counter); examples = defaultdict(list)
    for i, (owner, target, p) in enumerate(rows):
        raw, l = anchors[owner]; r = views[target]
        rel, street, name, core_equal, legal_swap = evidence(l, r)
        kept = target in selected_matches[owner]
        band = 'kept_owner_submission' if kept else 'removed_by_owner' if p >= .76 else 'uncertain_030_076' if p >= .3 else 'low_below_030'
        b = counts[band]; b['pairs'] += 1; b['number/'+rel] += 1
        shared = bool(l['core']) and len(groups[l['core']]) > 1
        if shared: b['shared_core_name'] += 1
        if not r['address']: b['empty_address'] += 1
        if shared and not r['address']: b['shared_core_empty_address'] += 1
        if street >= 90:
            b['same_street/number/'+rel] += 1
            if core_equal and legal_swap: b['same_street_core_legal_swap/number/'+rel] += 1
        # Restrict expensive global text comparisons to accepted/uncertain pairs.
        if p >= .3:
            ranked, exact_ids = competitors(l['core'], target)
            own = fuzz.token_sort_ratio(l['address'], r['address']) if r['address'] else 0.
            rivals = [(s, e) for s, e in ranked if e != owner]
            better = bool(rivals and rivals[0][0] > own + 5)
            alternative = next((e for e in exact_ids if e != owner), None)
            own_exact = l['house'] and l['house'] == r['house'] and street >= 95
            if better: b['same_core_other_address_5pts_better'] += 1
            if alternative and not own_exact:
                b['same_core_other_owner_exact_house_street_not_claimant'] += 1
                if kept and len(examples['kept_with_exact_location_alternative']) < 12:
                    examples['kept_with_exact_location_alternative'].append(dict(p=p, claimant=raw,
                        target=dict(targets[target]), alternative=anchors[alternative][0]))
            if kept and rel == 'small_shift_multiple_digits' and street >= 90 and len(examples['kept_multidigit_offset']) < 12:
                examples['kept_multidigit_offset'].append(dict(p=p, claimant=raw, target=dict(targets[target]),
                    legal_swap=legal_swap, alternative=anchors[alternative][0] if alternative else None))
        if (i+1) % 25000 == 0: print(json.dumps({'stage':'french_pairs','pairs':i+1,'total':len(rows)}), flush=True)
    report = dict(complete=True, labels_available=False, sample_shards=[str(x) for x in selected],
        sampling_frame_includes_all_partitions_and_local_outputs=True, sampled_owners=len(chosen_owners),
        sampled_pairs=len(rows), sampled_kept_matches=sum(map(len, selected_matches.values())),
        full_s1_profile=profile, counts=dict(counts), examples=dict(examples),
        caveat='Structural label-free flags, not errors or calibrated probabilities. Sample accepted status comes from the actual 0.958 owner-assigned TSV.',
        seconds=time.monotonic()-started)
    save('french.json', report)
    db.close()


def audit_existing_owner_errors():
    root = ROOT/'artifacts/improvement_v3/structural_v1/owner_error_analysis_v1'
    errors = [json.loads(x) for x in (root/'remaining_errors.jsonl').open()]
    by_owner = defaultdict(Counter); sources = defaultdict(Counter); numeric = defaultdict(Counter)
    for row in errors:
        by_owner[row['owner']][row['kind']] += 1
        sources[row['target'].split('-')[0]][row['kind']] += 1
        l, r = view(row['reference']), view(row['target_record'])
        rel, street, name, core, legal = evidence(l, r)
        numeric[rel][row['kind']] += 1
        if row['kind']=='FN': numeric[rel]['retrieved' if row['retrieved'] else 'not_retrieved'] += 1
    save('existing_owner_errors.json', dict(sources=dict(sources), number_relations=dict(numeric),
        owners_with_errors=len(by_owner), total_errors=len(errors)))


def audit_french_collisions():
    if (OUT/'french_identity_collisions.json').exists(): return
    db=connect(ROOT/'artifacts/test_store_v2');db.execute('PRAGMA cache_size=-8192')
    counts={k:Counter() for k in ('exact_raw_name_address','normalized_name_address',
                                 'normalized_name_address_token_order','core_name_address_token_order')}
    examples=defaultdict(list)
    for row in db.execute("SELECT entity_id,name,address FROM anchors WHERE country='France'"):
        raw=(row['name'],row['address'])
        n=folded(row['name']);addr=address_view(row['address'],'France')
        sorted_address=' '.join(sorted(addr.split()))
        keys={'exact_raw_name_address':raw,'normalized_name_address':(n,addr),
              'normalized_name_address_token_order':(n,sorted_address),
              'core_name_address_token_order':(core_name(row['name']),sorted_address)}
        for kind,key in keys.items():
            counts[kind][key]+=1
            if counts[kind][key]==2 and len(examples[kind])<6:
                examples[kind].append({'key':key,'second_owner':row['entity_id']})
    result={k:dict(duplicate_groups=sum(v>1 for v in c.values()),
                   owners_in_duplicate_groups=sum(v for v in c.values() if v>1),
                   excess_owners=sum(v-1 for v in c.values()),largest_group=max(c.values()))
            for k,c in counts.items()}
    save('french_identity_collisions.json',dict(counts=result,examples=dict(examples),
        caveat='Normalization-induced collisions do not establish identical real businesses or known test labels.'))
    print(json.dumps(result),flush=True);db.close()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    audit_existing_owner_errors()
    audit_fresh()
    gc.collect()
    audit_french()
    audit_french_collisions()
    print(json.dumps({'complete':True,'output':str(OUT)}), flush=True)

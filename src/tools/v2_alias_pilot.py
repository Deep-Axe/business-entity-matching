"""Bounded, prediction-seeded alias retrieval; no gold enters candidate decisions.

Existing frozen features/model/submission are read-only. New pairs have *zero*
original-query retrieval features: alias scores are recorded as provenance, never
misrepresented as original-query BM25 evidence. The full candidate union is used
for context features. This is a frozen-matcher screen, not a trained alias model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.context_features import model_matrix
from ber.data import read_tsv, write_json
from ber.deadline_retrieval import retrievers
from ber.disk_index import TargetStore
from ber.expansion import CHANNEL
from ber.features import FEATURE_NAMES, pair_features
from ber.metrics import entity_f05
from ber.shards import file_hash, gold_for
from ber.store import lookup, record_row
from ber.text import Record


def eligible_seed(features, probability, conflict=False, minimum=.995):
    """Use prediction/evidence only; labels intentionally absent from this API."""
    f = dict(zip(FEATURE_NAMES, features))
    return bool(not conflict and probability >= minimum
                and max(f['name_ratio'], f['ascii_name_ratio']) >= .8
                and f['address_overlap'] >= .9 and f['number_jaccard'] >= .5
                and not f['disjoint_numbers']
                and not f['reference_address_missing'] and not f['target_address_missing'])


def pick_seeds(items, records, reference, maximum=3):
    """Highest-confidence distinct alternate representations, never original text."""
    seen = {(reference.name, reference.address)}
    result = []
    for target, probability in sorted(items, key=lambda item: (-item[1], item[0])):
        record = records[target]
        key = record.name, record.address
        if key in seen or record.country != reference.country:
            continue
        seen.add(key)
        result.append((target, probability))
        if len(result) == maximum:
            break
    return result


def subset_arrays(arrays, q):
    mask = arrays['query_ids'] == q
    return {'X': arrays['X'][mask].copy(),
            'query_ids': np.zeros(mask.sum(), dtype=np.int32),
            'reference_ids': arrays['reference_ids'][q:q + 1].copy(),
            'target_ids': arrays['target_ids'][mask].copy()}


def union_arrays(original, reference, new_targets):
    """Keep complete baseline neighborhood; append label-free original pair evidence."""
    matrix = [pair_features(reference, target, {}) for target in new_targets]
    return {'X': np.concatenate((original['X'], np.asarray(matrix, np.float32).reshape(-1, len(FEATURE_NAMES)))),
            'query_ids': np.zeros(len(original['X']) + len(matrix), np.int32),
            'reference_ids': original['reference_ids'],
            'target_ids': np.concatenate((original['target_ids'], [r.entity_id for r in new_targets]))}


def summarize(rows, variant):
    true_positive = false_positive = false_negative = 0
    scores = []
    singleton_errors = 0
    for row in rows:
        truth, prediction = set(row['gold']), set(row[variant])
        true_positive += len(truth & prediction)
        false_positive += len(prediction - truth)
        false_negative += len(truth - prediction)
        scores.append(entity_f05(truth, prediction))
        singleton_errors += int(not truth and bool(prediction))
    return dict(references=len(rows), macro_f05=float(np.mean(scores)), tp=true_positive,
                fp=false_positive, fn=false_negative,
                pair_precision=true_positive / max(1, true_positive + false_positive),
                pair_recall=true_positive / max(1, true_positive + false_negative),
                singleton_false_merges=singleton_errors)


def run(root, output, count=200, minimum=.995):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen = json.loads((root / 'artifacts/deadline_frozen_v1.json').read_text())
    model = CatBoostClassifier(thread_count=1)
    model.load_model(frozen['model'])
    assert file_hash(frozen['model']) == frozen['model_sha256']
    threshold = frozen['threshold']
    caches = {fold: root / f'artifacts/deadline_features_full_{fold}_v1' for fold in ('tune', 'dev')}
    configs = {fold: json.loads((path / 'config.json').read_text()) for fold, path in caches.items()}
    selected = {}
    selected_features = {}
    claims = defaultdict(set)
    for fold, path in caches.items():
        rows = list(read_tsv(configs[fold]['references']))
        selected[fold] = {r['entity_id']: r for r in sorted(rows,
            key=lambda r: hashlib.sha256(('alias-v2:' + r['entity_id']).encode()).hexdigest())[:count]}
        for shard in sorted((path / 'shards').iterdir()):
            with np.load(shard / 'features.npz', allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            probability = model.predict_proba(model_matrix(arrays, model.feature_names_), thread_count=1)[:, 1]
            for q, target, p in zip(arrays['query_ids'], arrays['target_ids'], probability):
                if p >= threshold:
                    claims[str(target)].add(str(arrays['reference_ids'][q]))
            for q, ref in enumerate(arrays['reference_ids']):
                if ref in selected[fold]:
                    selected_features[str(ref)] = (subset_arrays(arrays, q), probability[arrays['query_ids'] == q].copy())
        print(json.dumps(dict(stage='loaded_baseline', fold=fold, selected=len(selected[fold]))), flush=True)
    conflict_targets = {target for target, owners in claims.items() if len(owners) > 1}
    targets = TargetStore(configs['tune']['store'], cache_size=256)
    base, extra = retrievers(configs['tune']['index'], frozen['lexicon'], 1, store=configs['tune']['store'])
    results = {}
    for fold in ('tune', 'dev'):
        fold_started = time.perf_counter()
        rows = []
        for position, (eid, row) in enumerate(selected[fold].items()):
            reference = Record.from_row(row)
            arrays, probability = selected_features[eid]
            original_ids = set(map(str, arrays['target_ids']))
            baseline = {str(target) for target, p in zip(arrays['target_ids'], probability) if p >= threshold}
            eligible = [(str(target), float(p)) for x, target, p in zip(arrays['X'], arrays['target_ids'], probability)
                        if eligible_seed(x, p, str(target) in conflict_targets, minimum)]
            seed_records = {target: Record.from_row(record_row(r)) for target, r in
                            lookup(targets.db, 'records', 'entity_id', [item[0] for item in eligible]).items()}
            seeds = pick_seeds(eligible, seed_records, reference)
            additions, provenance = {}, defaultdict(list)
            retrieval_seconds = 0.
            for seed_id, p in seeds:
                stamp = time.perf_counter()
                candidates = dict(base.candidates(seed_records[seed_id], 20))
                for row_id, hit in extra.candidates(seed_records[seed_id], 20):
                    candidates.setdefault(row_id, {})[CHANNEL] = hit
                retrieved = targets.get_many(list(candidates))
                retrieval_seconds += time.perf_counter() - stamp
                ordered = sorted(candidates, key=lambda rid: (-max(v[1] for v in candidates[rid].values()),
                    -max(v[0] for v in candidates[rid].values()), retrieved[rid].entity_id))[:20]
                for rid in ordered:
                    target = retrieved[rid]
                    assert target.country == reference.country
                    if target.entity_id in original_ids:
                        continue
                    additions[target.entity_id] = target
                    provenance[target.entity_id].append(dict(seed_id=seed_id, seed_probability=p, pass_number=1,
                        channels={channel: list(hit) for channel, hit in candidates[rid].items()},
                        conflict_decision='no competing predicted owner among 6000 cached references'))
            new_ids = sorted(additions)
            union = union_arrays(arrays, reference, [additions[target] for target in new_ids])
            rescored = model.predict_proba(model_matrix(union, model.feature_names_), thread_count=1)[:, 1]
            union_prediction = {str(target) for target, p in zip(union['target_ids'], rescored) if p >= threshold}
            new_prediction = union_prediction & set(new_ids)
            # Additive variant freezes old decisions; union variant reclassifies all.
            rows.append(dict(reference_id=eid, country=reference.country, original_candidates=sorted(original_ids),
                baseline=sorted(baseline), additive=sorted(baseline | new_prediction),
                union_rescored=sorted(union_prediction), seeds=[dict(target_id=target, probability=p) for target, p in seeds],
                new_candidates=new_ids, new_probabilities={target: float(p) for target, p in zip(new_ids, rescored[len(arrays['X']):])},
                provenance=dict(provenance), retrieval_seconds=retrieval_seconds))
            if (position + 1) % 20 == 0:
                print(json.dumps(dict(stage='alias_retrieval', fold=fold, complete=position + 1,
                    seconds=time.perf_counter() - fold_started,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)), flush=True)
        # Labels are fetched only AFTER all prediction/retrieval decisions for fold.
        gold = gold_for(targets.db, [row['reference_id'] for row in rows])
        for row in rows:
            row['gold'] = sorted(gold[row['reference_id']])
        with (output / f'{fold}_rows.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        added = sum(len(row['new_candidates']) for row in rows)
        new_truth = sum(len(set(row['new_candidates']) & set(row['gold'])) for row in rows)
        accepted_tp = sum(len((set(row['additive']) - set(row['baseline'])) & set(row['gold'])) for row in rows)
        accepted_fp = sum(len((set(row['additive']) - set(row['baseline'])) - set(row['gold'])) for row in rows)
        results[fold] = dict(variants={v: summarize(rows, v) for v in ('baseline', 'additive', 'union_rescored')},
            added_candidates=added, newly_retrieved_true_pairs=new_truth,
            new_true_pairs_per_1000_candidates=1000 * new_truth / max(1, added),
            accepted_addition_tp=accepted_tp, accepted_addition_fp=accepted_fp,
            accepted_addition_precision=accepted_tp / max(1, accepted_tp + accepted_fp),
            seeds=sum(len(row['seeds']) for row in rows),
            false_seeds=sum(seed['target_id'] not in row['gold'] for row in rows for seed in row['seeds']),
            seeded_references=sum(bool(row['seeds']) for row in rows),
            reference_ids=[row['reference_id'] for row in rows],
            alias_retrieval_seconds=sum(row['retrieval_seconds'] for row in rows),
            wall_seconds=time.perf_counter() - fold_started)
        write_json(output / f'{fold}_report.json', results[fold])
        print(json.dumps(dict(stage='fold_complete', fold=fold, report=results[fold]), default=str), flush=True)
    report = dict(results=results, threshold=threshold, seed_minimum=minimum, max_seeds=3,
        top_hits_per_seed=20, max_extra_passes=1, original_retrieval_features_for_new_pairs='all zero',
        context='full original plus new candidate union', model_sha256=frozen['model_sha256'],
        script_sha256=file_hash(__file__), external_data_used=False, inference_used_gold=False,
        conflict_scope='6000 cached tune/dev predictions only, not all S1 owners',
        conflict_targets_in_scope=len(conflict_targets),
        limitations=['Small tune/dev screen, not final audit; no French labeled examples.',
            'Frozen model was not trained on alias-retrieved candidates or their provenance.',
            'Seed distinctiveness/owner margin not globally calibrated; conflicting owners outside cohort are invisible.',
            'Top20 union across alias channels may discard useful lower-ranked hits.'],
        total_seconds=time.perf_counter() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(output / 'report.json', report)
    targets.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--minimum', type=float, default=.995)
    run(**vars(parser.parse_args()))

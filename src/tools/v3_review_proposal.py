"""Read-only revalidation of the September 26 proposal; no release changes.

New reports only. Uses already exploratory tune/dev, not the reserved audit.
Partial test conflicts are structural diagnostics, never accuracy estimates.
"""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from pathlib import Path
import re
import random
import sqlite3
import time

import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz

from v3_transfer_data import ROOT, PILOT, digest, load, metric, owner_scores, write

OUT = ROOT / 'artifacts/improvement_v3/proposal_review_v1'
DATA = ROOT / '6ab10eb3b23ba_student_resource/student_resource/dataset'


def database(split):
    db = sqlite3.connect(f'file:{ROOT}/artifacts/{split}_store_v2/records.sqlite3?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-4096')
    db.execute('PRAGMA query_only=ON')
    return db


def numbers(address):
    return [str(int(s)) for s in re.findall(r'\d+', anyascii(address))]


def subsequence(short, long):
    it = iter(long)
    return all(c in it for c in short)


def number_category(left, right, strict=False):
    ln, rn = numbers(left), numbers(right)
    if not ln or not rn:
        return 'missing_num'
    house = ln[0]
    if strict:
        rn = rn[:1]
    if house in rn:
        return 'house_in_target'
    for t in rn:
        difference = abs(len(t)-len(house))
        if difference and (not strict or difference == 1) and (subsequence(t, house) or subsequence(house, t)):
            return 'digit_deletion_or_insertion'
    if any(abs(int(t[:9])-int(house[:9])) <= 30 for t in rn):
        return 'small_offset'
    return 'other'


def summarize_slice(a, p, mask, threshold=.76):
    n = int(mask.sum())
    positive = int((mask & (a['y'] == 1)).sum())
    return dict(pairs=n, positives=positive, positive_rate=positive/n if n else None,
                fp=int((mask & (p >= threshold) & (a['y'] == 0)).sum()),
                rejected_true=int((mask & (p < threshold) & (a['y'] == 1)).sum()),
                rejected_false=int((mask & (p < threshold) & (a['y'] == 0)).sum()))


def errors():
    rows = [json.loads(line) for line in (PILOT / 'data/records.jsonl').open()]
    db = database('train')
    report = dict(data_manifest_sha256=digest(PILOT / 'data/manifest.json'), splits={},
                  caveat='Exploratory reused tune/dev; number heuristics are features, not generator ground truth.')
    for split in ('tune', 'dev'):
        a = load(split)
        p = a['baseline_100000']
        filename = 'enhanced_full_tune_probabilities.npy' if split == 'tune' else 'enhanced_dev_probabilities.npy'
        original_p = np.load(ROOT / 'artifacts/improvement_v2/cpu/fit_gate0_100k_v1' / filename)
        assert np.array_equal(p, original_p)
        broad, strict = [], []
        missing = np.empty(len(p), dtype=bool)
        leet_gain = np.empty(len(p), dtype=np.float32)
        for i, (li, ri) in enumerate(zip(a['left'], a['right'])):
            left, right = rows[li], rows[ri]
            broad.append(number_category(left['address'], right['address']))
            strict.append(number_category(left['address'], right['address'], strict=True))
            missing[i] = not right['address'].strip()
            l, r = anyascii(left['name']).lower(), anyascii(right['name']).lower()
            leet = r.translate(str.maketrans('013458', 'oieasb'))
            leet_gain[i] = fuzz.ratio(l, leet)-fuzz.ratio(l, r)
        result = dict(baseline=metric(a, p, .76), numeric={}, thresholds={})
        for name, values in [('scratch_style_any_target_number', broad), ('first_numbers_single_digit_change', strict)]:
            values = np.array(values)
            result['numeric'][name] = {}
            for subset, mask in [('all_candidates', np.ones(len(p), bool)), ('probability_gt_0.2', p > .2)]:
                result['numeric'][name][subset] = {
                    str(k): summarize_slice(a, p, mask & (values == k)) for k in np.unique(values)}
        # Deliberately naive diagnostic controls, never production policies.
        categories = np.array(broad)
        promote = p.copy()
        promote[(categories == 'digit_deletion_or_insertion') & (p > .2)] = 1.
        reject = p.copy()
        reject[categories == 'small_offset'] = -1.
        result['naive_rule_diagnostics_not_selected'] = dict(
            accept_all_digit_changes_above_02=metric(a, promote, .76),
            reject_all_small_offsets=metric(a, reject, .76))
        for floor in [0, .005, .01, .02, .05]:
            retained = p >= floor
            oracle = retained & (a['y'] == 1)
            result['thresholds'][str(floor)] = dict(
                retained_pairs=int(retained.sum()), retained_true=int(oracle.sum()),
                newly_unrecoverable_true=int(((a['y'] == 1) & ~retained).sum()),
                oracle_macro_f05=float(owner_scores(a, oracle.astype(float), .5).mean()))
        result['empty_target_address'] = summarize_slice(a, p, missing)
        result['empty_address_leet_ratio_gain_at_least_5'] = summarize_slice(a, p, missing & (leet_gain >= 5))
        result['leet_ratio_gain_at_least_5'] = summarize_slice(a, p, leet_gain >= 5)
        wrong = np.flatnonzero((p >= .76) & (a['y'] == 0))
        labeled_other = 0
        for i in wrong:
            row = db.execute('SELECT reference_id FROM labels WHERE target_id=?', (rows[a['right'][i]]['id'],)).fetchone()
            if row is not None:
                assert row[0] != rows[a['left'][i]]['id']
                labeled_other += 1
        result['false_positives_with_another_labeled_owner'] = labeled_other
        report['splits'][split] = result
        write(OUT / 'errors.json', report)
        print(json.dumps(dict(stage='errors', split=split, baseline=result['baseline'],
                             empty_address=result['empty_target_address'], other_owner=labeled_other)), flush=True)
    db.close()
    return report


def corpus():
    report, names, examples = {}, Counter(), []
    for split in ('train', 'test'):
        country = Counter()
        with (DATA / split / f'{split}_source1.tsv').open() as handle:
            next(handle)
            for line in handle:
                eid, name, address, co = line.rstrip('\n').split('\t')
                country[co] += 1
                if split == 'test' and co == 'France':
                    names[name] += 1
                    if (name == 'Bordeaux Amicale SARL' and address.startswith('237 ')) or eid in ('S1-802025215', 'S1-685644492'):
                        examples.append(dict(id=eid, name=name, address=address))
        manifest = json.loads((ROOT / f'artifacts/{split}_store_v2/manifest.json').read_text())
        assert sum(country.values()) == manifest['references']
        report[split] = dict(references_by_country=dict(country), references=manifest['references'],
                             targets=manifest['targets'], targets_per_reference=manifest['targets']/manifest['references'])
    report['french_names'] = dict(bordeaux_club_sarl=names['Bordeaux Club SARL'],
        top_repeated=names.most_common(10), distinct=len(names), examples=examples)
    db = database('train')
    count, total, mean, eight = db.execute('SELECT COUNT(*), SUM(n_matches), AVG(n_matches), SUM(n_matches>=8) FROM gold_rows').fetchone()
    labels = db.execute('SELECT COUNT(*) FROM labels').fetchone()[0]
    assert total == labels
    report['training_gold'] = dict(owners=count, pairs=total, mean_matches=mean,
        owners_eight_or_more=eight, owners_eight_or_more_rate=eight/count,
        target_exclusivity='labels.target_id PRIMARY KEY; full truth import rejects duplicates; count equals gold sum')
    db.close()
    fr = report['test']['references_by_country']['France']/report['test']['references']
    report['france_share'] = fr
    # Previously published v1 audit, not the reserved new 100k audit.
    old_audit = json.loads((ROOT / 'artifacts/duf-fel_bundle_v1/code/business_entity_resolution/reports/audit.json').read_text())
    counts = report['test']['references_by_country']
    non_fr = sum(counts[c]*old_audit['countries'][c]['macro_f05'] for c in ('India', 'US')) / (counts['India']+counts['US'])
    report['conditional_france_estimate'] = dict(score=(.922-(1-fr)*non_fr)/fr,
        assumed_public=.922, assumed_non_french=non_fr,
        caveat='Not measured: assumes public subset shares test country mix and non-French local accuracy transfers.')
    report['conditional_decoy_ratio'] = (report['test']['targets_per_reference']-mean)/(report['train']['targets_per_reference']-mean)
    report['decoy_caveat'] = 'Test match-count distribution is unknown; ratio assumes the training mean transfers.'
    old_counts = defaultdict(Counter)
    with (DATA / 'test/test_source1.tsv').open() as source, (ROOT / 'artifacts/duf-fel_bundle_v1/output/matching_results.tsv').open() as predictions:
        next(source)
        next(predictions)
        for reference, prediction in zip_longest(source, predictions):
            assert reference is not None and prediction is not None
            eid, name, address, co = reference.rstrip('\n').split('\t')
            pid, matches = prediction.rstrip('\n').split('\t')
            assert eid == pid
            n = len(matches.split(',')) if matches else 0
            old_counts[co]['references'] += 1
            old_counts[co]['predictions'] += n
            old_counts[co]['owners_eight_or_more'] += n >= 8
    report['v1_prediction_counts'] = {co: dict(d, mean_predictions=d['predictions']/d['references'],
        owners_eight_or_more_rate=d['owners_eight_or_more']/d['references']) for co, d in old_counts.items()}
    write(OUT / 'corpus.json', report)
    print(json.dumps(report), flush=True)
    return report


def pseudonyms():
    """Probability sample of supplied positive labels; keep train-fold only."""
    db = database('train')
    largest = db.execute('SELECT MAX(rowid) FROM labels').fetchone()[0]
    ids = random.Random(11).sample(range(1, largest+1), 30000)
    pairs = []
    for start in range(0, len(ids), 400):
        chunk = ids[start:start+400]
        pairs.extend(db.execute('SELECT target_id,reference_id,country FROM labels WHERE fold=\'train\' AND rowid IN ('
                                + ','.join('?' for _ in chunk) + ')', chunk).fetchall())
    low, words, country, low_country = [], Counter(), Counter(), Counter()
    for target, reference, co in pairs:
        target_name = db.execute('SELECT name FROM records WHERE entity_id=?', (target,)).fetchone()[0]
        reference_name = db.execute('SELECT name FROM anchors WHERE entity_id=?', (reference,)).fetchone()[0]
        l, r = anyascii(target_name).lower(), anyascii(reference_name).lower()
        country[co] += 1
        if fuzz.token_set_ratio(l, r) < 40 and fuzz.partial_ratio(l, r) < 50:
            low.append(dict(target=target_name, reference=reference_name, country=co))
            low_country[co] += 1
            words.update(re.findall('[a-z]+', l))
    result = dict(sampled_positive_pairs=len(pairs), low_name_similarity=len(low),
        fraction=len(low)/len(pairs), countries=dict(country), low_name_countries=dict(low_country),
        common_words=words.most_common(40), examples=low[:20], seed=11,
        scope='Random label rows, train fold only; no reserved audit labels.',
        caveat='Low name similarity is not proof of a synthetic generator, exact syllable inventory, or address-only identifiability.')
    db.close()
    write(OUT / 'pseudonyms.json', result)
    print(json.dumps(result), flush=True)
    return result


def conflicts():
    path = ROOT / 'artifacts/improvement_v2/inference/submission_controller_v1/queue.json'
    queue = json.loads(path.read_text())
    write(OUT / 'queue_snapshot.json', queue)
    totals = defaultdict(Counter)
    edges = {}
    seen_references = set()
    started = time.monotonic()
    for part, item in sorted(queue['complete'].items(), key=lambda x: int(x[0])):
        folder = Path(item['directory'])
        config = json.loads((folder / 'config.json').read_text())
        assert config['partition'] == int(part) and config['threshold'] == .76 and not config['labels_available']
        for shard in sorted((folder / 'shards').glob('[0-9]*')):
            done = json.loads((shard / 'done.json').read_text())
            for name in ('probabilities.npy', 'pair_ids.npz'):
                assert digest(shard / name) == done['output_sha256'][name]
            p = np.load(shard / 'probabilities.npy')
            with np.load(shard / 'pair_ids.npz') as z:
                a = {k: z[k] for k in z.files}
            assert len(p) == len(a['target_ids']) == len(a['query_ids'])
            refs = set(a['reference_ids'].tolist())
            assert len(refs) == len(a['reference_ids']) and not refs & seen_references
            seen_references.update(refs)
            accepted = p >= .76
            counts = np.bincount(a['query_ids'][accepted], minlength=len(a['reference_ids']))
            for co in np.unique(a['countries']):
                owner_mask = a['countries'] == co
                pair_mask = owner_mask[a['query_ids']]
                stat = totals[str(co)]
                stat['references'] += int(owner_mask.sum())
                stat['accepted_pairs'] += int((accepted & pair_mask).sum())
                stat['owners_eight_or_more'] += int((counts[owner_mask] >= 8).sum())
                stat['candidate_pairs'] += int(pair_mask.sum())
                stat['pairs_ge_0.02'] += int(((p >= .02) & pair_mask).sum())
            for i in np.flatnonzero(accepted):
                target = str(a['target_ids'][i])
                co = str(a['countries'][a['query_ids'][i]])
                value = float(p[i])
                if target not in edges:
                    edges[target] = [1, value, -1., co]
                else:
                    entry = edges[target]
                    assert entry[3] == co
                    entry[0] += 1
                    if value > entry[1]:
                        entry[2], entry[1] = entry[1], value
                    elif value > entry[2]:
                        entry[2] = value
        print(json.dumps(dict(stage='conflicts', partition=part, references=len(seen_references),
                              seconds=time.monotonic()-started)), flush=True)
    for n, first, second, co in edges.values():
        totals[co]['accepted_unique_targets'] += 1
        if n > 1:
            totals[co]['conflicting_targets'] += 1
            totals[co]['pairs_on_conflicting_targets'] += n
            totals[co]['minimum_incorrect_accepted_pairs_if_exclusive'] += n-1
            for margin in (0, .01, .025, .05, .10):
                if first-second <= margin:
                    totals[co][f'conflicting_targets_margin_le_{margin}'] += 1
    result = {}
    for co, stat in totals.items():
        d = dict(stat)
        d['mean_predictions_per_owner'] = d['accepted_pairs']/d['references']
        d['owners_eight_or_more_rate'] = d['owners_eight_or_more']/d['references']
        d['accepted_pair_conflict_rate'] = d.get('pairs_on_conflicting_targets', 0)/d['accepted_pairs']
        result[co] = d
    report = dict(complete=True, diagnostic_only=True, snapshot_partitions=sorted(map(int, queue['complete'])),
        references=len(seen_references), full_test_references=1732544,
        coverage=len(seen_references)/1732544, countries=result, seconds=time.monotonic()-started,
        caveat='Only complete partitions at snapshot. Missing rivals bias conflict counts down; no 1/coverage multiplier or accuracy estimate is justified.')
    write(OUT / 'conflicts.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['errors', 'corpus', 'conflicts', 'pseudonyms'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[args.mode]()

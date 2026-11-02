"""Measure an exact no-stealing prefilter on the unlabeled recovery probe.

Only sampled target IDs are retained while streaming the full frozen submission.
Already-owned targets cannot be added by the frozen additive recovery policy, so
excluding them before expensive scoring does not change that policy's decisions.
This is a cost estimate, not accuracy validation or a submission modification.
"""
from collections import Counter, defaultdict
import csv
import pickle

from v3_neural_test_pipeline import read, resources
from v3_transfer_data import ROOT, digest, write

WORK = ROOT / 'artifacts/improvement_v3/recovery_unclaimed_volume_v1'
SOURCE = ROOT / 'artifacts/improvement_v3/recovery_test_volume_v1'
PUBLIC = ROOT / 'artifacts/public_best_0976_20260926/matching_results.tsv'
PUBLIC_SHA = '5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118'


def claimed_sample(rows, interested):
    """Return ownership only for interested targets, bounded by sample size."""
    found = {}
    count = 0
    for row in rows:
        count += 1
        owner = row['source1_entity_id']
        for target in filter(None, row['matched_entity_ids'].split(',')):
            if target not in interested:
                continue
            if target in found:
                raise ValueError('Duplicate ownership in frozen submission sample')
            found[target] = owner
    return found, count


def run():
    WORK.mkdir(parents=True, exist_ok=False)
    resources()
    source = read(SOURCE / 'report.json')
    if not source['complete'] or digest(PUBLIC) != PUBLIC_SHA:
        raise ValueError('Frozen submitted file or recovery probe is invalid')
    for name, expected in source['outputs_sha256'].items():
        if digest(SOURCE / name) != expected:
            raise ValueError('Frozen recovery probe changed')
    raw = pickle.load((SOURCE / 'extra.pkl').open('rb'))
    routes = read(SOURCE / 'routes.json')
    selection = read(SOURCE / 'selection.json')
    if any(r[2] is not None or r[4] is not None for r in raw):
        raise ValueError('Unlabeled recovery rows unexpectedly have labels/scores')
    with PUBLIC.open(newline='') as handle:
        claims, owner_count = claimed_sample(csv.DictReader(handle, delimiter='\t'), {r[1] for r in raw})
    if owner_count != 1732544:
        raise ValueError('Submitted full-owner coverage differs')
    route_sets = {k: set(v) for k, v in routes.items()}
    stats = defaultdict(Counter)
    for r in selection['selected']:
        stats[r['country']]['sample_owners'] += 1
    kept, new_routes = [], defaultdict(list)
    for i, row in enumerate(raw):
        owned = row[1] in claims
        s = stats[row[3]]
        s['union_before'] += 1
        s['union_after'] += int(not owned)
        s['already_owned_by_same_s1'] += int(owned and claims[row[1]] == row[0])
        s['already_owned_elsewhere'] += int(owned and claims[row[1]] != row[0])
        for route, indices in route_sets.items():
            if i in indices:
                s[route + '_before'] += 1
                s[route + '_after'] += int(not owned)
                if not owned:
                    new_routes[route].append(len(kept))
        if not owned:
            kept.append(row)
    totals = selection['total_countries']
    estimate = {route: sum(totals[c] * s[route + '_after'] / s['sample_owners'] for c, s in stats.items())
                for route in ('native', 'website', 'union')}
    with (WORK / 'extra.pkl').open('wb') as handle:
        pickle.dump(kept, handle)
    write(WORK / 'routes.json', new_routes)
    if digest(PUBLIC) != PUBLIC_SHA:
        raise ValueError('Submission changed while computing prefilter')
    write(WORK / 'report.json', dict(complete=True, countries=stats, pairs_before=len(raw), pairs_after=len(kept),
        full_test_pair_volume_estimate=estimate, public_sha256=PUBLIC_SHA,
        source_report_sha256=digest(SOURCE / 'report.json'), code_sha256=digest(__file__),
        outputs_sha256={n: digest(WORK / n) for n in ('extra.pkl', 'routes.json')},
        labels_used=False, ready_for_submission=False,
        notes=['Exact prefilter only for additive no-stealing recovery; not valid for owner replacement.',
               'Stratified sample estimate is uncertain; no correctness or French accuracy inferred.']))


if __name__ == '__main__':
    run()

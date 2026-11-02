"""Replay frozen stackers against wider rivals without fitting or retuning.

Each original development result must first reproduce exactly. Only then report
the same policy against the wider, label-free rival cache. This is an exploratory
robustness check on reused owners, not an independent final accuracy estimate.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle

import joblib
import numpy as np

from ber.store import connect
from v3_neural_independent_audit import blend, lg, summary
from v3_neural_test_pipeline import SCRATCH, STRUCT, read, resources
from v3_owner_assignment import load_text, text_logit, winner
from v3_transfer_data import ROOT, PILOT, digest, write


def validate_same_edge(before, after):
    if before[:2] != after[:2] or before[3] != after[3] or before[5:9] != after[5:9]:
        raise ValueError('Duplicate edge text or country changed')
    if abs(float(before[4]) - float(after[4])) > 1e-8:
        raise ValueError('Duplicate edge CPU context changed')


def decisions(scores, base, text, groups, threshold, rule, eligible):
    keep = (scores >= threshold) & (base >= .01) & eligible
    for group in groups:
        chosen = group[keep[group]]
        if len(chosen) < 2:
            continue
        ix = winner(scores[chosen], text[chosen], **rule)
        keep[chosen] = False
        if ix is not None:
            keep[chosen[ix]] = True
    return keep


def structural(rows, manifest_path, output):
    """Reuse exact old features; compute only genuinely new rival features."""
    from v3_structural_features import View, core_name, city_proxy, pair_features, NAMES
    manifest = read(manifest_path)
    if manifest['features_code_sha256'] != digest(ROOT/'tools/v3_structural_features.py'):
        raise ValueError('Canonical structural definitions changed')
    fragments_path = ROOT/'artifacts/improvement_v3/structural_v1/feature_pilot_v2/fragments.json'
    if manifest['fragments_sha256'] != digest(fragments_path):
        raise ValueError('Training fragments changed')
    fragments = read(fragments_path)['fragments']
    known = {}
    for split in ('tune', 'dev', 'rivals'):
        info = manifest['inputs'][split]
        raw = Path(info['path']); feature = manifest_path.parent/f'{split}.npy'
        if digest(raw) != info['sha256'] or digest(feature) != info['features_sha256']:
            raise ValueError('Original structural input changed')
        raw_rows = pickle.load(raw.open('rb')); values = np.load(feature, allow_pickle=False)
        if values.shape != (len(raw_rows), len(NAMES)):
            raise ValueError('Structural feature shape changed')
        for row, value in zip(raw_rows, values):
            if row[:2] in known:
                before, old_value = known[row[:2]]
                validate_same_edge(before, row)
                if not np.array_equal(value, old_value):
                    raise ValueError('Repeated structural features differ')
            else:
                known[row[:2]] = (row, value.copy())
    new = [i for i, row in enumerate(rows) if row[:2] not in known]
    counts = Counter(); cities = Counter()
    interested = {(rows[i][3], core_name(rows[i][5])) for i in new}
    if interested:
        db = connect(ROOT/'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-8192')
        for n, row in enumerate(db.execute('SELECT country,name,address FROM anchors'), 1):
            key = (row['country'], core_name(row['name']))
            if key in interested:
                counts[key] += 1
                city = city_proxy(row['address'], row['country'])
                if city:
                    cities[(*key, city)] += 1
            if n % 250000 == 0:
                resources(); write(output/'progress.json', dict(stage='new_rival_full_s1_counts', anchors=n))
        db.close()
    @lru_cache(maxsize=16000)
    def view(name, address, country):
        return View.make(name, address, country, fragments)
    result = np.empty((len(rows), len(NAMES)), np.float32)
    for i, row in enumerate(rows):
        if row[:2] in known:
            before, value = known[row[:2]]; validate_same_edge(before, row)
            result[i] = value
        else:
            left = view(row[5], row[6], row[3]); right = view(row[7], row[8], row[3])
            key = (row[3], left.core)
            if counts[key] < 1:
                raise ValueError('New rival absent from full S1 name counts')
            result[i] = pair_features(left, right, counts[key], cities.get((*key, left.city), 0) if left.city else 0)
    if not np.isfinite(result).all():
        raise ValueError('Invalid structural values')
    return result, dict(reused_edges=len(rows)-len(new), new_edges=len(new), manifest_sha256=digest(manifest_path))


def run(baseline, candidate, widened, output):
    paths = [Path(baseline), Path(candidate)]; widened = Path(widened); output = Path(output)
    output.mkdir(parents=True, exist_ok=True); resources()
    reports = [read(p/'report.json') for p in paths]
    for report in reports:
        if not report['complete'] or report.get('widened_rival_coverage'):
            raise ValueError('Expected complete original-coverage audit')
        for path, expected in report['inputs_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Audited input changed: '+path)
    roots = [Path(r.get('ce_root') or SCRATCH/r['tag']) for r in reports]
    if roots[0].resolve() != roots[1].resolve():
        raise ValueError('Mixed CE precision/checkpoint sources')
    ce_root = roots[0]; union = read(widened/'union.json')
    if not union['complete']:
        raise ValueError('Wider rivals incomplete')
    for name, expected in union['outputs_sha256'].items():
        if digest(widened/name) != expected:
            raise ValueError('Wider-rival file changed')
    old_rows = pickle.load((SCRATCH/'rivals.pkl').open('rb')); old_keys = {r[:2] for r in old_rows}
    wider_rows = pickle.load((widened/'rivals.pkl').open('rb'))
    wider_ce = np.load(widened/'rivals_pce.npy', allow_pickle=False)
    wider_ops = np.load(widened/'rivals_ops.npy', allow_pickle=False)
    if wider_ce.shape != (len(wider_rows),) or wider_ops.shape != (len(wider_rows),28):
        raise ValueError('Wider-rival arrays misaligned')
    if not old_keys.issubset({r[:2] for r in wider_rows}):
        raise ValueError('Wider rivals lost original claims')
    import json
    ids = [json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    rows = []; ops = []; ce = []; data = {}; full_keys = set()
    for split in ('tune', 'dev'):
        with np.load(PILOT/'data'/f'{split}.npz', allow_pickle=False) as z:
            a = {k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','baseline_100000','left','right')}
        raw = pickle.load((SCRATCH/f'{split}.pkl').open('rb'))
        keys = {(ids[int(l)], ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        ix = np.asarray([keys[r[:2]] for r in raw]); expected = np.flatnonzero(a['baseline_100000'] >= .01)
        if len(set(ix)) != len(ix) or not np.array_equal(np.sort(ix), expected):
            raise ValueError('Incomplete focal neural route')
        if not np.array_equal(a['y'][ix], [r[2] for r in raw]):
            raise ValueError('Focal labels misaligned')
        start = len(rows); rows.extend(raw); full_keys.update(keys)
        data[split] = (a, ix, slice(start, len(rows)))
        ops.extend(np.load(SCRATCH/f'{split}.pkl.ops.npy', allow_pickle=False))
        ce.extend(np.load(ce_root/f'{split}_pce.npy', allow_pickle=False))
    focal_count = len(rows)
    for row, probability, features in zip(wider_rows, wider_ce, wider_ops):
        if row[:2] in full_keys:
            continue
        rows.append(row); ce.append(probability); ops.append(features)
    if len({r[:2] for r in rows}) != len(rows):
        raise ValueError('Duplicate wider claims')
    base = np.asarray([r[4] for r in rows]); ce = np.asarray(ce)
    if not np.isfinite(ce).all() or ((ce < 0) | (ce > 1)).any():
        raise ValueError('Invalid CE values')
    matrix = np.column_stack((np.asarray(ops), lg(base), lg(ce))).astype(np.float32)
    extra = None; extra_info = None
    if any(r['selected']['name'] == 'ops_ce_extra' for r in reports):
        source = next(r for r in reports if r['selected']['name'] == 'ops_ce_extra')
        manifests = [Path(p) for p in source['inputs_sha256'] if p.endswith('/manifest.json') and 'structural_neural_extra' in p]
        if len(manifests) != 1:
            raise ValueError('Unknown structural feature manifest')
        extra, extra_info = structural(rows, manifests[0], output)
    rule = read(STRUCT/'owner_eval/policy.json')['selected']; model = load_text(STRUCT/'text_scorer.json')
    text = np.asarray([text_logit({'name':r[5],'address':r[6]}, {'name':r[7],'address':r[8]}, model) for r in rows])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[1]].append(i)
    groups = [np.asarray(sorted(g, key=lambda i:rows[i][0])) for g in groups.values() if len(g)>1]
    old_eligible = np.asarray([i<focal_count or row[:2] in old_keys for i,row in enumerate(rows)])
    calibration = np.asarray([int(hashlib.sha256(str(o).encode()).hexdigest()[:8],16)%3==0 for o in data['tune'][0]['reference_ids']])
    results = []
    for path, report in zip(paths, reports):
        selected = report['selected']; name = selected['name']
        if name.startswith('blend_'):
            scores = blend(base, ce, float(name.split('_')[1])); model_sha = None
        else:
            mpath = path/(name+'.joblib'); stack = joblib.load(mpath); model_sha = digest(mpath)
            full = np.column_stack((matrix, extra)) if extra is not None else matrix
            scores = stack.predict_proba(full[:,report['feature_columns'][name]])[:,1]
        metrics = {}
        for label, eligible in (('original', old_eligible), ('widened', np.ones(len(rows),bool))):
            keep = decisions(scores, base, text, groups, selected['threshold'], rule, eligible)
            per_split = {}
            for split,(a,ix,sl) in data.items():
                accepted = np.zeros(len(a['y']),bool); accepted[ix] = keep[sl]
                s, metric = summary(a,accepted)
                if split == 'tune':
                    metric['calibration_macro_f05'] = float(s[calibration].mean())
                per_split[split] = metric
            metrics[label] = per_split
        original = metrics['original']
        if abs(original['dev']['macro_f05'] - report['models'][name]['development']['macro_f05']) > 1e-12:
            raise ValueError('Frozen original development policy did not replay exactly')
        if abs(original['tune']['calibration_macro_f05'] - selected['calibration_macro_f05']) > 1e-12:
            raise ValueError('Frozen original calibration policy did not replay exactly')
        results.append(dict(audit=str(path), report_sha256=digest(path/'report.json'), model_sha256=model_sha,
                            frozen_selection=selected, metrics=metrics))
    first, second = [r['metrics']['widened'] for r in results]
    comparison = dict(calibration_gain=second['tune']['calibration_macro_f05']-first['tune']['calibration_macro_f05'],
        development_gain=second['dev']['macro_f05']-first['dev']['macro_f05'],
        country_delta={c:second['dev']['countries'][c]-v for c,v in first['dev']['countries'].items()})
    write(output/'report.json', dict(complete=True, models=results, comparison=comparison, structural=extra_info,
        union_sha256=digest(widened/'union.json'), code_sha256=digest(__file__),
        fitted_or_retuned=False, ready_for_submission=False,
        note='Same frozen models and thresholds, wider bounded rivals. Reused development; no French labels.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','widened','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))

"""Evaluate a frozen empty-address-only expert; no fitting or deployment.

Both heads were fitted on the same fresh 5k owners. Fix their independently
tune-selected thresholds and compare conditional routing with the unchanged
structural head. Full owner-set scoring includes the original bounded rivals.
"""
from collections import defaultdict
import hashlib
import json
import pickle

import joblib
import numpy as np

from v3_empty_name_pilot import missing
from v3_neural_independent_audit import SCRATCH, STRUCT, lg, summary
from v3_owner_assignment import load_text, text_logit, winner
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

WORK = ROOT / 'artifacts/improvement_v3/empty_name_conditional_v1'
BASE = ROOT / 'artifacts/improvement_v3/fresh_stack_peerhalf_extra_v1/evaluation'
EXPERT = ROOT / 'artifacts/improvement_v3/empty_name_specialist_v1/combined_features/evaluation'
FEATURES = EXPERT.parent
CE = ROOT / 'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/half_calibration'


def conditional_scores(base, specialist, empty):
    if base.shape != specialist.shape or base.shape != empty.shape:
        raise ValueError('Conditional arrays are misaligned')
    if empty.dtype != np.bool_:
        raise ValueError('Conditional route must be Boolean')
    return np.where(empty, specialist, base)


def run():
    WORK.mkdir(parents=True, exist_ok=False)
    reports = [json.loads((p / 'report.json').read_text()) for p in (BASE, EXPERT)]
    for report in reports:
        for name, value in report['inputs_sha256'].items():
            if digest(name) != value:
                raise ValueError('Frozen audit input changed: ' + name)
    selected = [r['models']['ops_ce_extra']['selection'] for r in reports]
    if selected[0]['threshold'] != selected[1]['threshold']:
        raise ValueError('Separate-threshold ownership requires explicit calibration; not this pilot')
    threshold = selected[0]['threshold']
    identity = dict(baseline=digest(BASE / 'report.json'), specialist=digest(EXPERT / 'report.json'),
                    code=digest(__file__), fixed_threshold=threshold,
                    rule='Specialist head only when target address is naturally absent')
    write(WORK / 'identity.json', identity)
    ids = [json.loads(line)['id'] for line in (PILOT / 'data/records.jsonl').open()]
    rows, matrices, data, spans, full = [], [], {}, {}, set()
    for split in ('tune', 'dev'):
        with np.load(PILOT / 'data' / (split + '.npz'), allow_pickle=False) as z:
            a = {k: z[k] for k in ('reference_ids', 'query_ids', 'y', 'true_counts', 'countries',
                                 'left', 'right', 'baseline_100000')}
        raw = pickle.load((SCRATCH / (split + '.pkl')).open('rb'))
        keys = {(ids[int(l)], ids[int(r)]): i for i, (l, r) in enumerate(zip(a['left'], a['right']))}
        ix = np.asarray([keys[r[:2]] for r in raw])
        if not np.array_equal(np.sort(ix), np.flatnonzero(a['baseline_100000'] >= .01)):
            raise ValueError('Routed focal coverage changed')
        full.update(keys); data[split] = (a, ix)
        start = len(rows); rows.extend(raw); spans[split] = slice(start, len(rows))
        matrices.append(np.column_stack((np.load(SCRATCH / (split + '.pkl.ops.npy')),
                        lg([r[4] for r in raw]), lg(np.load(CE / (split + '_pce.npy'))),
                        np.load(FEATURES / (split + '.npy')))))
    rivals = pickle.load((SCRATCH / 'rivals.pkl').open('rb'))
    ix = np.asarray([i for i, r in enumerate(rivals) if r[:2] not in full])
    rows.extend(rivals[i] for i in ix)
    matrices.append(np.column_stack((np.load(SCRATCH / 'rivals.pkl.ops.npy')[ix],
                    lg([rivals[i][4] for i in ix]), lg(np.load(CE / 'rivals.npy')[ix]),
                    np.load(FEATURES / 'rivals.npy')[ix])))
    X = np.vstack(matrices).astype(np.float32)
    if X.shape != (len(rows), 57):
        raise ValueError('Frozen feature dimensions changed')
    heads = [joblib.load(root / 'ops_ce_extra.joblib') for root in (BASE, EXPERT)]
    pbase = heads[0].predict_proba(X[:, :56])[:, 1]
    pexpert = heads[1].predict_proba(X)[:, 1]
    empty = np.asarray([missing(r[8]) for r in rows])
    pmix = conditional_scores(pbase, pexpert, empty)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[1]].append(i)
    groups = [np.asarray(sorted(v, key=lambda i: rows[i][0])) for v in groups.values() if len(v) > 1]
    base = np.asarray([r[4] for r in rows])
    text_model = load_text(STRUCT / 'text_scorer.json')
    rule = json.loads((STRUCT / 'owner_eval/policy.json').read_text())['selected']
    text = np.asarray([text_logit({'name': r[5], 'address': r[6]},
                                  {'name': r[7], 'address': r[8]}, text_model) for r in rows])
    def decide(p):
        keep = (p >= threshold) & (base >= .01)
        for group in groups:
            eligible = group[keep[group]]
            if len(eligible) < 2:
                continue
            chosen = winner(p[eligible], text[eligible], **rule)
            keep[eligible] = False
            if chosen is not None:
                keep[eligible[chosen]] = True
        return keep
    decisions = {k: decide(p) for k, p in [('baseline', pbase), ('specialist_all', pexpert), ('conditional', pmix)]}
    result = {}
    for split, (a, ix) in data.items():
        result[split] = {}; values = {}
        calibration = np.asarray([int(hashlib.sha256(str(o).encode()).hexdigest()[:8], 16) % 3 == 0
                                  for o in a['reference_ids']])
        for key, mask in decisions.items():
            full_keep = np.zeros(len(a['y']), bool); full_keep[ix] = mask[spans[split]]
            score, metrics = summary(a, full_keep); values[key] = score
            result[split][key] = dict(metrics=metrics)
            if split == 'tune':
                result[split][key]['calibration_macro_f05'] = float(score[calibration].mean())
            if split == 'dev' and key != 'conditional':
                j = 0 if key == 'baseline' else 1
                expected = reports[j]['models']['ops_ce_extra']['development']['macro_f05']
                if abs(metrics['macro_f05'] - expected) > 1e-12:
                    raise ValueError('Frozen baseline/expert failed exact replay')
        old = decisions['baseline'][spans[split]]
        new = decisions['conditional'][spans[split]]
        y = a['y'][ix]
        result[split]['comparison'] = bootstrap(values['baseline'], values['conditional'])
        result[split]['changes'] = dict(recovered_true=int((~old & new & (y == 1)).sum()),
            lost_true=int((old & ~new & (y == 1)).sum()), added_false=int((~old & new & (y == 0)).sum()),
            removed_false=int((old & ~new & (y == 0)).sum()))
        if np.any((old != new) & ~empty[spans[split]]):
            raise ValueError('A nonempty target changed under empty-only routing')
    write(WORK / 'report.json', dict(complete=True, results=result, identity_sha256=digest(WORK / 'identity.json'),
        ready_for_submission=False, limitations=['Previously explored US/India development; no French labels.',
        'Original bounded rival cache, not exhaustive competition.', 'No refitting, no threshold search, no deployment.']))


if __name__ == '__main__':
    run()

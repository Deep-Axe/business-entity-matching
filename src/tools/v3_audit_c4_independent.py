"""Read-only review of peer c4 cascade and hard-training mix; no peer writes."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from v3_neural_independent_audit import SCRATCH, STRUCT, summary
from v3_owner_assignment import load_text, text_logit, winner
from v3_public0976_error_audit import features, model
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap


WORK = ROOT / 'artifacts/improvement_v3/c4_independent_review_v1'


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'report.json').exists():
        raise FileExistsError('Preserve completed audit')
    started = time.monotonic()
    bound = {}

    def track(path):
        path = Path(path)
        bound[str(path)] = digest(path)
        return path

    def raw(path):
        with track(path).open('rb') as handle:
            return pickle.load(handle)

    def arr(path):
        return np.load(track(path), allow_pickle=False)

    def matrix(rows, path, ce):
        return features(rows, arr(str(path) + '.ops.npy'), ce,
                        np.column_stack((arr(str(path) + '.nf.npy'), arr(str(path) + '.nf2.npy'))))

    ids = [json.loads(line)['id'] for line in track(PILOT / 'data/records.jsonl').open()]
    rows, mats, data, all_keys = [], {'old': [], 'c4': []}, {}, set()
    for split in ('dev', 'tune'):
        path = SCRATCH / (split + '.pkl')
        r = raw(path)
        span = slice(len(rows), len(rows) + len(r))
        rows.extend(r)
        for name, tag in (('old', 'e5_cont1300k'), ('c4', 'c4')):
            mats[name].append(matrix(r, path, arr(SCRATCH / tag / (split + '_pce.npy'))))
        with arr(PILOT / 'data' / (split + '.npz')) as z:
            a = {k: z[k] for k in ('reference_ids', 'query_ids', 'true_counts', 'countries', 'y', 'left', 'right')}
        keys = {(ids[int(l)], ids[int(t)]): i for i, (l, t) in enumerate(zip(a['left'], a['right']))}
        ix = np.asarray([keys[x[:2]] for x in r])
        if not np.array_equal(a['y'][ix], [x[2] for x in r]):
            raise ValueError('Labels misaligned')
        all_keys.update(keys)
        data[split] = (a, ix, span)
    M = {k: np.vstack(v) for k, v in mats.items()}
    Y = np.asarray([r[2] for r in rows])
    owners = np.asarray([r[0] for r in rows])
    focal = len(rows)
    rivals = raw(SCRATCH / 'rivals.pkl')
    aux = np.asarray([i for i, r in enumerate(rivals) if r[:2] not in all_keys])
    all_rows = rows + [rivals[i] for i in aux]
    AM = {}
    for name, tag in (('old', 'e5_cont1300k'), ('c4', 'c4')):
        rm = matrix(rivals, SCRATCH / 'rivals.pkl', arr(SCRATCH / ('rivals.pkl.' + tag + '.npy')))
        AM[name] = np.vstack((M[name], rm[aux]))
    tm = load_text(track(STRUCT / 'text_scorer.json'))
    rule = json.loads(track(STRUCT / 'owner_eval/policy.json').read_text())['selected']
    text = np.asarray([text_logit({'name': r[5], 'address': r[6]}, {'name': r[7], 'address': r[8]}, tm) for r in all_rows])
    base = np.asarray([r[4] for r in all_rows])
    groups = defaultdict(list)
    for i, r in enumerate(all_rows):
        groups[r[1]].append(i)
    groups = [np.asarray(sorted(v, key=lambda i: all_rows[i][0])) for v in groups.values() if len(v) > 1]

    def decide(p):
        keep = (p >= .75) & (base >= .01)
        for g in groups:
            ix = g[keep[g]]
            if len(ix) < 2:
                continue
            ch = winner(p[ix], text[ix], **rule)
            keep[ix] = False
            if ch is not None:
                keep[ix[ch]] = True
        return keep

    full, band = np.zeros(focal, bool), np.zeros(focal, bool)
    full_p, band_p = np.zeros(focal), np.zeros(focal)
    for fold, (fit, held) in enumerate(GroupKFold(5).split(M['old'], Y, owners)):
        print('fold', fold, flush=True)
        op = model().fit(M['old'][fit], Y[fit]).predict_proba(AM['old'])[:, 1]
        cp = model().fit(M['c4'][fit], Y[fit]).predict_proba(AM['c4'])[:, 1]
        bp = np.where((op >= .05) & (op < .98), cp, op)
        full[held], band[held] = decide(cp)[held], decide(bp)[held]
        full_p[held], band_p[held] = cp[held], bp[held]
    with arr(ROOT / 'artifacts/improvement_v3/v6_band_error_audit_v1/decisions.npz') as z:
        v6 = z['band_cC_keep']
    metrics, comparisons, transitions = {}, {}, {}
    for split, (a, ix, span) in data.items():
        metrics[split], per_owner = {}, {}
        for tag, mask in (('v6_band', v6), ('c4_full', full), ('c4_band', band)):
            keep = np.zeros(len(a['y']), bool)
            keep[ix] = mask[span]
            per_owner[tag], metrics[split][tag] = summary(a, keep)
        comparisons[split] = bootstrap(per_owner['v6_band'], per_owner['c4_band'])
        add = band[span] & ~v6[span]
        remove = v6[span] & ~band[span]
        y = Y[span]
        transitions[split] = dict(added_true=int((add & (y == 1)).sum()), added_false=int((add & (y == 0)).sum()),
                                  removed_true=int((remove & (y == 1)).sum()), removed_false=int((remove & (y == 0)).sum()))
    peer_rows, peer_p = raw(SCRATCH / 'stack_oof_c4.pkl')
    ordered = [(s,) + r for s in ('dev', 'tune') for r in rows[data[s][2]]]
    if peer_rows != ordered or not np.allclose(peer_p, full_p, atol=1e-12, rtol=0):
        raise ValueError('Peer c4 OOF recipe differs')
    final = model().fit(M['c4'], Y)
    path = SCRATCH / 'test/part_000.pkl'
    tr = raw(path)
    ce = arr(str(path) + '.c4.npy')
    use = np.isfinite(ce)
    ops = arr(str(path) + '.ops.npy')[use]
    freq = np.column_stack((arr(str(path) + '.nf.npy'), arr(str(path) + '.nf2.npy')))[use]
    fm = features([r for r, k in zip(tr, use) if k], ops, ce[use], freq)
    generated = final.predict_proba(fm)[:, 1].astype(np.float32)
    saved = arr(str(path) + '.stack_c4_nocty_nf2.npy')[use]
    if not np.array_equal(generated, saved):
        raise ValueError('Final fit deployment mismatch')
    partitions, band_count = 0, 0
    for path in sorted((SCRATCH / 'test').glob('part_*.pkl')):
        old = arr(str(path) + '.stack_e5_cont1300k_nocty_nf2.npy')
        new = arr(str(path) + '.stack_c4_nocty_nf2.npy')
        deployed = arr(str(path) + '.stack_c4_band.npy')
        mask = (old >= .05) & (old < .98)
        if not np.array_equal(deployed, np.where(mask, new, old)) or not np.isfinite(deployed).all():
            raise ValueError('Test band mismatch: ' + str(path))
        partitions += 1
        band_count += int(mask.sum())
    # Check the actual uploaded training mix; do not trust file naming alone.
    mix = raw(SCRATCH / 'hard_mix.pkl')
    heldout = set()
    for a, _, _ in data.values():
        heldout.update(map(str, a['reference_ids']))
    mix_owners = {r[0] for r in mix}
    keys = [r[:2] for r in mix]
    mix_report = dict(rows=len(mix), unique_pairs=len(set(keys)), owners=len(mix_owners),
                      overlap_with_all_heldout_owners=len(mix_owners & heldout),
                      positive_rate=float(np.mean([r[2] for r in mix])), countries=dict(Counter(r[3] for r in mix)),
                      labels=dict(Counter(str(r[2]) for r in mix)),
                      row4_caveat='Training row[4] is selector probability, not the actual 100k model probability.')
    if mix_report['overlap_with_all_heldout_owners'] or len(set(keys)) != len(keys):
        raise ValueError('Training leakage or duplicate hard pairs')
    matching = SCRATCH / 'sub_v8_c4_fr090/matching_results.tsv'
    candidate = SCRATCH / 'sub_v8_c4_fr090/candidate_pairs.tsv'
    report = dict(metrics=metrics, paired_comparison=comparisons, transitions=transitions, hard_mix=mix_report,
                  test_band_partitions=partitions, test_band_pairs=band_count, final_fit_partition0_exact=True,
                  matching_sha256=digest(matching), candidate_sha256=digest(candidate),
                  candidate_identical_to_prior_validated_set=digest(candidate) == '3626305ffcc2cefb1e0a5d36b4fe7d736dfa1c3c4711635bd59108e92183c065',
                  caveats=['Reused US/India owner-grouped OOF is exploratory, not fresh validation.',
                           'French threshold has no labeled validation here.',
                           'Bounded existing rivals; no complete test-like owner competition.',
                           'No Kaggle relaunch, peer-file changes or external submission.'],
                  seconds=time.monotonic()-started, inputs_sha256=bound)
    for path, h in bound.items():
        if digest(path) != h:
            raise ValueError('Inputs changed: ' + path)
    write(WORK / 'report.json', report)
    np.savez_compressed(WORK / 'decisions.npz', full_keep=full, band_keep=band, full_p=full_p, band_p=band_p)
    write(WORK / 'completion.json', dict(complete=True, report_sha256=digest(WORK / 'report.json')))
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    run()

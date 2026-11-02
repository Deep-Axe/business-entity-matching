"""Audit the frozen v10 dense addition without modifying peer artifacts.

Reconstruct final-fit scores first, then use shared owner-held-out folds for
both stackers and the deployed ownership rule. The rival gallery is bounded;
this is reused US/India validation, not a measurement of French test accuracy.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from ber.store import connect
from v3_owner_assignment import load_text, text_logit, winner
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

C = Path('work/ce')
MULTICE = '--individual-ce' in sys.argv
OUT = ROOT / ('artifacts/improvement_v3/v10_multice_pilot_v1' if MULTICE else 'artifacts/improvement_v3/v10_combined_audit_v1')
STRUCT = ROOT / 'artifacts/improvement_v3/structural_v1'


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=.04,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.,
        categorical_features=[14], random_state=0)


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    return np.log(p / (1-p))


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = {}

    def track(p):
        p = Path(p); bound[str(p)] = digest(p); return p

    def raw(p):
        with track(p).open('rb') as h:
            return pickle.load(h)

    def arr(p):
        return np.load(track(p), allow_pickle=False)

    def progress(stage, **kw):
        d = dict(stage=stage, seconds=time.monotonic()-started, **kw)
        write(OUT/'progress.json', d); print(json.dumps(d), flush=True)

    def matrix(rows, path, ce):
        x = np.column_stack((arr(str(path)+'.ops.npy'),
            logit([r[4] for r in rows]), logit(ce),
            arr(str(path)+'.nf.npy'), arr(str(path)+'.nf2.npy'))).astype(np.float32)
        assert x.shape == (len(rows), 33)
        assert np.isfinite(np.delete(x, 28, axis=1)).all()
        return x

    decl = json.loads(track(C/'sub_v10_c8_dense/report.json').read_text())
    assert decl['score'] == 'stack_c8_band' and decl['t'] == .75
    assert decl['extra_t'] == .9 and decl['extra_country_t'] == {'France': .95}
    assert digest(track(C/'sub_v10_c8_dense/matching_results.tsv')) == decl['matching_sha256']
    rule = json.loads(track(STRUCT/'owner_eval/policy.json').read_text())['selected']
    tm = load_text(track(STRUCT/'text_scorer.json'))

    rows, frows = [], []
    mats = {'old': [], 'c8': [], 'dense': []}
    arrays, spans = {}, {}
    for sp in ('dev', 'tune'):
        r = raw(C/f'{sp}.pkl'); fr = raw(C/f'{sp}_f.pkl')
        assert [v[:2] for v in fr[:len(r)]] == [v[:2] for v in r]
        assert all(np.isnan(v[4]) for v in fr[len(r):])
        start = len(rows); rows.extend(r); spans[sp] = slice(start, len(rows))
        frows.extend(fr)
        for name, tag in (('old', 'e5_cont1300k'), ('c8', 'c8')):
            mats[name].append(matrix(r, C/f'{sp}.pkl', arr(C/tag/f'{sp}_pce.npy')))
        mats['dense'].append(matrix(fr, C/f'{sp}_f.pkl', arr(C/'e5_cont1300k'/f'{sp}_f_pce.npy')))
        with arr(PILOT/'data'/f'{sp}.npz') as z:
            arrays[sp] = {k: z[k] for k in ('reference_ids', 'true_counts', 'countries', 'query_ids', 'y', 'left', 'right')}
    mats = {k: np.vstack(v) for k, v in mats.items()}
    y = np.asarray([r[2] for r in rows]); fy = np.asarray([r[2] for r in frows])
    owners = np.asarray([r[0] for r in rows]); fowners = np.asarray([r[0] for r in frows])
    oldkeys = {r[:2] for r in rows}
    newix = np.asarray([i for i, r in enumerate(frows) if np.isnan(r[4])])
    newrows = [frows[i] for i in newix]; newkeys = {r[:2] for r in newrows}
    assert not (oldkeys & newkeys) and len(newkeys) == len(newrows)
    refs = [str(o) for sp in ('dev', 'tune') for o in arrays[sp]['reference_ids']]
    db = connect(ROOT/'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-8192')
    gold = {o: {v[0] for v in db.execute('SELECT target_id FROM labels WHERE reference_id=?', (o,))} for o in refs}
    db.close()
    for sp, a in arrays.items():
        assert [len(gold[str(o)]) for o in a['reference_ids']] == list(a['true_counts'])
    assert all(int(r[1] in gold[r[0]]) == r[2] for r in frows)

    # Confirm the actual top5/similarity selection, independent of file naming.
    dense = defaultdict(lambda: {'name_addr': [], 'name': []})
    with arr(C/'dense_devtune.npz') as z:
        qid = z['qid']
        for k in z.files:
            if not k.endswith('_ix'):
                continue
            view, country = k[:-3].rsplit('_', 1)
            for qi, ids, ss in zip(z[k], z[f'{view}_{country}_id'], z[f'{view}_{country}_s']):
                dense[str(qid[qi])][view] = [
                    (f'S3-{int(v)-1000000000}' if v >= 1000000000 else f'S2-{int(v)}', float(s))
                    for v, s in zip(ids, ss) if v >= 0]
    expected_new = {(o, t) for o, views in dense.items()
        for view, pairs in views.items() for t, s in pairs[:5]
        if s >= (.895 if view == 'name_addr' else .95)} - oldkeys
    assert expected_new == newkeys, (len(expected_new), len(newkeys), len(expected_new ^ newkeys))
    progress('inputs_verified', original_pairs=len(rows), new_pairs=len(newrows), new_true=sum(r[2] for r in newrows))

    # Verify reconstruction against actual exported test score caches.
    finals = {k: model().fit(mats[k], fy if k == 'dense' else y) for k in mats}
    parity = {}
    p = C/'test/part_000.pkl'; tr = raw(p)
    for name, tag in (('old', 'e5_cont1300k'), ('c8', 'c8')):
        ce = arr(str(p)+f'.{tag}.npy'); valid = np.flatnonzero(np.isfinite(ce))[:4096]
        x = matrix(tr, p, np.nan_to_num(ce, nan=.5))[valid]
        got = finals[name].predict_proba(x)[:, 1].astype(np.float32)
        saved = arr(str(p)+f'.stack_{tag}_nocty_nf2.npy')[valid]
        parity[name] = dict(sample_pairs=len(valid), max_delta=float(np.max(np.abs(got-saved))), exact=bool(np.array_equal(got, saved)))
        assert parity[name]['max_delta'] <= 1e-7, parity
    op = arr(str(p)+'.stack_e5_cont1300k_nocty_nf2.npy')
    cp = arr(str(p)+'.stack_c8_nocty_nf2.npy')
    bp = arr(str(p)+'.stack_c8_band.npy')
    assert np.array_equal(bp, np.where((op >= .05) & (op < .98), cp, op))
    p = C/'test_dense/chunk_011.pkl'; tr = raw(p)
    x = matrix(tr, p, arr(str(p)+'.e5_cont1300k.npy'))
    ix = np.arange(0, len(tr), max(1, len(tr)//4096))
    got = finals['dense'].predict_proba(x[ix])[:, 1].astype(np.float32)
    saved = arr(str(p)+'.stack_e5_cont1300k_nocty_nf2.npy')[ix]
    parity['dense'] = dict(sample_pairs=len(ix), max_delta=float(np.max(np.abs(got-saved))), exact=bool(np.array_equal(got, saved)))
    write(OUT/'test_probability_parity.json', parity)
    assert parity['dense']['max_delta'] <= 1e-7, parity
    del finals, tr, x
    progress('actual_test_score_recipe_reproduced', parity=parity)

    # Bound competitors identically to prior coherent audits, plus ALL new
    # queried dev/tune edges. No gold-based competitor selection.
    ids = [json.loads(s)['id'] for s in track(PILOT/'data/records.jsonl').open()]
    all_raw_keys = {(ids[int(l)], ids[int(r)]) for a in arrays.values() for l, r in zip(a['left'], a['right'])}
    del ids
    rivals = raw(C/'rivals.pkl')
    aux = np.asarray([i for i, r in enumerate(rivals) if r[:2] not in all_raw_keys and r[:2] not in newkeys])
    oldall = rows + [rivals[i] for i in aux]
    am = {}
    for name, tag in (('old', 'e5_cont1300k'), ('c8', 'c8')):
        rm = matrix(rivals, C/'rivals.pkl', arr(C/f'rivals.pkl.{tag}.npy'))
        am[name] = np.vstack((mats[name], rm[aux]))
    if MULTICE:
        tags = ['e5_cont1300k'] + ['k'+c for c in 'ABCDEFG']
        individual = np.column_stack([logit(np.concatenate([
            arr(C/tag/f'{sp}_pce.npy') for sp in ('dev','tune')])) for tag in tags])
        individual_r = np.column_stack([logit(arr(C/f'rivals.pkl.{tag}.npy')) for tag in tags])
        # Keep the existing ensemble feature; add each component separately.
        um = np.column_stack((mats['c8'], individual)).astype(np.float32)
        uam = np.column_stack((am['c8'], np.vstack((individual, individual_r[aux])))).astype(np.float32)
        assert np.isfinite(um).all() and np.isfinite(uam).all()
    allrows = oldall + newrows
    assert len({r[:2] for r in allrows}) == len(allrows)
    nold, nbase = len(rows), len(oldall)
    text = np.asarray([text_logit({'name':r[5], 'address':r[6]}, {'name':r[7], 'address':r[8]}, tm) for r in allrows])
    groups = defaultdict(list)
    for i, r in enumerate(allrows):
        groups[r[1]].append(i)
    groups = [np.asarray(sorted(v, key=lambda i: allrows[i][0])) for v in groups.values() if len(v) > 1]
    basegate = np.asarray([r[4] >= .01 for r in oldall])
    focalix = np.r_[np.arange(nold), np.arange(nbase, len(allrows))]
    focalrows = rows + newrows

    def decide(p, add):
        keep = np.r_[(p[:nbase] >= .75) & basegate, (p[nbase:] >= .9) & add]
        before = keep.copy()
        for g in groups:
            ix = g[keep[g]]
            if len(ix) > 1:
                ch = winner(p[ix], text[ix], **rule); keep[ix] = False
                if ch is not None:
                    keep[ix[ch]] = True
        return keep[focalix], before[focalix]

    folds = list(GroupKFold(5).split(mats['old'], y, owners))
    ofold = {o: k for k, (_, held) in enumerate(folds) for o in owners[held]}
    for o in refs:
        ofold.setdefault(o, int(hashlib.sha256(o.encode()).hexdigest()[:8], 16) % 5)
    ff = np.asarray([ofold[o] for o in fowners])
    rf = np.asarray([ofold[r[0]] for r in focalrows])
    keep = {k: np.zeros(len(focalrows), bool) for k in ('c8', 'combined')}
    if MULTICE:
        keep['individual_ce'] = np.zeros(len(focalrows), bool)
    before = {k: np.zeros(len(focalrows), bool) for k in keep}
    oof8 = np.zeros(nold)
    fpred = np.zeros(len(focalrows))
    for k, (fit, held) in enumerate(folds):
        progress('coherent_owner_fold', fold=k)
        assert not (set(owners[fit]) & set(owners[held]))
        assert not (set(fowners[ff != k]) & {o for o in refs if ofold[o] == k})
        op = model().fit(mats['old'][fit], y[fit]).predict_proba(am['old'])[:, 1]
        cp = model().fit(mats['c8'][fit], y[fit]).predict_proba(am['c8'])[:, 1]
        dp = model().fit(mats['dense'][ff != k], fy[ff != k]).predict_proba(mats['dense'][newix])[:, 1]
        bp = np.where((op >= .05) & (op < .98), cp, op)
        p = np.r_[bp, dp]
        mask = rf == k
        oof8[held] = cp[held]
        fpred[mask] = p[focalix][mask]
        for name, add in (('c8', False), ('combined', True)):
            after, pre = decide(p, add)
            keep[name][mask] = after[mask]; before[name][mask] = pre[mask]
        if MULTICE:
            up = model().fit(um[fit], y[fit]).predict_proba(uam)[:, 1]
            up = np.where((op >= .05) & (op < .98), up, op)
            after, pre = decide(np.r_[up, dp], True)
            keep['individual_ce'][mask] = after[mask]
            before['individual_ce'][mask] = pre[mask]
    sr, sp = raw(C/'stack_oof_c8.pkl')
    assert [r[1:3] for r in sr] == [r[:2] for r in rows]
    assert np.allclose(sp, oof8, atol=1e-12, rtol=0)

    v6 = arr(ROOT/'artifacts/improvement_v3/v6_band_error_audit_v1/decisions.npz')
    keep['v6'] = np.r_[v6['band_cC_keep'], np.zeros(len(newrows), bool)]
    v6.close()
    old_by_owner, expanded_by_owner = defaultdict(set), defaultdict(set)
    for r in rows:
        old_by_owner[r[0]].add(r[1]); expanded_by_owner[r[0]].add(r[1])
    for r in newrows:
        expanded_by_owner[r[0]].add(r[1])

    def metric(a, mask):
        pred = defaultdict(set)
        for r, k in zip(focalrows, mask):
            if k:
                pred[r[0]].add(r[1])
        tt = a['true_counts']; rr = list(map(str, a['reference_ids']))
        tp = np.asarray([len(pred[o] & gold[o]) for o in rr])
        kk = np.asarray([len(pred[o]) for o in rr]); den = tt + 4*kk
        s = np.divide(5*tp, den, out=np.ones(len(rr)), where=den!=0)
        return s, dict(macro_f05=float(s.mean()), tp=int(tp.sum()), fp=int((kk-tp).sum()),
            fn=int((tt-tp).sum()), countries={str(c): float(s[a['countries']==c].mean()) for c in np.unique(a['countries'])})

    reports = {}; transitions = {}
    for split, a in arrays.items():
        rr = set(map(str, a['reference_ids'])); scores = {}; met = {}
        for name, mask in keep.items():
            scores[name], met[name] = metric(a, mask)
        ceilings = {}
        for name, cand in (('routed', old_by_owner), ('v10_filtered_union', expanded_by_owner)):
            r = np.asarray([len(cand[str(o)] & gold[str(o)]) for o in a['reference_ids']]); t = a['true_counts']
            s = np.divide(5*r, t+4*r, out=np.ones(len(r)), where=(t+4*r)!=0)
            ceilings[name] = dict(macro_f05=float(s.mean()), missed_true=int((t-r).sum()))
        r = np.bincount(a['query_ids'], weights=a['y'], minlength=len(a['reference_ids'])); t = a['true_counts']
        ceilings['raw_original_retrieval'] = dict(macro_f05=float(np.divide(5*r,t+4*r,out=np.ones(len(r)),where=t+4*r!=0).mean()), missed_true=int((t-r).sum()))
        for top in (5, 30):
            r = np.asarray([len((old_by_owner[str(o)] | {t for v in dense[str(o)].values() for t, s in v[:top]}) & gold[str(o)]) for o in a['reference_ids']])
            ceilings[f'unfiltered_top{top}_union'] = dict(macro_f05=float(np.divide(5*r,t+4*r,out=np.ones(len(r)),where=t+4*r!=0).mean()), missed_true=int((t-r).sum()))
        added = keep['combined'] & ~keep['c8']; removed = keep['c8'] & ~keep['combined']
        trans = Counter(); errors = []
        for i, row in enumerate(focalrows):
            if row[0] not in rr:
                continue
            if added[i] or removed[i]:
                kind = ('added_' if added[i] else 'removed_') + ('true' if row[2] else 'false')
                trans[kind] += 1
                if i >= nold:
                    trans['accepted_new_true' if row[2] else 'accepted_new_false'] += int(keep['combined'][i])
                errors.append(dict(owner=row[0], target=row[1], kind=kind, country=row[3], score=float(fpred[i]),
                    new_route=i>=nold, original_raw_retrieval=row[:2] in all_raw_keys))
        write(OUT/f'{split}_changed_links.json', errors)
        cg = ceilings['v10_filtered_union']['macro_f05'] - ceilings['routed']['macro_f05']
        dg = met['combined']['macro_f05'] - met['c8']['macro_f05']
        reports[split] = dict(metrics=met, oracle=ceilings,
            dense_delta=bootstrap(scores['c8'], scores['combined']),
            total_delta_vs_v6=bootstrap(scores['v6'], scores['combined']), transitions=dict(trans),
            captured_net_macro_fraction_of_added_oracle=dg/cg if cg else None,
            fraction_remaining_classification_gap_closed=dg/(ceilings['v10_filtered_union']['macro_f05']-met['c8']['macro_f05']))
        if MULTICE:
            reports[split]['individual_ce_delta'] = bootstrap(scores['combined'], scores['individual_ce'])
            ta = keep['individual_ce'] & ~keep['combined']; tr = keep['combined'] & ~keep['individual_ce']
            t = Counter()
            for i, r in enumerate(focalrows):
                if r[0] in rr:
                    if ta[i]: t['added_true' if r[2] else 'added_false'] += 1
                    if tr[i]: t['removed_true' if r[2] else 'removed_false'] += 1
            reports[split]['individual_ce_transitions'] = dict(t)
    for p, h in bound.items():
        assert digest(p) == h, 'Input changed during audit: '+p
    report = dict(complete=True, submission_sha256=decl['matching_sha256'], frozen_policy=decl,
        test_probability_parity=parity, peer_c8_oof_reproduced=True, splits=reports,
        original_pairs=len(rows), new_pairs=len(newrows), extra_rival_pairs=len(aux),
        caveats=['Reused US/India owner-held-out stacker evaluation; not a fresh untouched holdout.',
            'No French labels: French accuracy and its new .95 acceptance rule remain unmeasured.',
            'Available bounded original rivals plus expanded dev/tune edges, not exhaustive reverse-owner competition.',
            'Dense stacker uses NaN for unavailable p100k, exactly as deployed; it does not compute new CatBoost scores.',
            'Shared folds prevent same-owner leakage across the two stackers; unchanged policy, no threshold sweep.',
            'Test probability parity checks samples, not every test pair.'],
        inputs_sha256=bound, seconds=time.monotonic()-started)
    np.savez_compressed(OUT/'decisions.npz', c8_keep=keep['c8'], combined_keep=keep['combined'], scores=fpred,
        **({'individual_ce_keep':keep['individual_ce']} if MULTICE else {}))
    write(OUT/'report.json', report)
    progress('complete', splits=reports)


if __name__ == '__main__':
    run()

"""Read-only peer-artifact verification and coherent-fold key-channel audit.

Does not modify peer weights, scores, outputs, or running jobs. The only writes
are a new project audit directory. Replays existing thresholds and bounded
reverse rivals; this is NOT a full-gallery rival or fresh-holdout evaluation.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from ber.store import connect
from v3_owner_assignment import load_text, text_logit, winner
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap
from v3_v10_combined_audit import model, logit

C = Path('work/ce')
OUT = ROOT / 'artifacts/improvement_v3/kc_union_audit_v1'
STRUCT = ROOT / 'artifacts/improvement_v3/structural_v1'


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = {}

    def track(path):
        path = Path(path)
        h = digest(path)
        assert hashes.get(str(path), h) == h, 'Changed input: ' + str(path)
        hashes[str(path)] = h
        return path

    def raw(path):
        with track(path).open('rb') as f:
            return pickle.load(f)

    def arr(path):
        return np.load(track(path), allow_pickle=False)

    def progress(stage, **kw):
        result = dict(stage=stage, seconds=time.monotonic()-started, **kw)
        write(OUT/'progress.json', result)
        print(json.dumps(result), flush=True)

    def matrix(rows, path, ce):
        x = np.column_stack((arr(str(path)+'.ops.npy'),
            logit([r[4] for r in rows]), logit(ce),
            arr(str(path)+'.nf.npy'), arr(str(path)+'.nf2.npy'))).astype(np.float32)
        assert x.shape == (len(rows), 33)
        assert np.isfinite(np.delete(x, 28, axis=1)).all()
        assert np.isfinite(ce).all() and ((ce >= 0) & (ce <= 1)).all()
        return x

    arrays, records = {}, {'old': [], 'k10': [], 'kc': []}
    mats = {k: [] for k in ('old', 'c8', 'k10', 'kc')}
    counts, newsets = {}, {}
    channel = {(a, t): ch for a, t, ch in raw(C/'keych_devtune.pkl')}
    for sp in ('dev', 'tune'):
        with arr(PILOT/'data'/f'{sp}.npz') as z:
            arrays[sp] = {k: z[k] for k in ('reference_ids','true_counts','countries','left','right')}
        refs = set(arrays[sp]['reference_ids'])
        old = raw(C/f'{sp}.pkl')
        r10 = raw(C/f'{sp}_k10.pkl')
        new = raw(C/f'{sp}_kcnew.pkl')
        rkc = raw(C/f'{sp}_kc.pkl')
        key = lambda rows: [r[:2] for r in rows]
        assert key(rkc) == key(r10) + key(new)
        assert key(r10[:len(old)]) == key(old)
        assert len(set(key(rkc))) == len(rkc)
        assert not (set(key(new)) & set(key(r10)))
        assert all(r[0] in refs and np.isnan(r[4]) for r in new)
        assert all(a[:4]+a[5:] == b[:4]+b[5:] for a,b in zip(rkc, r10+new))
        for suffix in ('ops', 'nf', 'nf2'):
            got = arr(C/f'{sp}_kc.pkl.{suffix}.npy')
            want = np.concatenate((arr(C/f'{sp}_k10.pkl.{suffix}.npy'), arr(C/f'{sp}_kcnew.pkl.{suffix}.npy')))
            assert np.array_equal(got, want, equal_nan=True), (sp, suffix)
        ce = arr(C/f'e5_cont1300k/{sp}_kc_pce.npy')
        want = np.concatenate((arr(C/f'e5_cont1300k/{sp}_k10_pce.npy'), arr(C/f'{sp}_kcnew.pkl.e5_cont1300k.npy')))
        assert np.array_equal(ce, want)
        for name, tag in (('old','e5_cont1300k'),('c8','c8')):
            mats[name].append(matrix(old, C/f'{sp}.pkl', arr(C/f'{tag}/{sp}_pce.npy')))
        mats['k10'].append(matrix(r10, C/f'{sp}_k10.pkl', arr(C/f'e5_cont1300k/{sp}_k10_pce.npy')))
        mats['kc'].append(matrix(rkc, C/f'{sp}_kc.pkl', ce))
        records['old'].extend(old); records['k10'].extend(r10); records['kc'].extend(rkc)
        newsets[sp] = new
        bychannel = defaultdict(Counter)
        for r in new:
            ch = channel[r[:2]]
            bychannel[ch]['rows'] += 1
            bychannel[ch]['true'] += r[2]
        counts[sp] = dict(rows=len(new), true=sum(r[2] for r in new),
            true_fraction=sum(r[2] for r in new)/len(new),
            by_channel={k:dict(v) for k,v in bychannel.items()},
            merged_rows=len(rkc), k10_rows=len(r10))
    assert not (set(arrays['dev']['reference_ids']) & set(arrays['tune']['reference_ids']))
    mats = {k:np.vstack(v) for k,v in mats.items()}
    refs = [str(o) for a in arrays.values() for o in a['reference_ids']]
    db = connect(ROOT/'artifacts/train_store_v2')
    db.execute('PRAGMA cache_size=-8192')
    gold = {o:{r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in refs}
    for sp,a in arrays.items():
        assert [len(gold[str(o)]) for o in a['reference_ids']] == list(a['true_counts'])
    assert all(r[2] == int(r[1] in gold[r[0]]) for r in records['kc'])
    for new in newsets.values():
        for r in new:
            owner = db.execute('SELECT country,name,address FROM anchors WHERE entity_id=?',(r[0],)).fetchone()
            target = db.execute('SELECT country,name,address FROM records WHERE entity_id=?',(r[1],)).fetchone()
            assert tuple(owner) == (r[3],r[5],r[6])
            assert tuple(target) == (r[3],r[7],r[8])
    db.close()

    # Re-verify earlier dictionary claim against the current saved cache.
    cnt = raw(C/'translit_cnt.pkl')
    dictionary = {k:c.most_common(1)[0][0] for k,c in cnt.items()
        if c.most_common(1)[0][1] >= 2 and c.most_common(1)[0][1]/sum(c.values()) >= .4}
    dictionary_check = dict(native_tokens=len(dictionary), english_words=len(set(dictionary.values())),
        alignment_votes=sum(sum(c.values()) for c in cnt.values()),
        word_variants={word:sum(v == word for v in dictionary.values()) for word in ('sky','apex','finance')})

    def metric(mask, rows, sp):
        pred = defaultdict(set)
        for r,k in zip(rows,mask):
            if k:
                pred[r[0]].add(r[1])
        a = arrays[sp]; rr = list(map(str,a['reference_ids'])); t = a['true_counts']
        tp = np.asarray([len(pred[o] & gold[o]) for o in rr])
        n = np.asarray([len(pred[o]) for o in rr]); den = t+4*n
        s = np.divide(5*tp,den,out=np.ones(len(rr)),where=den!=0)
        return s,dict(macro_f05=float(s.mean()),tp=int(tp.sum()),fp=int((n-tp).sum()),fn=int((t-tp).sum()),
            countries={str(c):float(s[a['countries']==c].mean()) for c in np.unique(a['countries'])})

    # Reproduce cached quick comparison before conducting a stronger replay.
    cached = {}
    for key,tag in (('c8','c8'),('k10','e5_cont1300k_k10'),('kc','e5_cont1300k_kc')):
        rr,p = raw(C/f'stack_oof_{tag}.pkl')
        base = records['old' if key == 'c8' else key]
        assert [r[1:3] for r in rr] == [r[:2] for r in base]
        assert len(rr) == len(p) and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        cached[key] = {r[1:3]:float(v) for r,v in zip(rr,p)}
    oldkeys = {r[:2] for r in records['old']}
    densekeys = {r[:2] for r in records['k10']} - oldkeys
    keykeys = {r[:2] for rows in newsets.values() for r in rows}
    assert not (densekeys & keykeys)
    quick = {}
    for name in ('k10','kc_refit_dense','kc_all','k10_plus_key'):
        best = {}
        for r in records['kc']:
            key = r[:2]
            if key in oldkeys:
                p,t = cached['c8'][key],.75
            elif key in densekeys:
                p,t = cached['k10' if name in ('k10','k10_plus_key') else 'kc'][key],.9
            elif name in ('kc_all','k10_plus_key'):
                p,t = cached['kc'][key],.9
            else:
                continue
            if p >= t and (r[1] not in best or p > best[r[1]][1]):
                best[r[1]] = (key,p)
        kept = {key for key,p in best.values()}
        mask = [r[:2] in kept for r in records['kc']]
        quick[name] = {sp:metric(mask,records['kc'],sp)[1] for sp in arrays}
    keyacc = [r for rows in newsets.values() for r in rows if cached['kc'][r[:2]] >= .9]
    input_report = dict(counts=counts,total_new=sum(len(r) for r in newsets.values()),
        true_new=sum(r[2] for rows in newsets.values() for r in rows),
        rows_features_scores_labels_text_verified=True,dictionary=dictionary_check,
        cached_quick_comparison=quick, cached_key_acceptance=dict(rows=len(keyacc),true=sum(r[2] for r in keyacc)),
        caveat='Cached quick comparison lacks score band and text-margin reverse rivals; separate stackers use different grouped folds.')
    write(OUT/'input_checks.json',input_report)
    progress('inputs_and_cached_comparison_verified',**input_report)

    # One shared assignment of owners to folds for ALL model arms.
    old = records['old']; r10 = records['k10']; rkc = records['kc']
    y = np.asarray([r[2] for r in old]); owners = np.asarray([r[0] for r in old])
    folds = list(GroupKFold(5).split(mats['old'],y,owners))
    ofold = {o:k for k,(_,held) in enumerate(folds) for o in owners[held]}
    for o in refs:
        ofold.setdefault(o,int(hashlib.sha256(o.encode()).hexdigest()[:8],16)%5)
    splitfold = {k:np.asarray([ofold[r[0]] for r in records[k]]) for k in ('k10','kc')}
    fold_agreement = {}
    for name in ('k10','kc'):
        rr = records[name]; gg = np.asarray([r[0] for r in rr]); yy = np.asarray([r[2] for r in rr])
        alt = {o:k for k,(_,held) in enumerate(GroupKFold(5).split(mats[name],yy,gg)) for o in gg[held]}
        common = set(alt) & set(owners)
        fold_agreement[name] = dict(shared_owners=len(common),different_fold=sum(alt[o]!=ofold[o] for o in common))

    # Same bounded competitor treatment as the independently audited V10.
    ids = [json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    rawkeys = {(ids[int(l)],ids[int(r)]) for a in arrays.values() for l,r in zip(a['left'],a['right'])}
    del ids
    newix = np.asarray([i for i,r in enumerate(rkc) if r[:2] not in oldkeys])
    newrows = [rkc[i] for i in newix]
    newkeys = {r[:2] for r in newrows}
    rivals = raw(C/'rivals.pkl')
    aux = np.asarray([i for i,r in enumerate(rivals) if r[:2] not in rawkeys and r[:2] not in newkeys])
    oldall = old + [rivals[i] for i in aux]
    allrows = oldall + newrows
    assert len({r[:2] for r in allrows}) == len(allrows)
    am = {}
    for name,tag in (('old','e5_cont1300k'),('c8','c8')):
        rm = matrix(rivals,C/'rivals.pkl',arr(C/f'rivals.pkl.{tag}.npy'))
        am[name] = np.vstack((mats[name],rm[aux]))
    rule = json.loads(track(STRUCT/'owner_eval/policy.json').read_text())['selected']
    tm = load_text(track(STRUCT/'text_scorer.json'))
    text = np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},tm) for r in allrows])
    groups = defaultdict(list)
    for i,r in enumerate(allrows):
        groups[r[1]].append(i)
    groups = [np.asarray(sorted(ix,key=lambda i:allrows[i][0])) for ix in groups.values() if len(ix)>1]
    nold,nbase = len(old),len(oldall)
    focalrows = old + newrows
    focalix = np.r_[np.arange(nold),np.arange(nbase,len(allrows))]
    focalfold = np.asarray([ofold[r[0]] for r in focalrows])
    gate = np.asarray([r[4]>=.01 for r in oldall])
    densemask = np.asarray([r[:2] in densekeys for r in newrows])
    keymask = ~densemask
    index10 = {r[:2]:i for i,r in enumerate(r10)}
    new10ix = np.asarray([index10[r[:2]] for r in newrows if r[:2] in densekeys])

    def decide(bp,dp,enabled):
        p = np.r_[bp,dp]
        keep = np.r_[(bp>=.75)&gate,(dp>=.9)&enabled]
        pre = keep.copy()
        for g in groups:
            ix = g[keep[g]]
            if len(ix)>1:
                chosen = winner(p[ix],text[ix],**rule)
                keep[ix] = False
                if chosen is not None:
                    keep[ix[chosen]] = True
        return keep[focalix],pre[focalix]

    names = ('c8','k10','kc_refit_dense','kc_all','k10_plus_key')
    keep = {n:np.zeros(len(focalrows),bool) for n in names}
    before = {n:np.zeros(len(focalrows),bool) for n in names}
    for k,(fit,held) in enumerate(folds):
        progress('coherent_fold',fold=k)
        assert not (set(owners[fit]) & set(owners[held]))
        op = model().fit(mats['old'][fit],y[fit]).predict_proba(am['old'])[:,1]
        cp = model().fit(mats['c8'][fit],y[fit]).predict_proba(am['c8'])[:,1]
        bp = np.where((op>=.05)&(op<.98),cp,op)
        pp = {}
        for tag in ('k10','kc'):
            f = splitfold[tag] != k
            assert not ({r[0] for r,t in zip(records[tag],f) if t} & {o for o in refs if ofold[o]==k})
            yy = np.asarray([r[2] for r in records[tag]])
            xx = mats[tag][new10ix if tag=='k10' else newix]
            pp[tag] = model().fit(mats[tag][f],yy[f]).predict_proba(xx)[:,1]
        p10 = np.zeros(len(newrows)); p10[densemask] = pp['k10']
        mixed = p10.copy(); mixed[keymask] = pp['kc'][keymask]
        configs = {'c8':(p10,np.zeros(len(newrows),bool)),
            'k10':(p10,densemask),'kc_refit_dense':(pp['kc'],densemask),
            'kc_all':(pp['kc'],np.ones(len(newrows),bool)),
            'k10_plus_key':(mixed,np.ones(len(newrows),bool))}
        mask = focalfold==k
        for name,(p,enabled) in configs.items():
            after,pre = decide(bp,p,enabled)
            keep[name][mask] = after[mask]; before[name][mask] = pre[mask]

    result = {}
    for sp,a in arrays.items():
        ss = {}; met = {}; rr = set(map(str,a['reference_ids']))
        for n in names:
            ss[n],met[n] = metric(keep[n],focalrows,sp)
        transitions = {}
        for n in names:
            if n=='k10':
                continue
            c = Counter()
            for i,r in enumerate(focalrows):
                if r[0] not in rr:
                    continue
                if keep[n][i] and not keep['k10'][i]:
                    c['added_true' if r[2] else 'added_false'] += 1
                if keep['k10'][i] and not keep[n][i]:
                    c['removed_true' if r[2] else 'removed_false'] += 1
                if r[:2] in keykeys and keep[n][i]:
                    c['key_true' if r[2] else 'key_false'] += 1
            transitions[n] = dict(c)
        ceiling = {}
        for tag in ('old','k10','kc'):
            cand = defaultdict(set)
            for r in records[tag]:
                cand[r[0]].add(r[1])
            hit = np.asarray([len(cand[str(o)]&gold[str(o)]) for o in a['reference_ids']])
            t = a['true_counts']; den = t+4*hit
            ceiling[tag] = dict(oracle=float(np.divide(5*hit,den,out=np.ones(len(hit)),where=den!=0).mean()),missed_true=int((t-hit).sum()))
        result[sp] = dict(metrics=met,oracle=ceiling,transitions_vs_k10=transitions,
            delta_vs_k10={n:bootstrap(ss['k10'],ss[n]) for n in ('kc_refit_dense','kc_all','k10_plus_key')})
    for path,h in hashes.items():
        assert digest(path)==h,'Input changed during audit: '+path
    report = dict(complete=True,inputs=input_report,splits=result,original_pairs=nold,
        added_dense_pairs=int(densemask.sum()),added_key_pairs=int(keymask.sum()),bounded_rival_pairs=len(aux),
        shared_fold_difference_from_cached_stacker_folds=fold_agreement,policy=rule,
        base_threshold=.75,new_threshold=.9,old_fallback_band=[.05,.98],
        caveats=['Reused US/India evaluation, not fresh reporting data or a French score.',
            'Same owner folds across all stackers; cached quick scores use different fold assignments.',
            'Original bounded reverse rivals plus all new dev/tune edges, not exhaustive true-owner discovery.',
            'No new threshold tuning. This is validation replay, not a parity check of completed KC test output.',
            'Validation changed scores on ordinary dense edges are separated from effects of added key edges.'],
        input_sha256=hashes,seconds=time.monotonic()-started)
    np.savez_compressed(OUT/'decisions.npz',**{n+'_keep':keep[n] for n in names})
    write(OUT/'report.json',report)
    progress('complete',splits=result)


if __name__=='__main__':
    run()

"""Append key-channel pairs (keych_devtune.pkl) absent from <sp>_k10.pkl as new NaN-p100k rows.
stage new:   writes <sp>_kcnew.pkl (new rows only) for feats_for.py / ce_score.py
stage merge: writes <sp>_kc.pkl + .ops/.nf/.nf2 + e5_cont1300k/<sp>_kc_pce.npy
usage: keych_union.py new|merge"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, pickle, sys
import numpy as np

C = ce_data(); DS = dataset() + 'train/'
A = ce_prep()
stage = sys.argv[1]
if stage == 'new':
    split = {}
    for sp in ('dev', 'tune'):
        for r in np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy'):
            split[r] = sp
    truth = {}
    with open(DS + 'train_ground_truth.tsv') as h:
        for a, b in csv.reader(h, delimiter='\t'):
            if a in split:
                truth[a] = set(b.split(',')) - {''}
    have = set()
    for sp in ('dev', 'tune'):
        have.update((r[0], r[1]) for r in pickle.load(open(C + f'{sp}_k10.pkl', 'rb')))
    P = [(a, t) for a, t, _ in pickle.load(open(C + 'keych_devtune.pkl', 'rb')) if (a, t) not in have]
    need = {t for _, t in P}; s1 = {}; rec = {}
    with open(DS + 'train_source1.tsv') as h:
        for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
            if x[0] in split:
                s1[x[0]] = (x[3], x[1], x[2])
    for s in ('train_source2.tsv', 'train_source3.tsv'):
        with open(DS + s) as h:
            for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
                if x[0] in need:
                    rec[x[0]] = (x[1], x[2])
    for sp in ('dev', 'tune'):
        new = [(a, t, int(t in truth.get(a, ())), s1[a][0], float('nan'), s1[a][1], s1[a][2]) + rec[t]
               for a, t in sorted(P) if split[a] == sp]
        pickle.dump(new, open(C + f'{sp}_kcnew.pkl', 'wb'))
        print(f'{sp}: new {len(new)} true {sum(r[2] for r in new)}')
else:
    def _pce(sp, usuf, pkl):
        for p in (C + f'e5_cont1300k/{sp}{usuf}_pce.npy', pkl + '.e5_cont1300k.npy'):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f'no e5_cont1300k scores for {pkl}')
    os.makedirs(C + 'e5_cont1300k', exist_ok=True)
    for sp in ('dev', 'tune'):
        a, b = pickle.load(open(C + f'{sp}_k10.pkl', 'rb')), pickle.load(open(C + f'{sp}_kcnew.pkl', 'rb'))
        pickle.dump(a + b, open(C + f'{sp}_kc.pkl', 'wb'))
        for s in ('ops', 'nf', 'nf2'):
            np.save(C + f'{sp}_kc.pkl.{s}.npy', np.concatenate([np.load(C + f'{sp}_k10.pkl.{s}.npy'), np.load(C + f'{sp}_kcnew.pkl.{s}.npy')]))
        pce = np.concatenate([np.load(_pce(sp, '_k10', C + f'{sp}_k10.pkl')), np.load(_pce(sp, '_kcnew', C + f'{sp}_kcnew.pkl'))])
        np.save(C + f'e5_cont1300k/{sp}_kc_pce.npy', pce)
        np.save(C + f'{sp}_kc.pkl.e5_cont1300k.npy', pce)
        print(sp, len(a), '+', len(b))

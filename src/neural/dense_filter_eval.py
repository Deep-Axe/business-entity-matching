"""Slice a dense union (<sp>_u10.pkl + features + e5 scores) by rank depth and similarity filter into <sp>_<suf>.pkl.
usage: dense_filter_eval.py SUF TOP T_NA T_N"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import pickle, sys, collections
import numpy as np

C = ce_data()
suf, top, t_na, t_n = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
z = np.load(C + 'dense_devtune.npz'); qid = z['qid']
dec = lambda v: f'S3-{v - 1_000_000_000}' if v >= 1_000_000_000 else f'S2-{v}'
ok = set()
for k in z.files:
    if k.endswith('_ix'):
        v, c = k[:-3].rsplit('_', 1); thr = t_na if v == 'name_addr' else t_n
        for qi, ids, ss in zip(z[k], z[f'{v}_{c}_id'][:, :top], z[f'{v}_{c}_s'][:, :top]):
            for i, s in zip(ids, ss):
                if i >= 0 and s >= thr:
                    ok.add((qid[qi], dec(int(i))))
for sp in ('dev', 'tune'):
    rows = pickle.load(open(C + f'{sp}_u10.pkl', 'rb'))
    keep = np.array([not np.isnan(r[4]) or (r[0], r[1]) in ok for r in rows])
    pickle.dump([r for r, k in zip(rows, keep) if k], open(C + f'{sp}_{suf}.pkl', 'wb'))
    for s in ('ops', 'nf', 'nf2'):
        np.save(C + f'{sp}_{suf}.pkl.{s}.npy', np.load(C + f'{sp}_u10.pkl.{s}.npy')[keep])
    sc = np.load(C + f'{sp}_u10.pkl.e5_cont1300k.npy'); o = np.load(C + f'e5_cont1300k/{sp}_pce.npy'); sc[:len(o)] = o
    np.save(C + f'e5_cont1300k/{sp}_{suf}_pce.npy', sc[keep])
    new = [r for r, k in zip(rows, keep) if k and np.isnan(r[4])]
    print(f'{sp}: new {len(new)} ({len(new) / 3000:.2f}/S1) true {sum(r[2] for r in new)}')

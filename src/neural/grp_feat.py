"""Group-consistency features: how well a candidate target agrees (text scorer) with the S1's other confident targets.
usage: grp_feat.py dev|tune|test [WORKER NWORKERS]   writes {pkl}.grp.npy (test: band rows only, NaN elsewhere)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, glob, pickle, collections
import numpy as np
sys.path.insert(0, 'tools'); sys.path.insert(0, 'code/business_entity_resolution/src')
from v3_owner_assignment import load_text, text_logit

C = ce_data()
tm = load_text('artifacts/improvement_v3/structural_v1/text_scorer.json')
NF = 6

def lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6); return float(np.log(p / (1 - p)))

def feats(rows, need):
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        g[r[0]].append(i)
    F = np.full((len(rows), NF), np.nan, np.float32); memo = {}
    def sim(i, j):
        k = (rows[i][1], rows[j][1]) if rows[i][1] < rows[j][1] else (rows[j][1], rows[i][1])
        if k not in memo:
            memo[k] = text_logit({'name': rows[i][7], 'address': rows[i][8]}, {'name': rows[j][7], 'address': rows[j][8]}, tm)
        return memo[k]
    for i in need:
        sib = [j for j in g[rows[i][0]] if j != i]
        conf = [j for j in sib if rows[j][4] >= 0.5]
        F[i, 5] = len(conf)
        if sib:
            top = max(sib, key=lambda j: rows[j][4]); F[i, 1] = sim(i, top)
            F[i, 4] = max(min(sim(i, j), lg(rows[j][4])) for j in sib)
        if conf:
            s = [sim(i, j) for j in conf]; F[i, 0] = max(s); F[i, 2] = sum(v > 0 for v in s)
            o = [v for j, v in zip(conf, s) if rows[j][1][:2] != rows[i][1][:2]]
            if o:
                F[i, 3] = max(o)
    return F

sp = sys.argv[1]
if sp in ('dev', 'tune', 'rivals'):
    r = pickle.load(open(C + f'{sp}.pkl', 'rb')); np.save(C + f'{sp}.pkl.grp.npy', feats(r, range(len(r)))); print(sp, 'done')
else:
    w, nw = int(sys.argv[2]), int(sys.argv[3])
    bi = np.load(C + 'band_index.npy'); band = collections.defaultdict(list)
    for p, i in bi:
        band[int(p)].append(int(i))
    fs = sorted(glob.glob(C + 'test/part_*.pkl'))
    for f in fs:
        k = int(f[-7:-4])
        if k % nw != w or k not in band:
            continue
        r = pickle.load(open(f, 'rb')); np.save(f + '.grp.npy', feats(r, band[k]))
    print('worker', w, 'done')

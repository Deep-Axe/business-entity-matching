"""Per-S1 expected-F0.5 link selection on the c8 stack OOF, against the global 0.75 threshold.
Isotonic calibration is fitted on one half and applied to the other; truth is sampled from the calibrated probabilities."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import pickle, collections, sys
import numpy as np
from sklearn.isotonic import IsotonicRegression

C = ce_data(); A = ce_prep()
r, o = pickle.load(open(C + 'stack_oof_c8.pkl', 'rb'))
tc, spl = {}, {}
for sp in ('dev', 'tune'):
    ids = np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy')
    tc.update(zip(ids, np.load(A + f'prepared_gate0_full_{sp}_v1/true_counts.npy'))); spl.update({x: sp for x in ids})
best = {}
for x, v in zip(r, o):
    k = x[2]
    if k not in best or v > best[k][1]:
        best[k] = (x, v)
keep = {id(x) for x, v in best.values()}

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

y = np.array([x[3] for x in r]); s = np.array(o); sp_of = np.array([x[0] for x in r])
cal = {}
for fit, app in (('tune', 'dev'), ('dev', 'tune')):
    m = sp_of == fit; iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(s[m], y[m]); cal[app] = iso
by = collections.defaultdict(list)
for i, x in enumerate(r):
    if id(x) in keep:
        by[x[1]].append(i)
rng = np.random.default_rng(0)
MISS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.11

def choose(ix, p):
    order = np.argsort(-p); ix, p = np.array(ix)[order][:12], p[order][:12]
    S = 400; draws = rng.random((S, len(p))) < p; extra = rng.poisson(MISS, S)
    t = draws.sum(1) + extra; best_k, best_e = 0, -1
    for k in range(len(p) + 1):
        tp = draws[:, :k].sum(1)
        e = np.mean([f05(a, k, b) for a, b in zip(tp, t)])
        if e > best_e:
            best_k, best_e = k, e
    return set(ix[:best_k].tolist())

res = collections.defaultdict(lambda: collections.defaultdict(list))
for o_id, ix in by.items():
    sp = spl[o_id]; p = cal[sp].predict(s[ix])
    acc_t = {i for i in ix if s[i] >= 0.75}; acc_e = choose(ix, p)
    for name, acc in (('t0.75', acc_t), ('expF', acc_e)):
        tp = sum(y[i] for i in acc); res[sp][name].append(f05(tp, len(acc), tc[o_id]))
for sp in ('tune', 'dev'):
    n0 = sum(1 for o_id in spl if spl[o_id] == sp and o_id not in by)
    for name, v in res[sp].items():
        print(sp, name, round((sum(v) + n0 * 1.0 * 0 + sum(1.0 if tc[o] == 0 else 0.0 for o in spl if spl[o] == sp and o not in by)) / (len(v) + n0), 5))

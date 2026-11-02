import numpy as np, sys
from sklearn.isotonic import IsotonicRegression

B = 'artifacts/improvement_v2/cpu/'
def load(split, pfile):
    d = B + f'prepared_gate0_full_{split}_v1/'
    y = np.load(d + 'y.npy').astype(int)
    q = np.load(d + 'query_ids.npy')
    t = np.load(d + 'true_counts.npy')
    p = np.load(B + 'fit_gate0_100k_v1/' + pfile).astype(float)
    return y, q, t, p

def score_sets(y, q, t, chosen):
    n = len(t)
    tp = np.bincount(q, weights=(chosen & (y == 1)), minlength=n)
    k = np.bincount(q, weights=chosen, minlength=n)
    f = np.where(k + t == 0, 1.0, 1.25 * tp / np.maximum(1.25 * tp + 0.25 * (t - tp) + (k - tp), 1e-9))
    return f.mean(), int(tp.sum()), int((k - tp).sum())

def expected_decode(q, p, n, draws=2000, miss=0.0, rng=np.random.default_rng(0)):
    chosen = np.zeros(len(p), bool)
    order = np.lexsort((-p, q))
    starts = np.searchsorted(q[order], np.arange(n))
    ends = np.searchsorted(q[order], np.arange(n), side='right')
    for o in range(n):
        idx = order[starts[o]:ends[o]]
        idx = idx[p[idx] > 1e-4]
        if len(idx) == 0:
            continue
        pp = p[idx]
        s = rng.random((draws, len(pp))) < pp
        G = s.sum(1) + (rng.poisson(miss, draws) if miss else 0)
        best, bestk = (G == 0).mean(), 0
        ctp = np.cumsum(s, 1)
        for k in range(1, len(pp) + 1):
            tp = ctp[:, k - 1]
            ef = (1.25 * tp / (tp + 0.25 * G + k)).mean()
            if ef > best:
                best, bestk = ef, k
        chosen[idx[:bestk]] = True
    return chosen

ty, tq, tt, tp_ = load('tune', 'enhanced_full_tune_probabilities.npy')
dy, dq, dt, dp = load('dev', 'enhanced_dev_probabilities.npy')
print('tune owners', len(tt), 'singleton', (tt == 0).mean(), 'dev singleton', (dt == 0).mean())
for name, y, q, t, p in [('tune', ty, tq, tt, tp_), ('dev', dy, dq, dt, dp)]:
    print(name, 'threshold0.76', score_sets(y, q, t, p >= 0.76))
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(tp_, ty)
for miss in [0.0, 0.05]:
    for name, y, q, t, p in [('tune', ty, tq, tt, tp_), ('dev', dy, dq, dt, dp)]:
        print(name, 'expectedF raw miss', miss, score_sets(y, q, t, expected_decode(q, p, len(t), miss=miss)))
    print('dev expectedF isotonic(tune) miss', miss, score_sets(dy, dq, dt, expected_decode(dq, iso.predict(dp), len(dt), miss=miss)))

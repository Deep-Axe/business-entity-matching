"""Expected public-score change of sub_v7_fr090 vs sub_v6_cC_band, simulating French truth from dev/tune calibration."""
import glob, pickle, csv, collections
import numpy as np

def read(p):
    with open(p) as h:
        r = csv.reader(h, delimiter='\t'); next(r)
        return {a: set(b.split(',')) - {''} for a, b in r}

A = read('ce/sub_v6_cC_band/matching_results.tsv'); B = read('ce/sub_v7_fr090/matching_results.tsv')
N = len(A); ch = {k for k in A if A[k] != B[k]}
print(f'S1s {N}  changed {len(ch)}  pairs dropped {sum(len(A[k] - B[k]) for k in ch)}  added {sum(len(B[k] - A[k]) for k in ch)}')

rows, oof = pickle.load(open('ce/stack_oof_cC.pkl', 'rb'))
y = np.array([r[3] for r in rows]); edges = np.array([0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95, .98, 1.01])
b = np.digitize(oof, edges) - 1
cal = np.array([y[b == i].mean() if (b == i).any() else 0 for i in range(len(edges) - 1)])
print('dev calibration: ' + ' '.join(f'{edges[i]:.2f}:{cal[i]:.2f}' for i in range(len(cal))))

cand = collections.defaultdict(list)
for f in sorted(glob.glob('ce/test/part_*.pkl')):
    rs = pickle.load(open(f, 'rb')); s = np.load(f + '.stack_cC_band.npy')
    for r, v in zip(rs, s):
        if r[0] in ch:
            cand[r[0]].append((r[1], float(v)))
    del rs

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

rng = np.random.default_rng(0)
for m in (1.0, 0.5, 0.3, 0.2):
    tot = 0.0
    for k in ch:
        ids = [c[0] for c in cand[k]]; p = cal[np.digitize([c[1] for c in cand[k]], edges) - 1]
        o = p / (1 - p + 1e-9) * m; p = o / (1 + o)
        a = np.array([i in A[k] for i in ids]); bb = np.array([i in B[k] for i in ids])
        T = rng.random((400, len(ids))) < p; t = T.sum(1) + rng.poisson(0.06, 400)
        fa = [f05((T[j] & a).sum(), a.sum(), t[j]) for j in range(400)]
        fb = [f05((T[j] & bb).sum(), bb.sum(), t[j]) for j in range(400)]
        tot += np.mean(fb) - np.mean(fa)
    print(f'French odds x{m}: expected public change {tot / N:+.5f}')

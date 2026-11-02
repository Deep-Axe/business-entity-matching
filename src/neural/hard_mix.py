"""Hard-example training mix from training pairs only (never dev/tune): rule-defined strata + p100k-hard + random slice.
usage: hard_mix.py OUT.pkl"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, glob, pickle, re, collections
import numpy as np

C = ce_data()
rng = np.random.default_rng(11)
tok = lambda s: set(re.findall(r'\w+', s.lower()))
num = lambda a: (re.findall(r'\d+', a) or [None])[0]
CAP = {'noaddr_pos': 60000, 'tradename_pos': 60000, 'coloc_neg': 60000, 'samename_neg': 60000, 'p100k_hard': 80000, 'random': 100000}
pools = collections.defaultdict(list); n = 0
for f in sorted(glob.glob(C + 'train_*.pkl')):
    for r in pickle.load(open(f, 'rb')):
        n += 1; y = r[2]; na, nb = tok(r[5]), tok(r[7]); ta = tok(r[8]); aa = tok(r[6])
        nj = len(na & nb) / max(1, len(na | nb)); aj = len(aa & ta) / max(1, len(aa | ta))
        s = []
        if y == 1 and len(ta) <= 1:
            s.append('noaddr_pos')
        if y == 1 and nj < 0.2 and aj >= 0.4:
            s.append('tradename_pos')
        if y == 0 and aj >= 0.5 and num(r[6]) is not None and num(r[6]) == num(r[8]):
            s.append('coloc_neg')
        if y == 0 and nj >= 0.6:
            s.append('samename_neg')
        if (y == 1 and r[4] < 0.5) or (y == 0 and r[4] > 0.2):
            s.append('p100k_hard')
        for k in s:
            pools[k].append(r)
        if rng.random() < 0.03:
            pools['random'].append(r)
out, seen = [], set()
for k, cap in CAP.items():
    v = pools[k]; pick = rng.choice(len(v), min(cap, len(v)), replace=False) if v else []
    added = 0
    for i in pick:
        key = (v[i][0], v[i][1])
        if key not in seen:
            seen.add(key); out.append(v[i]); added += 1
    print(f'{k}: pool {len(v)} added {added}')
rng.shuffle(out)
print(f'train pairs scanned {n}  mix {len(out)}  positive rate {np.mean([r[2] for r in out]):.3f}  countries {collections.Counter(r[3] for r in out)}')
pickle.dump(out, open(sys.argv[1], 'wb'))

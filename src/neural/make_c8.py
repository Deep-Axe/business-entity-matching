"""Mean-logit ensemble c8 = e5_cont1300k + kA..kG.
On the test band (band_index.npy) the eight member scores are averaged;
outside the band, test .c8.npy equals the E5-small probability.
Also writes c8/{dev,tune}_pce.npy, {dev,tune}.pkl.c8.npy and rivals.pkl.c8.npy
when those member scores exist.
usage: make_c8.py
Set ALLOW_PARTIAL=1 to average whatever members are present (not V12).
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import band_chunk_paths, ce_data

C = ce_data()
MEMBERS = ['e5_cont1300k', 'kA', 'kB', 'kC', 'kD', 'kE', 'kF', 'kG']
idx = np.load(C + 'band_index.npy')
n = len(idx)
partial = os.environ.get('ALLOW_PARTIAL') == '1'

def lg(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sig(x):
    return (1 / (1 + np.exp(-x))).astype(np.float32)

def load_band(tag):
    paths = band_chunk_paths(C + f'scores_{tag}/band_test_*.pkl.{tag}.npy') or band_chunk_paths(C + f'band_test_*.pkl.{tag}.npy')
    if paths:
        v = np.concatenate([np.load(p) for p in paths])
        if len(v) != n:
            raise SystemExit(f'{tag}: {len(v)} band scores vs band {n}')
        return v
    if tag == 'e5_cont1300k':
        out = np.empty(n, np.float32)
        for f in sorted(glob.glob(C + 'test/part_*.pkl')):
            part = int(f[-7:-4]); s = np.load(f + f'.{tag}.npy')
            m = idx[:, 0] == part
            out[m] = s[idx[m, 1]]
        return out
    raise FileNotFoundError(tag)

def first_score(tag, *cands):
    for p in cands:
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(tag)

found, band_lg = [], []
missing = []
for tag in MEMBERS:
    try:
        p = load_band(tag); found.append(tag); band_lg.append(lg(p))
        print(f'band {tag}', flush=True)
    except FileNotFoundError:
        missing.append(tag); print(f'skip band {tag}', flush=True)
if 'e5_cont1300k' not in found:
    raise SystemExit('e5_cont1300k scores are required')
if missing and not partial:
    raise SystemExit('missing band scores for ' + ','.join(missing) + ' (retrain with scripts/train_encoders.sh, or ALLOW_PARTIAL=1)')
blend = sig(np.mean(np.stack(band_lg, 0), 0))
print('c8 members', found, 'band', n, flush=True)
for f in sorted(glob.glob(C + 'test/part_*.pkl')):
    part = int(f[-7:-4]); small = np.load(f + '.e5_cont1300k.npy')
    out = small.astype(np.float32).copy()
    m = idx[:, 0] == part
    out[idx[m, 1]] = blend[m]
    np.save(f + '.c8.npy', out)
print('wrote test .c8.npy', flush=True)

os.makedirs(C + 'c8', exist_ok=True)
for sp in ('dev', 'tune'):
    try:
        mats = [lg(first_score(tag, C + f'{tag}/{sp}_pce.npy', C + f'{sp}.pkl.{tag}.npy')) for tag in found]
    except FileNotFoundError as e:
        print(f'skip {sp} c8 ({e})', flush=True); continue
    v = sig(np.mean(np.stack(mats, 0), 0))
    np.save(C + f'c8/{sp}_pce.npy', v)
    np.save(C + f'{sp}.pkl.c8.npy', v)
    print(f'wrote {sp} c8 {len(v)}', flush=True)
try:
    mats = [lg(first_score(tag, C + f'rivals.pkl.{tag}.npy', C + f'scores_{tag}/rivals.pkl.{tag}.npy')) for tag in found]
    v = sig(np.mean(np.stack(mats, 0), 0))
    np.save(C + 'rivals.pkl.c8.npy', v)
    print('wrote rivals c8', flush=True)
except FileNotFoundError as e:
    print(f'skip rivals c8 ({e})', flush=True)

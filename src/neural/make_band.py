"""Build the uncertain band of original test candidates for E5-base scoring.
Reads test/part_*.pkl.stack_e5_cont1300k_nocty_nf2.npy (or STACK env suffix).
Writes CE_DATA/band_index.npy as (part, row) and CE_DATA/band_test_{0..}.pkl row chunks.
usage: make_band.py [LO HI]   default lo=0.05 hi=0.98
"""
import glob, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data

C = ce_data()
lo, hi = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.05, 0.98)
suf = os.environ.get('STACK', 'stack_e5_cont1300k_nocty_nf2')
idx, chunks, buf = [], [], []
CH = 250_000
for f in sorted(glob.glob(C + 'test/part_*.pkl')):
    s = np.load(f + f'.{suf}.npy'); part = int(f[-7:-4])
    rows = pickle.load(open(f, 'rb'))
    keep = np.flatnonzero((s >= lo) & (s < hi))
    for i in keep:
        idx.append((part, int(i))); buf.append(rows[int(i)])
        if len(buf) >= CH:
            chunks.append(buf); buf = []
if buf:
    chunks.append(buf)
np.save(C + 'band_index.npy', np.array(idx, np.int32))
for j, ch in enumerate(chunks):
    pickle.dump(ch, open(C + f'band_test_{j}.pkl', 'wb'))
print(f'band [{lo}, {hi}) rows {len(idx)} chunks {len(chunks)}', flush=True)

"""Merge band rescoring into full test score arrays.
Usage: band_merge.py OLD_SUFFIX BAND_SUFFIX OUT_SUFFIX
For each test part: rows listed in ce/band_index.npy take <part>.BAND_SUFFIX.npy, every other row keeps <part>.OLD_SUFFIX.npy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, glob
import numpy as np

C = ce_data()
old, band, out = sys.argv[1:4]
idx = np.load(C + 'band_index.npy')
n_band = changed = 0
for f in sorted(glob.glob(C + 'test/part_*.pkl')):
    p = int(f[-7:-4]); rows = idx[idx[:, 0] == p, 1]
    o = np.load(f'{f}.{old}.npy').astype(np.float32); b = np.load(f'{f}.{band}.npy').astype(np.float32)
    assert len(o) == len(b) and not np.isnan(b[rows]).any()
    s = o.copy(); s[rows] = b[rows]
    keep = np.ones(len(s), bool); keep[rows] = False
    assert np.array_equal(s[keep], o[keep])
    n_band += len(rows); changed += int((s != o).sum())
    np.save(f'{f}.{out}.npy', s)
print('band rows', n_band, 'changed', changed)

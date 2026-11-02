"""Compute the stack's row features (.ops/.nf/.nf2) for arbitrary row pickles.
usage: feats_for.py train|test PKL...
"""
import sys, os, pickle
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from ber_paths import store as store_path
from editops import feats
from editops2 import offset_feats
from name_freq import freq as freq_nf, run as run_nf
from name_freq2 import freq as freq_nf2, run as run_nf2

store, files = sys.argv[1], sys.argv[2:]
if not files:
    raise SystemExit('usage: feats_for.py train|test PKL...')
for f in files:
    if not os.path.exists(f + '.ops.npy'):
        rows = pickle.load(open(f, 'rb'))
        np.save(f + '.ops.npy', np.array([feats(r[5], r[6], r[7], r[8]) + offset_feats(r[6], r[8]) for r in rows], np.float32))
        print(f, 'ops', flush=True)
st = store_path(store)
run_nf(files, freq_nf(st))
run_nf2(files, freq_nf2(st))

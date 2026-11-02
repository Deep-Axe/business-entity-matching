"""Edit-operation + signed-offset features for row pickles; writes <pkl>.ops.npy (skips existing)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, pickle, os, glob
import numpy as np
exec(open('src/neural/editops.py').read().split('con = sqlite3.connect')[0])
exec(open('src/neural/editops2.py').read().split('con = sqlite3.connect')[0].replace('from editops import feats', ''))
C = ce_data()
files = [C + 'dev.pkl', C + 'tune.pkl', C + 'rivals.pkl'] + sorted(glob.glob(C + 'test/part_*.pkl'))
if len(sys.argv) > 2:
    k, m = int(sys.argv[1]), int(sys.argv[2]); files = files[k::m]
for f in files:
    out = f + '.ops.npy'
    if os.path.exists(out):
        continue
    rows = pickle.load(open(f, 'rb'))
    F = np.array([feats(r[5], r[6], r[7], r[8]) + offset_feats(r[6], r[8]) for r in rows], np.float32)
    np.save(out + '.tmp.npy', F); os.replace(out + '.tmp.npy', out)
    print(os.path.basename(f), F.shape, flush=True)

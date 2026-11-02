"""Full-name frequency (legal form kept): S1 anchors sharing the target's normalised full name; plus whether the focal
S1 has that exact normalised name. Writes <pkl>.nf3.npy = [log1p(count), focal_exact]."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import pickle, glob, os, sqlite3, sys, collections, re, unicodedata
import numpy as np
C = ce_data()
P = re.compile(r'[^a-z0-9]+')

def norm(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    return P.sub(' ', s).strip()

def freq(store):
    f = collections.Counter()
    for n, c in sqlite3.connect(f'file:{store}?mode=ro', uri=True).execute('select name, country from anchors'):
        f[(c, norm(n))] += 1
    return f

def run(files, f):
    for p in files:
        out = p + '.nf3.npy'
        if os.path.exists(out):
            continue
        rows = pickle.load(open(p, 'rb'))
        F = np.array([[np.log1p(f[(r[3], norm(r[7]))]), float(norm(r[5]) == norm(r[7]))] for r in rows], np.float32)
        np.save(out + '.tmp.npy', F); os.replace(out + '.tmp.npy', out); print(os.path.basename(p), F.shape, flush=True)

if '--test' in sys.argv:
    run(sorted(glob.glob(C + 'test/part_*.pkl')), freq('artifacts/test_store_v2/records.sqlite3'))
else:
    run([C + 'dev.pkl', C + 'tune.pkl', C + 'rivals.pkl'], freq('artifacts/train_store_v2/records.sqlite3'))

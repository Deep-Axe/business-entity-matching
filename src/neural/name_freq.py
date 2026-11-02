"""Name-frequency features: S1 records (same store, same country) sharing the core name.
Writes <pkl>.nf.npy with [log1p(freq S1 core), log1p(freq target core)]."""
import pickle, glob, os, sqlite3, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, store
from name_unique_lib import core

C = ce_data()

def freq(store_path):
    f = collections.Counter()
    for n, c in sqlite3.connect(f'file:{store_path}?mode=ro', uri=True).execute('select name, country from anchors'):
        f[(c, ' '.join(core(n)))] += 1
    return f

def run(files, f):
    for p in files:
        out = p + '.nf.npy'
        if os.path.exists(out):
            continue
        rows = pickle.load(open(p, 'rb'))
        F = np.array([[np.log1p(f[(r[3], ' '.join(core(r[5])))]), np.log1p(f[(r[3], ' '.join(core(r[7])))])] for r in rows], np.float32)
        np.save(out + '.tmp.npy', F); os.replace(out + '.tmp.npy', out); print(os.path.basename(p), F.shape, flush=True)

if __name__ == '__main__':
    if '--test' in sys.argv:
        run(sorted(glob.glob(C + 'test/part_*.pkl')), freq(store('test')))
    else:
        run([C + 'dev.pkl', C + 'tune.pkl', C + 'rivals.pkl'], freq(store('train')))

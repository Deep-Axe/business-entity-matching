"""Address-frequency features (unlabelled, per store and country): how many S1 anchors / S2-S3 records share the
(house number, first street word) key of the S1 address / of the target address. Writes <pkl>.af.npy (4 columns)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import pickle, glob, os, sqlite3, sys, collections, re, unicodedata
import numpy as np
C = ce_data()
NUM = re.compile(r'\d+'); WORD = re.compile(r'[a-z]{3,}')
STOP = {'rue', 'street', 'road', 'avenue', 'the', 'unit', 'floor', 'near', 'opp', 'plot', 'house', 'flat', 'shop', 'drive',
        'lane', 'boulevard', 'allee', 'impasse', 'chemin', 'place', 'route', 'block', 'sector', 'nagar', 'main', 'cross'}

def key(a):
    a = unicodedata.normalize('NFKD', a or '').encode('ascii', 'ignore').decode().lower()
    n = NUM.findall(a); w = [x for x in WORD.findall(a) if x not in STOP]
    return (n[0] if n else '', w[0] if w else '')

def freq(store, table):
    f = collections.Counter()
    for a, c in sqlite3.connect(f'file:{store}?mode=ro', uri=True).execute(f'select address, country from {table}'):
        if a:
            f[(c,) + key(a)] += 1
    return f

def run(files, fa, fr):
    for p in files:
        out = p + '.af.npy'
        if os.path.exists(out):
            continue
        rows = pickle.load(open(p, 'rb'))
        F = []
        for r in rows:
            ka, kb = (r[3],) + key(r[6]), (r[3],) + key(r[8])
            F.append([np.log1p(fa[ka]) if r[6] else -1, np.log1p(fr[ka]) if r[6] else -1,
                      np.log1p(fa[kb]) if r[8] else -1, np.log1p(fr[kb]) if r[8] else -1])
        F = np.array(F, np.float32); np.save(out + '.tmp.npy', F); os.replace(out + '.tmp.npy', out); print(os.path.basename(p), F.shape, flush=True)

if '--test' in sys.argv:
    s = 'artifacts/test_store_v2/records.sqlite3'; run(sorted(glob.glob(C + 'test/part_*.pkl')), freq(s, 'anchors'), freq(s, 'records'))
else:
    s = 'artifacts/train_store_v2/records.sqlite3'; run([C + 'dev.pkl', C + 'tune.pkl', C + 'rivals.pkl'], freq(s, 'anchors'), freq(s, 'records'))

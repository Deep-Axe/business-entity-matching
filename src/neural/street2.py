"""Street-name agreement between owner and target: 'same' / 'diff' / 'none' (target has no street).
Reference: dev/tune accepted pairs by label (US/India). Test: V12 accepted links by country."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, pickle, re, unicodedata, collections, sys
import numpy as np

C = ce_data()
TYPES = set('rue r avenue av ave aven boulevard bd blvd chemin chem ch allee all place pl impasse imp route rte quai cours crs square sq '
            'street st road rd drive dr lane ln way court ct parkway pkwy circle cir terrace ter highway hwy trail trl place '
            'residence res lieu dit'.split())
STOP = set('de la du des le les l d the of et en sur bis ter nº no n'.split())

def norm(s):
    return unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()

def street(addr):
    for seg in norm(addr).split(','):
        w = re.findall(r'[a-z]+|\d+', seg)
        if any(t in TYPES for t in w) or (w and w[0].isdigit()):
            s = [t for t in w if t.isalpha() and t not in TYPES and t not in STOP and len(t) >= 2]
            if s:
                return s
    return None

def close(a, b):
    if a == b or (len(a) >= 4 and (b.startswith(a) or a.startswith(b))):
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75

def agree(a1, a2):
    s, t = street(a1), street(a2)
    if not t:
        return 'none'
    if not s:
        return 'nos1'
    hit = sum(any(close(x, y) for y in s) for x in t) / len(t)
    return 'same' if hit >= 0.5 else 'diff'

def dist(v):
    c = collections.Counter(v); n = len(v)
    return {k: round(c[k] / n, 4) for k in ('same', 'diff', 'none', 'nos1')} | {'n': n}

if __name__ == '__main__':
    r, o = pickle.load(open(C + 'stack_oof_c8.pkl', 'rb'))
    ref = collections.defaultdict(list)
    for x, v in zip(r, o):
        if v >= 0.75:
            ref[(x[4], x[3])].append(agree(x[7], x[9]))
    for k in sorted(ref):
        print('dev accepted', k, dist(ref[k]))
    own = set()
    with open(C + 'sub_v12_c8_dense10_keych/matching_results.tsv') as h:
        rr = csv.reader(h, delimiter='\t'); next(rr)
        for a, b in rr:
            for t in b.split(','):
                if t:
                    own.add((a, t))
    te = collections.defaultdict(list); ex = []
    for f in sorted(glob.glob(C + 'test/part_*.pkl')) + sorted(glob.glob(C + 'test_dense_k10/*.pkl')):
        for x in pickle.load(open(f, 'rb')):
            if (x[0], x[1]) in own:
                g = agree(x[6], x[8]); te[x[3]].append(g)
                if x[3] == 'France' and g == 'diff' and len(ex) < 30000:
                    ex.append((x[0], x[1], x[5], x[6], x[7], x[8]))
    for c in ('US', 'India', 'France'):
        print('test V12 accepted', c, dist(te[c]))
    pickle.dump(ex, open('/tmp/fr_diff.pkl', 'wb'))

"""Street-agreement audit: count target-address words that do not appear (even fuzzily) in the owner's address.
Reference = dev/tune true pairs (US/India generator noise); compared with V12's accepted French links on test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, pickle, re, sys, unicodedata, collections
import numpy as np

C = ce_data()
GEN = set('rue avenue ave boulevard blvd chemin chem allee place impasse route quai cours square parvis rond point lieu dit '
          'street road drive lane court suite floor unit apt nagar marg bis ter des del les the and null france region '
          'north south east west main near opp'.split())

def toks(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    return [t for t in re.findall(r'[a-z]+', s) if len(t) >= 3 and t not in GEN]

def close(a, b):
    if a == b or (len(a) >= 4 and (b.startswith(a) or a.startswith(b))):
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    d = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return d <= 1 or (len(a) == len(b) and sorted(a) == sorted(b) and d <= 2)

def novel(s1addr, taddr):
    ref = set(toks(s1addr)); tt = toks(taddr)
    if not tt:
        return -1
    return sum(1 for t in set(tt) if not any(close(t, r) for r in ref))

def summary(vals):
    v = np.array(vals); n = len(v)
    return {'n': n, 'no_street_words': round(np.mean(v == -1), 4), '0': round(np.mean(v == 0), 4),
            '1': round(np.mean(v == 1), 4), '2+': round(np.mean(v >= 2), 4)}

if __name__ == '__main__':
    r, o = pickle.load(open(C + 'stack_oof_c8.pkl', 'rb'))
    ref = collections.defaultdict(list); neg = collections.defaultdict(list)
    for x, v in zip(r, o):
        if v >= 0.75:
            (ref if x[3] else neg)[x[4]].append(novel(x[7], x[9]))
    for c in ('US', 'India'):
        print(f'dev accepted TRUE  {c:6s}', summary(ref[c]))
        print(f'dev accepted FALSE {c:6s}', summary(neg[c]))
    own = {}
    with open(C + 'sub_v12_c8_dense10_keych/matching_results.tsv') as h:
        rr = csv.reader(h, delimiter='\t'); next(rr)
        for a, b in rr:
            for t in b.split(','):
                if t:
                    own[(a, t)] = 1
    te = collections.defaultdict(list); ex = []
    for f in sorted(glob.glob(C + 'test/part_*.pkl')) + sorted(glob.glob(C + 'test_dense_k10/*.pkl')):
        for x in pickle.load(open(f, 'rb')):
            if (x[0], x[1]) in own:
                k = novel(x[6], x[8]); te[x[3]].append(k)
                if x[3] == 'France' and k >= 1 and len(ex) < 4000:
                    ex.append((k, x[5], x[6], x[7], x[8]))
    for c in ('US', 'India', 'France'):
        print(f'test V12 accepted  {c:6s}', summary(te[c]))
    pickle.dump(ex, open('/tmp/fr_novel_examples.pkl', 'wb'))

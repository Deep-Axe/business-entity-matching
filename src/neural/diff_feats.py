"""Binary appended/dropped-name-token and legal-form-change indicators for a fixed vocabulary; writes <pkl>.dtok.npy.
Vocabulary is built once (unlabelled document frequency over dev+tune candidate pairs) and saved."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, pickle, os, glob, re, unicodedata, json, collections
import numpy as np
from rapidfuzz import fuzz
exec(open('src/neural/editops.py').read().split('con = sqlite3.connect')[0])
C = ce_data()

def toks(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.findall(r'[a-z]+', s)

def diff_tokens(an, bn):
    A, B = toks(an), toks(bn); sA, sB = set(A), set(B)
    ua = [t for t in A if t not in sB and not any(fuzz.ratio(t, u) >= 75 for u in B)]
    ub = [t for t in B if t not in sA and not any(fuzz.ratio(t, u) >= 75 for u in A)]
    la = sorted({t for t in A if t in LEGAL}); lb = sorted({t for t in B if t in LEGAL})
    out = {'+' + t for t in ub} | {'-' + t for t in ua}
    if la != lb:
        out.add('legal:' + '_'.join(la) + '>' + '_'.join(lb))
    return out

V = C + 'dtok_vocab.json'
if not os.path.exists(V):
    df = collections.Counter()
    for sp in ('dev', 'tune'):
        for r in pickle.load(open(C + f'{sp}.pkl', 'rb')):
            df.update(diff_tokens(r[5], r[7]))
    json.dump([t for t, c in df.most_common() if c >= 20], open(V, 'w'))
vocab = {t: i for i, t in enumerate(json.load(open(V)))}
files = [C + 'dev.pkl', C + 'tune.pkl', C + 'rivals.pkl'] + sorted(glob.glob(C + 'test/part_*.pkl'))
if len(sys.argv) > 2:
    k, m = int(sys.argv[1]), int(sys.argv[2]); files = files[k::m]
for f in files:
    out = f + '.dtok.npy'
    if os.path.exists(out):
        continue
    rows = pickle.load(open(f, 'rb')); F = np.zeros((len(rows), len(vocab)), np.uint8)
    for i, r in enumerate(rows):
        for t in diff_tokens(r[5], r[7]):
            j = vocab.get(t)
            if j is not None:
                F[i, j] = 1
    np.save(out + '.tmp.npy', F); os.replace(out + '.tmp.npy', out)
    print(os.path.basename(f), F.shape, int(F.sum()), flush=True)

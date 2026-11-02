"""Two exact-key recovery channels for records BM25/dense miss.
translit: native-script (Indic) target names mapped word-by-word to Latin with a dictionary learned from training true pairs
          (dev/tune owners excluded), then matched to S1s with the same core name.
web:      website/handle-style target names reduced to a stem, matched to joined/acronym keys of S1 names.
usage: key_channels.py devtune|test OUT.pkl   (OUT = list of (s1, target, channel))"""
import csv, re, sys, pickle, collections, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_unique_lib import core, LEGAL
from ber_paths import dataset, ce_prep

DS = dataset(); A = ce_prep()
split, out = sys.argv[1], sys.argv[2]
NAT = re.compile(r'[\u0900-\u0DFF]'); WEB = re.compile(r'\.(com|in|net|org|co|fr)\b|^@|www\.')
MAXHIT = 20

def rows(path):
    with open(path, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        yield from r

def toks(s):
    return [t for t in re.split(r'[\s,()\[\]/&]+', s) if t]

held = set()
for sp in ('dev', 'tune'):
    held.update(np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy').tolist())

s1n = {x[0]: x[1] for x in rows(DS + 'train/train_source1.tsv') if x[3] == 'India' and x[0] not in held}
nat_train = {}
for s in ('train_source2.tsv', 'train_source3.tsv'):
    for x in rows(DS + 'train/' + s):
        if x[3] == 'India' and NAT.search(x[1]):
            nat_train[x[0]] = x[1]
cnt = collections.defaultdict(collections.Counter)
with open(DS + 'train/train_ground_truth.tsv') as h:
    for a, b in csv.reader(h, delimiter='\t'):
        if a not in s1n:
            continue
        la = toks(s1n[a])
        for t in b.split(','):
            if t in nat_train:
                nb = toks(nat_train[t])
                if len(nb) == len(la):
                    for p, q in zip(nb, la):
                        if NAT.search(p):
                            cnt[p][q.lower()] += 1
D = {}
for p, c in cnt.items():
    q, n = c.most_common(1)[0]
    if n >= 2 and n / sum(c.values()) >= 0.4:
        D[p] = q
print('dictionary entries', len(D), flush=True)
del s1n, nat_train, cnt

def translit(name):
    out = []
    for t in toks(name):
        if NAT.search(t):
            if t in D:
                out.append(D[t])
        else:
            out.append(t)
    return core(' '.join(out))

def stem(name):
    s = name.lower().strip()
    s = re.sub(r'^(smt|m/s|shri)\s+', '', s); s = re.sub(r'^@', '', s); s = re.sub(r'^www\.', '', s)
    s = re.split(r'\.(?:com|in|net|org|co|fr)\b', s)[0]
    s = re.split(r'\s+-\s+', s)[0]
    return re.sub(r'[^a-z0-9]', '', s)

def s1_keys(name):
    w = re.findall(r'[a-z0-9]+', name.lower())
    nl = [t for t in w if t not in LEGAL]
    ks = set()
    for ws in (w, nl):
        if not ws:
            continue
        ks.add(''.join(ws))
        for k in range(2, len(ws)):
            ks.add(''.join(ws[:k]))
        for m in range(1, len(ws)):
            ks.add(''.join(x[0] for x in ws[:m]) + ''.join(ws[m:]))
        if len(ws) >= 3:
            ks.add(''.join(x[0] for x in ws))
    return {k for k in ks if len(k) >= 4}

sub = 'train' if split == 'devtune' else 'test'
if split == 'devtune':
    q = {x[0]: (x[1], x[3], x[2]) for x in rows(DS + 'train/train_source1.tsv') if x[0] in held}
else:
    q = {x[0]: (x[1], x[3], x[2]) for x in rows(DS + 'test/test_source1.tsv')}
def atoks(s):
    return frozenset(re.findall(r'[a-z0-9]+', s.lower())) - {'india', 'null', 'road', 'no', 'floor', 'nagar'}

tkey = collections.defaultdict(list); wkey = collections.defaultdict(list); gaddr = {}
for s in (f'{sub}_source2.tsv', f'{sub}_source3.tsv'):
    for x in rows(DS + f'{sub}/' + s):
        k = st = None
        if x[3] == 'India' and NAT.search(x[1]):
            k = translit(x[1])
            if k:
                tkey[(x[3], k)].append(x[0])
        if WEB.search(x[1].lower()):
            st = stem(x[1])
            if len(st) >= 4:
                wkey[(x[3], st)].append(x[0])
            else:
                st = None
        if k or st:
            gaddr[x[0]] = atoks(x[2])
print('translit keys', len(tkey), 'web stems', len(wkey), flush=True)

KEEP = int(__import__('os').environ.get('KC_KEEP', 5))
def pick(hit, at):
    if len(hit) <= KEEP:
        return hit
    sc = sorted(((len(at & gaddr[t]), t) for t in hit), reverse=True)
    return [t for s, t in sc[:KEEP] if s > 0]

pairs = []
for a, (n, c, ad) in q.items():
    at = atoks(ad)
    if c == 'India':
        hit = tkey.get((c, core(n)), [])
        pairs += [(a, t, 'translit') for t in pick(hit, at)]
    for k in s1_keys(n):
        hit = wkey.get((c, k), [])
        pairs += [(a, t, 'web') for t in pick(hit, at)]
pairs = list({(a, t): (a, t, ch) for a, t, ch in pairs}.values())
print('pairs', len(pairs), collections.Counter(p[2] for p in pairs), flush=True)
pickle.dump(pairs, open(out, 'wb'))

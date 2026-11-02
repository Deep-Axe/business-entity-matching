"""Streaming dense retrieval (frozen multilingual-e5-small): S1 queries vs the full S2+S3 gallery, within country.
Views: name_addr (every gallery record) and name (gallery records with empty address only). Keeps top-K per query per view.
usage: dense_search.py devtune|test OUT.npz"""
import csv, sys, time, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import dataset, ce_prep, e5_small, pretrained_kwargs

DS = dataset()
P = e5_small()
K, CHUNK, QT = 30, 100_000, 2048
split, out = sys.argv[1], sys.argv[2]
sub = 'train' if split == 'devtune' else 'test'

def rows(path):
    with open(path, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        yield from r

def enc_id(s):
    return int(s[3:]) + (1_000_000_000 if s[1] == '3' else 0)

if split == 'devtune':
    A = ce_prep()
    want = set()
    for sp in ('dev', 'tune'):
        want.update(np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy').tolist())
    qs = [x for x in rows(DS + 'train/train_source1.tsv') if x[0] in want]
else:
    qs = list(rows(DS + 'test/test_source1.tsv'))
qid = [x[0] for x in qs]; qc = np.array([x[3] for x in qs])
print('queries', len(qs), flush=True)

_pre = pretrained_kwargs(P)
tok = AutoTokenizer.from_pretrained(P, **_pre)
model = AutoModel.from_pretrained(P, **_pre).half().cuda().eval()
VIEWS = {'name_addr': lambda n, a: f'query: {n} | {a}', 'name': lambda n, a: f'query: {n}'}

def emb(texts, bs=512):
    order = np.argsort([len(t) for t in texts]); out = torch.empty(len(texts), 384, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            ix = order[i:i + bs]
            b = tok([texts[j] for j in ix], padding=True, truncation=True, max_length=64, return_tensors='pt').to('cuda')
            o = model(**b).last_hidden_state; m = b['attention_mask'][..., None].half()
            out[torch.from_numpy(ix).cuda()] = torch.nn.functional.normalize(((o * m).sum(1) / m.sum(1)).float(), dim=-1).half()
    return out

countries = sorted(set(qc))
Q = {v: {c: emb([f(qs[i][1], qs[i][2]) for i in np.flatnonzero(qc == c)]) for c in countries} for v, f in VIEWS.items()}
TS = {v: {c: torch.full((len(Q[v][c]), K), -2.0, dtype=torch.float16, device='cuda') for c in countries} for v in VIEWS}
TI = {v: {c: torch.full((len(Q[v][c]), K), -1, dtype=torch.int64, device='cuda') for c in countries} for v in VIEWS}
print('queries embedded', flush=True)

def search(v, c, G, gid):
    q = Q[v][c]
    for i in range(0, len(q), QT):
        s = q[i:i + QT] @ G.T
        k = min(K, s.shape[1]); vs, ix = s.topk(k, dim=1)
        allv = torch.cat([TS[v][c][i:i + QT], vs], 1); alli = torch.cat([TI[v][c][i:i + QT], gid[ix]], 1)
        tv, tix = allv.topk(K, dim=1); TS[v][c][i:i + QT] = tv; TI[v][c][i:i + QT] = alli.gather(1, tix)

def flush(buf):
    cs = np.array([x[3] for x in buf])
    for c in countries:
        m = np.flatnonzero(cs == c)
        if not len(m):
            continue
        gid = torch.tensor([enc_id(buf[j][0]) for j in m], device='cuda')
        search('name_addr', c, emb([VIEWS['name_addr'](buf[j][1], buf[j][2]) for j in m]), gid)
        e = [j for j in m if not buf[j][2].strip()]
        if e:
            search('name', c, emb([VIEWS['name'](buf[j][1], '') for j in e]), torch.tensor([enc_id(buf[j][0]) for j in e], device='cuda'))

t0 = time.time(); n = 0; buf = []
for s in ('source2', 'source3'):
    for x in rows(DS + f'{sub}/{sub}_{s}.tsv'):
        buf.append(x)
        if len(buf) == CHUNK:
            flush(buf); n += len(buf); buf = []
            if n % 1_000_000 == 0:
                print(f'gallery {n} {n / (time.time() - t0):.0f}/s', flush=True)
if buf:
    flush(buf); n += len(buf)
res = {'qid': np.array(qid)}
for v in VIEWS:
    for c in countries:
        m = np.flatnonzero(qc == c)
        res[f'{v}_{c}_ix'] = m; res[f'{v}_{c}_s'] = TS[v][c].float().cpu().numpy(); res[f'{v}_{c}_id'] = TI[v][c].cpu().numpy()
np.savez(out, **res); print('done', n, f'{time.time() - t0:.0f}s', flush=True)

"""Dense-retrieval pilot on dev retrieval misses: off-the-shelf E5-small, 20% within-country training gallery sample.
Estimated full-gallery rank of each missed true target = 5 x (sample records scoring above it)."""
import csv, collections, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

R = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/'
E = 'artifacts/improvement_v3/v6_country_errors_v1/'
P = 'artifacts/improvement_v3/transfer_pilot_v1/models/e5'
RATE = 0.2
rng = np.random.default_rng(3)

miss = []
for c in ('US', 'India'):
    with open(E + f'{c}_errors.tsv') as h:
        for r in csv.DictReader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['stage'] == 'retrieval_missing':
                miss.append(r)
print('retrieval misses', len(miss), collections.Counter(m['country'] for m in miss), flush=True)
skip = {m['target_entity_id'] for m in miss}

gal = collections.defaultdict(list)
for s in ('train_source2.tsv', 'train_source3.tsv'):
    with open(R + s) as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        for x in r:
            if x[0] not in skip and rng.random() < RATE:
                gal[x[3]].append((x[1], x[2]))
print({k: len(v) for k, v in gal.items()}, flush=True)

tok = AutoTokenizer.from_pretrained(P); model = AutoModel.from_pretrained(P).half().cuda().eval()

def emb(texts, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors='pt').to('cuda')
            o = model(**b).last_hidden_state; m = b['attention_mask'][..., None].half()
            e = (o * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(e.float(), dim=-1).half())
    return torch.cat(out)

views = {'name_addr': lambda n, a: f'query: {n} | {a}', 'name': lambda n, a: f'query: {n}'}
for vname, fmt in views.items():
    res = collections.defaultdict(list)
    for c in ('US', 'India'):
        G = emb([fmt(n, a) for n, a in gal[c]])
        ms = [m for m in miss if m['country'] == c]
        Q = emb([fmt(m['source1_name'], m['source1_address']) for m in ms]); T = emb([fmt(m['target_name'], m['target_address']) for m in ms])
        st = (Q.float() * T.float()).sum(1)
        for i in range(0, len(ms), 64):
            s = Q[i:i + 64] @ G.T
            above = (s.float() > st[i:i + 64, None]).sum(1).cpu().numpy() / RATE
            for m, a in zip(ms[i:i + 64], above):
                res[c].append((a, m['bucket'], m['target_address'] == ''))
        del G; torch.cuda.empty_cache()
    for c, v in res.items():
        a = np.array([x[0] for x in v]); ea = np.array([x[2] for x in v])
        print(f'[{vname}] {c}: n {len(a)}  est rank<=10 {(a < 10).sum()}  <=20 {(a < 20).sum()}  <=50 {(a < 50).sum()}  <=200 {(a < 200).sum()}  '
              f'| empty target addr {ea.sum()} of which <=20 {((a < 20) & ea).sum()}', flush=True)

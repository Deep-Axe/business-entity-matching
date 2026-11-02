"""Do unowned decoys come as multi-record entities (siblings that share the same 'corruption')?"""
import json, re, sqlite3, sys, collections
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

def house(a):
    m = re.findall(r'\d+', a or '')
    return m[0] if m else None

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
err = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
keys = {(r['reference']['country'], core(r['reference']['business_name'])) for r in err}
group = collections.defaultdict(list)
for eid, src, country, name, addr in con.execute('select entity_id, source, country, name, address from records'):
    k = (country, core(name))
    if k in keys:
        group[k].append((eid, src, name, addr))
owner = {}
for k, v in group.items():
    for eid, *_ in v:
        o = con.execute('select reference_id from labels where target_id=?', (eid,)).fetchone()
        owner[eid] = o[0] if o else None

def show(r):
    k = (r['reference']['country'], core(r['reference']['business_name']))
    t = r['target']; th = house(t['business_address'])
    print(f"\n[{r['kind']}] p={r['probability']:.2f}  S1 {r['reference_id']}: {r['reference']['business_name']} | {r['reference']['business_address']}")
    print(f"   target {r['target_id']} owner={owner.get(r['target_id'])}: {t['business_name']} | {t['business_address']}")
    for eid, src, name, addr in group[k]:
        if eid == r['target_id']:
            continue
        tag = 'SAME-HOUSE' if th and house(addr) == th else ''
        print(f"     sib S{src} {eid} owner={owner[eid]} {tag}: {name} | {addr}")

fps = [r for r in err if r['kind'] == 'false_positive' and owner.get(r['target_id']) is None]
stat = collections.Counter()
for r in fps:
    k = (r['reference']['country'], core(r['reference']['business_name']))
    th = house(r['target']['business_address'])
    sib = [e for e, s, n, a in group[k] if e != r['target_id'] and th and house(a) == th]
    stat['unowned FP'] += 1
    stat['has same-name same-house sibling'] += bool(sib)
    stat['... and that sibling is also unowned'] += any(owner[e] is None for e in sib)
print(dict(stat))
for r in fps[:14]:
    show(r)

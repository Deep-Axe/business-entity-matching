"""Profile the unlabeled test output against development, per country."""
import csv, sqlite3, sys, random, collections
import numpy as np
sys.path.insert(0, 'tools'); sys.path.insert(0, 'src/neural')
from fn_cats import cat
from v3_owner_assignment import winner
import json

csv.field_size_limit(1 << 30)
OUT = 'artifacts/duf-fel_full_output_v3_100k_owner_v1/matching_results.tsv'
test = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)
train = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)

def rec(con, table, eid):
    r = con.execute(f'select name,address,country from {table} where entity_id=?', (eid,)).fetchone()
    return {'business_name': r[0], 'business_address': r[1], 'country': r[2]}

country = dict(test.execute('select entity_id,country from anchors'))
counts = collections.defaultdict(list)
sample = collections.defaultdict(list)
random.seed(0)
with open(OUT) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        m = [x for x in row['matched_entity_ids'].split(',') if x]
        c = country[row['source1_entity_id']]
        counts[c].append(len(m))
        if m and random.random() < 0.01:
            sample[c].append((row['source1_entity_id'], random.choice(m)))

print('TEST output per country')
for c, v in sorted(counts.items()):
    v = np.array(v)
    print(f'  {c:7s} S1s {len(v):7d}  mean matches {v.mean():.2f}  empty {np.mean(v == 0):.3f}  >=6 {np.mean(v >= 6):.3f}')

# Development: predicted after owner assignment vs gold
policy = json.load(open('artifacts/improvement_v3/structural_v1/owner_eval/policy.json'))['selected']
db = sqlite3.connect('file:artifacts/improvement_v3/structural_v1/owner_eval/rivals.sqlite3?mode=ro', uri=True)
D = 'artifacts/improvement_v2/cpu/prepared_gate0_full_dev_v1/'
refs = np.load(D + 'reference_ids.npy'); tc = np.load(D + 'true_counts.npy')
dev_country = {r: rec(train, 'anchors', r)['country'] for r in refs}
pred = collections.Counter(); dev_pairs = []
for owner, target, p, y in db.execute("select owner,target,p,y from seeds where split='dev'"):
    claims = list(db.execute('select owner,p,text from claims where target=? order by owner', (target,)))
    if len(claims) > 1:
        ch = winner([c[1] for c in claims], [c[2] for c in claims], **policy)
        if ch is None or claims[ch][0] != owner:
            continue
    pred[owner] += 1; dev_pairs.append((owner, target, y))
print('DEV per country (after owner assignment)')
for c in ('India', 'US'):
    o = [r for r in refs if dev_country[r] == c]
    pv = np.array([pred[r] for r in o]); gv = np.array([t for r, t in zip(refs, tc) if dev_country[r] == c])
    print(f'  {c:7s} S1s {len(o):7d}  mean predicted {pv.mean():.2f} gold {gv.mean():.2f}  empty predicted {np.mean(pv == 0):.3f} gold {np.mean(gv == 0):.3f}')

def mix(pairs, con_ref, con_tgt):
    out = collections.Counter()
    for o, t in pairs:
        out[cat({'reference': rec(con_ref, 'anchors', o), 'target': rec(con_tgt, 'records', t)})] += 1
    n = sum(out.values())
    return {k: v / n for k, v in out.items()}, n

dev_prec = collections.defaultdict(lambda: [0, 0])
for o, t, y in dev_pairs:
    k = cat({'reference': rec(train, 'anchors', o), 'target': rec(train, 'records', t)})
    dev_prec[k][0] += y; dev_prec[k][1] += 1
dev_mix = {k: v[1] / len(dev_pairs) for k, v in dev_prec.items()}
test_mix = {c: mix(sample[c], test, test) for c in sample}
keys = sorted(set(dev_mix) | {k for m, _ in test_mix.values() for k in m}, key=lambda k: -dev_mix.get(k, 0))
print('\nAccepted-match mix (share of accepted pairs); dev precision per category')
print(f"{'category':30s} {'dev':>6s} {'devprec':>7s} " + ' '.join(f'{c:>7s}' for c in sorted(test_mix)))
for k in keys:
    tp, n = dev_prec.get(k, [0, 0])
    print(f'{k:30s} {dev_mix.get(k, 0):6.3f} {tp / n if n else float("nan"):7.3f} ' +
          ' '.join(f'{test_mix[c][0].get(k, 0):7.3f}' for c in sorted(test_mix)))
print('test sample sizes', {c: n for c, (_, n) in test_mix.items()})

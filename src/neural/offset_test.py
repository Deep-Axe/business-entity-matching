"""How many ACCEPTED pairs in the 0.958 submission carry a decoy-step house-number offset?"""
import csv, re, sqlite3, collections, random
csv.field_size_limit(1 << 30)
STEPS = {3, 4, 5, 7, 9, 11, 13, 21}

def nums(s):
    return [x for x in re.findall(r'\d+', s or '') if len(x) <= 7]

def step(a, b):
    A, B = nums(a), nums(b)
    sa, sb = set(A), set(B)
    oa = [x for x in A if x not in sb]; ob = [y for y in B if y not in sa]
    if not oa or not ob:
        return None
    if any(len(y) == len(x) - 1 and (x[1:] == y or x[:-1] == y) for x in oa for y in ob):
        return None
    return min(((int(y) - int(x)) for x in oa for y in ob), key=abs)

con = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)
random.seed(0)
stats = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
with open('artifacts/duf-fel_full_output_v3_100k_owner_v1/matching_results.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if random.random() > 0.05:
            continue
        ms = [m for m in row['matched_entity_ids'].split(',') if m]
        if not ms:
            continue
        country, an, aa = con.execute('select country,name,address from anchors where entity_id=?', (row['source1_entity_id'],)).fetchone()
        for m in ms:
            bn, ba = con.execute('select name,address from records where entity_id=?', (m,)).fetchone()
            d = step(aa, ba)
            stats[country]['accepted'] += 1
            if d in STEPS:
                stats[country]['decoy step'] += 1
                if len(ex[country]) < 5:
                    ex[country].append((d, an, aa, bn, ba))
            elif d in (1, 2):
                stats[country]['+1/+2'] += 1
for c, s in sorted(stats.items()):
    print(f"{c:7s} accepted {s['accepted']:6d}  decoy-step {s['decoy step']:5d} ({s['decoy step'] / s['accepted']:.4f})  +1/+2 {s['+1/+2']:5d} ({s['+1/+2'] / s['accepted']:.4f})")
for c, v in ex.items():
    for d, an, aa, bn, ba in v:
        print(f'  {c} +{d}: {an} | {aa}\n        -> {bn} | {ba}')

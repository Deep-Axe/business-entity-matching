import sqlite3, collections
con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
con.execute('PRAGMA cache_size=-400000')
n_targets = con.execute('select count(*) from records').fetchone()[0]
n_labels = con.execute('select count(*) from labels').fetchone()[0]
n_s1 = con.execute('select count(*) from anchors').fetchone()[0]
print(f'targets {n_targets}  owned {n_labels} ({n_labels / n_targets:.3f})  S1 {n_s1}')
per = collections.defaultdict(lambda: [0, 0])
q = '''select l.reference_id, r.source from labels l join records r on r.entity_id = l.target_id'''
for ref, src in con.execute(q):
    per[ref][src - 2] += 1
dist = collections.Counter(tuple(v) for v in per.values())
zero = n_s1 - len(per)
print('S1 with zero matches', zero, f'{zero / n_s1:.3f}')
print('most common (S2 count, S3 count) per S1:')
for k, v in dist.most_common(25):
    print(f'  {k}: {v} ({v / n_s1:.3f})')
print('max S2', max(v[0] for v in per.values()), 'max S3', max(v[1] for v in per.values()))

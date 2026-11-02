"""usage: make_candidates.py OUTDIR [EXTRA_GLOB]
Write <OUTDIR>/candidate_pairs.tsv = exactly the pairs the final model scores (every pair in ce/test/part_*.pkl),
one row per test Source 1 entity in test_source1.tsv order, and check matching_results.tsv is a subset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, os, glob, pickle, csv, collections

out = sys.argv[1]
T = dataset() + 'test/test_source1.tsv'
cand = collections.defaultdict(set)
for f in sorted(glob.glob(ce_data() + 'test/part_*.pkl')) + ([f for g in sys.argv[2].split(';') if g for f in sorted(glob.glob(g))] if len(sys.argv) > 2 else []):
    for r in pickle.load(open(f, 'rb')):
        cand[r[0]].add(r[1])
p = os.path.join(out, 'candidate_pairs.tsv')
if os.path.lexists(p):
    os.remove(p)
n = pairs = empty = 0
with open(T, newline='') as h, open(p + '.tmp', 'w') as o:
    rd = csv.reader(h, delimiter='\t'); next(rd)
    o.write('source1_entity_id\tcandidate_entity_ids\n')
    for row in rd:
        c = sorted(cand.get(row[0], ())); n += 1; pairs += len(c); empty += not c
        o.write(f"{row[0]}\t{','.join(c)}\n")
os.replace(p + '.tmp', p)
bad = 0
with open(os.path.join(out, 'matching_results.tsv')) as h:
    next(h)
    for l in h:
        s, t = l.rstrip('\n').split('\t')
        bad += sum(1 for x in t.split(',') if x and x not in cand.get(s, ()))
print(f'S1 rows {n}  candidate pairs {pairs}  per S1 {pairs / n:.2f}  empty rows {empty}  matches outside candidates {bad}')

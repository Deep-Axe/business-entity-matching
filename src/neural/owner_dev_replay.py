"""Replay the frozen owner-assignment policy on development and attribute the remaining loss."""
import json, sqlite3, sys, collections
import numpy as np
sys.path.insert(0, 'tools')
sys.path.insert(0, 'src/neural')
from v3_owner_assignment import winner
from fn_cats import cat

policy = json.load(open('artifacts/improvement_v3/structural_v1/owner_eval/policy.json'))['selected']
db = sqlite3.connect('file:artifacts/improvement_v3/structural_v1/owner_eval/rivals.sqlite3?mode=ro', uri=True)
D = 'artifacts/improvement_v2/cpu/prepared_gate0_full_dev_v1/'
refs = np.load(D + 'reference_ids.npy')
true_counts = dict(zip(refs, np.load(D + 'true_counts.npy')))
seeds = list(db.execute("select owner,target,p,y from seeds where split='dev'"))
kept, removed = [], []
for owner, target, p, y in seeds:
    claims = list(db.execute('select owner,p,text from claims where target=? order by owner', (target,)))
    owners = [c[0] for c in claims]
    if owner not in owners:
        raise SystemExit(f'focal claim missing for {target}')
    if len(claims) < 2:
        kept.append((owner, target, y)); continue
    choice = winner([c[1] for c in claims], [c[2] for c in claims], **policy)
    (kept if choice is not None and owners[choice] == owner else removed).append((owner, target, y))

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

def macro(kept_pairs, extra_tp=None):
    tp = collections.Counter(); k = collections.Counter()
    for o, _, y in kept_pairs:
        k[o] += 1; tp[o] += y
    extra_tp = extra_tp or {}
    return np.mean([f05(tp[o] + extra_tp.get(o, 0), k[o] + extra_tp.get(o, 0), true_counts[o]) for o in refs])

base = macro([(o, t, y) for o, t, _, y in seeds])
after = macro(kept)
print(f'dev baseline {base:.5f}  after owner assignment {after:.5f}')
print('removed FP', sum(1 - y for *_, y in removed), 'removed TP', sum(y for *_, y in removed))

err = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
by_pair = {(r['reference_id'], r['target_id']): r for r in err}
kept_set = {(o, t) for o, t, _ in kept}

remaining_fp = [by_pair[(o, t)] for o, t, y in kept if y == 0]
lost_tp = [(o, t) for o, t, y in removed if y == 1]
fn = [r for r in err if r['kind'] != 'false_positive']
print('remaining FP', len(remaining_fp), 'FN', len(fn), '+ lost TP', len(lost_tp))

# Oracle gain per category: fix every error in that category, holding the rest fixed.
cats_fp = collections.defaultdict(set)
for r in remaining_fp:
    cats_fp[('FP', r['reference']['country'][:2], cat(r))].add((r['reference_id'], r['target_id']))
cats_fn = collections.defaultdict(list)
for r in fn:
    cats_fn[('FN-' + ('retrieved' if r['kind'] == 'retrieved_false_negative' else 'not_retrieved'),
             r['reference']['country'][:2], cat(r))].append(r['reference_id'])
rows = []
for key, pairs in cats_fp.items():
    rows.append((macro([x for x in kept if (x[0], x[1]) not in pairs]) - after, len(pairs), key))
for key, owners in cats_fn.items():
    rows.append((macro(kept, collections.Counter(owners)) - after, len(owners), key))
rows.append((macro(kept + [(o, t, 1) for o, t in lost_tp]) - after, len(lost_tp), ('TP lost by owner assignment', '', '')))
print(f"\n{'oracle gain':>11} {'pairs':>5}  category")
for g, n, key in sorted(rows, reverse=True):
    if g > 0.0002:
        print(f'{g:11.5f} {n:5d}  {" / ".join(key)}')
print('total remaining loss', 1 - after)

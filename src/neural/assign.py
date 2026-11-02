"""Write matching_results.tsv from scored candidate pairs and exclusive assignment.

usage: assign.py --w W --t T --out DIR
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import (ce_data, dataset, expected_test_partitions, expand_route_globs,
                       part_pkl_id, policy_json, text_scorer_json)
import argparse, glob, pickle, json, csv, time, hashlib
from pathlib import Path
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'tools'))
sys.path.insert(0, os.path.join(_here, '..'))
from v3_owner_assignment import winner, load_text, text_logit

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default=''); ap.add_argument('--w', type=float, required=True); ap.add_argument('--t', type=float, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--countries', default='', help='comma list: only these countries use new scores')
ap.add_argument('--fallback-t', type=float, default=0.76)
ap.add_argument('--score', default='', help='precomputed score suffix, e.g. stack_TAG_nocty (overrides --w blend)')
ap.add_argument('--country-score', default='', help='comma list COUNTRY=SUFFIX: per-country score override')
ap.add_argument('--country-t', default='', help='comma list COUNTRY=THRESHOLD: per-country threshold')
ap.add_argument('--extra-glob', default='', help='extra candidate row pickles (e.g. dense route) added as edges')
ap.add_argument('--extra-score', default='', help='score suffix for the extra pickles')
ap.add_argument('--extra-t', type=float, default=0.9)
ap.add_argument('--extra-country-t', default='', help='comma list COUNTRY=THRESHOLD for the extra pickles')
ap.add_argument('--strict', action='store_true',
                help='fail if any route, expected partition, or stack score is missing')
args = ap.parse_args()
override = dict(x.split('=') for x in args.country_score.split(',') if x)
country_t = {k: float(v) for k, v in (x.split('=') for x in args.country_t.split(',') if x)}
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
policy = json.load(open(policy_json()))['selected']; text_model = load_text(str(text_scorer_json()))
only = set(args.countries.split(',')) if args.countries else None

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

edges = {}   # target -> list of (score, owner, owner_name, owner_addr, t_name, t_addr)
t0 = time.time(); n_pairs = 0; n_fallback = 0
for f in sorted(glob.glob(ce_data() + 'test/part_*.pkl')):
    rows = pickle.load(open(f, 'rb')); p = np.array([r[4] for r in rows])
    cty = np.array([r[3] for r in rows])
    thr = np.full(len(rows), args.t)
    if args.score and not os.path.exists(f'{f}.{args.score}.npy'):
        if args.strict:
            raise SystemExit(f'missing {f}.{args.score}.npy')
        s = p.copy(); thr[:] = args.fallback_t; n_fallback += len(rows)
    elif args.score:
        s = np.load(f'{f}.{args.score}.npy').astype(float); assert len(s) == len(rows)
    elif args.w > 0:
        pce = np.load(f'{f}.{args.tag}.npy'); assert len(pce) == len(rows)
        s = 1 / (1 + np.exp(-(args.w * lg(pce) + (1 - args.w) * lg(p))))
    else:
        s = p.copy()
    for c, t_c in country_t.items():
        thr = np.where(cty == c, t_c, thr)
    for c, suf in override.items():
        if os.path.exists(f'{f}.{suf}.npy'):
            s = np.where(cty == c, np.load(f'{f}.{suf}.npy').astype(float), s)
    if only is not None:
        new = np.isin(cty, list(only)); s = np.where(new, s, p); thr = np.where(new, args.t, args.fallback_t)
    for i in np.flatnonzero(s >= thr):
        r = rows[i]
        edges.setdefault(r[1], []).append((float(s[i]), r[0], r[5], r[6], r[7], r[8]))
    n_pairs += len(rows); del rows
extra_t = {k: float(v) for k, v in (x.split('=') for x in args.extra_country_t.split(',') if x)}
n_extra = n_extra_acc = 0
try:
    extra_files = expand_route_globs(args.extra_glob, strict=args.strict)
except FileNotFoundError as exc:
    raise SystemExit(f'strict: {exc}') from exc
for f in extra_files:
    rows = pickle.load(open(f, 'rb')); s = np.load(f'{f}.{args.extra_score}.npy').astype(float); assert len(s) == len(rows)
    thr = np.array([extra_t.get(r[3], args.extra_t) for r in rows])
    for i in np.flatnonzero(s >= thr):
        r = rows[i]; edges.setdefault(r[1], []).append((float(s[i]), r[0], r[5], r[6], r[7], r[8])); n_extra_acc += 1
    n_extra += len(rows); del rows
print(f'extra pairs {n_extra} accepted {n_extra_acc}', flush=True)
print(f'pairs {n_pairs} accepted edges {sum(len(v) for v in edges.values())} targets {len(edges)} {time.time() - t0:.0f}s', flush=True)
if args.strict:
    part_files = sorted(glob.glob(ce_data() + 'test/part_*.pkl'), key=part_pkl_id)
    found_parts = {part_pkl_id(path) for path in part_files}
    expected_parts = expected_test_partitions()
    missing_parts = sorted(expected_parts - found_parts)
    extra_parts = sorted(found_parts - expected_parts)
    if missing_parts:
        raise SystemExit(f'strict: missing test partitions {missing_parts}')
    if extra_parts:
        raise SystemExit(f'strict: unexpected test partitions {extra_parts}')
    if n_fallback:
        raise SystemExit(f'strict: {n_fallback} pairs used CPU-gate fallback')
    if args.score:
        missing_scores = [f'{path}.{args.score}.npy' for path in part_files
                          if not os.path.exists(f'{path}.{args.score}.npy')]
        if missing_scores:
            raise SystemExit('strict: missing stack scores: ' + ', '.join(missing_scores[:8]))
    if args.extra_score:
        missing_extra = [f'{path}.{args.extra_score}.npy' for path in extra_files
                         if not os.path.exists(f'{path}.{args.extra_score}.npy')]
        if missing_extra:
            raise SystemExit('strict: missing extra-route scores: ' + ', '.join(missing_extra[:8]))

keep = {}  # owner -> list of targets
removed = abstained = 0
for tgt, cl in edges.items():
    if len(cl) > 1:
        cl = sorted(cl, key=lambda c: c[1])
        tx = [text_logit({'name': c[2], 'address': c[3]}, {'name': c[4], 'address': c[5]}, text_model) for c in cl]
        ch = winner([c[0] for c in cl], tx, **policy)
        removed += len(cl) - (ch is not None); abstained += ch is None
        if ch is None:
            continue
        cl = [cl[ch]]
    keep.setdefault(cl[0][1], []).append(tgt)
del edges
refs = dataset() + 'test/test_source1.tsv'
n = empty = k = 0
with open(refs, newline='') as h, open(out / 'matching_results.tsv', 'w', newline='') as o:
    r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
    o.write('source1_entity_id\tmatched_entity_ids\n')
    for row in r:
        ts = sorted(keep.get(row[0], [])); n += 1; empty += not ts; k += len(ts)
        o.write(f"{row[0]}\t{','.join(ts)}\n")
rep = dict(extra_glob=args.extra_glob, extra_score=args.extra_score, extra_t=args.extra_t, extra_country_t=extra_t, extra_pairs=n_extra, extra_accepted=n_extra_acc, tag=args.tag, score=args.score, country_score=override, country_t=country_t, fallback_pairs=n_fallback, w=args.w, t=args.t, countries=args.countries or 'all', references=n, empty=empty, targets=k,
           removed_edges=removed, abstained_targets=abstained,
           matching_sha256=hashlib.sha256(open(out / 'matching_results.tsv', 'rb').read()).hexdigest())
json.dump(rep, open(out / 'report.json', 'w'), indent=1); print(json.dumps(rep))

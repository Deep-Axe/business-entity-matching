"""Read-only error triage for the frozen v10 validation decisions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from rapidfuzz import fuzz
from v3_transfer_data import ROOT, PILOT, write
from v3_structural_features import folded, house_number, single_digit_change, street_component

sys.path.insert(0, 'src/neural')
from name_unique_lib import core

C = Path('work/ce')
O = ROOT/'artifacts/improvement_v3/v10_combined_audit_v1'
raw = lambda p: pickle.load(p.open('rb'))
rows, new, nf, newnf = [], [], [], []
tags = ['e5_cont1300k'] + ['k'+c for c in 'ABCDEFG']
ces = {t: [] for t in tags}
for sp in ('dev', 'tune'):
    r = raw(C/f'{sp}.pkl'); fr = raw(C/f'{sp}_f.pkl')
    rows.extend(r); new.extend(fr[len(r):])
    nf.append(np.load(C/f'{sp}.pkl.nf.npy'))
    newnf.append(np.load(C/f'{sp}_f.pkl.nf.npy')[len(r):])
    for t in tags:
        ces[t].append(np.load(C/t/f'{sp}_pce.npy'))
nold = len(rows); rows += new; nf = np.vstack(nf+newnf)
ce = np.column_stack([np.r_[np.concatenate(ces[t]), np.full(len(new), np.nan)] for t in tags])
z = np.load(O/'decisions.npz'); keep = z['combined_keep']; p = z['scores']
assert len(rows) == len(keep) == len(nf)
oldp = np.load(ROOT/'artifacts/improvement_v3/public0976_error_audit_v2/decisions.npz')['oof']
refs = {sp: set(np.load(PILOT/'data'/f'{sp}.npz')['reference_ids']) for sp in ('dev', 'tune')}
report = {}; exported = []
for sp, owners in refs.items():
    patterns = defaultdict(Counter); bands = defaultdict(Counter); stages = Counter(); totals = Counter()
    examples = []
    for i, r in enumerate(rows):
        if r[0] not in owners:
            continue
        y = r[2]; typ = 'TP' if keep[i] and y else 'FP' if keep[i] else 'FN' if y else 'TN'
        totals[typ] += 1
        if typ not in ('FN', 'TN'):
            continue
        cutoff = .75 if i < nold else .9
        stage = 'owner_removed' if p[i] >= cutoff else 'new_route_rejected' if i >= nold else 'original_classifier_rejected'
        if typ == 'FN':
            stages[stage] += 1
        empty = not r[8].strip(); lc, rc = core(r[5]), core(r[7])
        native = any('\u0900' <= c <= '\u0dff' for c in r[7])
        lh, rh = house_number(r[6], r[3]), house_number(r[8], r[3])
        ls, rs = street_component(r[6], r[3]), street_component(r[8], r[3])
        nr = fuzz.ratio(folded(r[5]), folded(r[7]))
        flags = dict(empty_address=empty,
            empty_raw_name_equal=empty and r[5].casefold().strip() == r[7].casefold().strip(),
            empty_core_equal_shared=empty and bool(lc) and lc == rc and np.expm1(nf[i,0]) > 1.5,
            empty_core_unique=empty and bool(lc) and lc == rc and np.expm1(nf[i,0]) < 1.5,
            native_script=native, native_core_erased=native and not rc,
            low_name_similarity_usable_address=not empty and nr < 50,
            low_name_same_house_street=not empty and nr < 50 and bool(lh) and lh == rh and bool(ls) and fuzz.token_sort_ratio(ls,rs) >= 90,
            digit_drop=bool(lh and rh and len(lh) == len(rh)+1 and single_digit_change(lh,rh)),
            new_dense_candidate=i>=nold,
            old_band_excluded=i<nold and not (.05 <= oldp[i] < .98),
            any_ce_ge_09=i<nold and bool(np.max(ce[i]) >= .9),
            any_ce_ge_099=i<nold and bool(np.max(ce[i]) >= .99),
            ce_disagree_09_01=i<nold and bool(np.max(ce[i]) >= .9 and np.min(ce[i]) < .1),
            mean_ce_ge_09=i<nold and bool(np.mean(ce[i]) >= .9))
        for name, flag in flags.items():
            if flag:
                patterns[name][typ] += 1
        lo = next(v for v in (0.9, 0.75, 0.7, 0.6, 0.5, 0.2, 0.05, 0.01, 0.) if p[i] >= v)
        bands[str(lo)][typ] += 1
        if typ == 'FN':
            examples.append(dict(split=sp, owner=r[0], target=r[1], country=r[3], stage=stage,
                final_score=float(p[i]), flags=[k for k,v in flags.items() if v],
                source_name=r[5], source_address=r[6], target_name=r[7], target_address=r[8],
                ce_scores={t: float(v) for t,v in zip(tags,ce[i])} if i<nold else {},
                source_core_count=float(np.expm1(nf[i,0])), target_core_count=float(np.expm1(nf[i,1]))))
    report[sp] = dict(retrieved_only_counts=dict(totals), fn_stages=dict(stages),
        overlapping_patterns={k:dict(v) for k,v in patterns.items()}, score_bins_lower_bound={k:dict(v) for k,v in bands.items()})
    exported.extend(examples)
write(O/'recovery_triage.json', report)
write(O/'retrieved_misses.json', exported)
print(json.dumps(report, indent=2), flush=True)

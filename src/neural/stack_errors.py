"""Categorise remaining dev errors of the stacked model (OOF, no owner step) vs the p100k baseline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, pickle, collections
import numpy as np
sys.argv = sys.argv[:1] + ['x'] if len(sys.argv) < 2 else sys.argv
from fn_cats import cat

tag, t = sys.argv[1], float(sys.argv[2])
rows, oof = pickle.load(open(fce_data() + 'stack_oof_{tag}.pkl', 'rb'))

def c(r):
    return cat({'reference': {'business_name': r[6], 'business_address': r[7]}, 'target': {'business_name': r[8], 'business_address': r[9]}})

for sp in ('dev', 'tune'):
    fn_new = collections.Counter(); fp_new = collections.Counter(); fn_old = collections.Counter(); fp_old = collections.Counter()
    for r, v in zip(rows, oof):
        if r[0] != sp:
            continue
        k = (r[4], c(r))
        if r[3] == 1 and v < t: fn_new[k] += 1
        if r[3] == 0 and v >= t: fp_new[k] += 1
        if r[3] == 1 and r[5] < 0.76: fn_old[k] += 1
        if r[3] == 0 and r[5] >= 0.76: fp_old[k] += 1
    print(f'== {sp}: FN old {sum(fn_old.values())} new {sum(fn_new.values())} | FP old {sum(fp_old.values())} new {sum(fp_new.values())}')
    keys = sorted(set(fn_new) | set(fn_old) | set(fp_new) | set(fp_old), key=lambda k: -(fn_new[k] + fp_new[k]))
    for k in keys[:18]:
        print(f'   {k[0]:6s} {k[1]:32s} FN {fn_old[k]:4d}->{fn_new[k]:4d}   FP {fp_old[k]:3d}->{fp_new[k]:3d}')

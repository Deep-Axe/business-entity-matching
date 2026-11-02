"""Signed house-number offset features used by the stacker."""
import re

from editops import feats  # noqa: F401  (imported by stack_feats via exec)

DECOY_STEPS = {3, 4, 5, 7, 9, 11, 13, 21}

def nums(s):
    return [x for x in re.findall(r'\d+', s or '') if len(x) <= 7]

def offset_feats(a, b):
    A, B = nums(a), nums(b)
    if not A or not B:
        return [0, 0, 0, 0, 0, 99]
    sa, sb = set(A), set(B)
    oa = [x for x in A if x not in sb]; ob = [y for y in B if y not in sa]
    dropped = int(any(len(y) == len(x) - 1 and (x[1:] == y or x[:-1] == y) for x in oa for y in ob))
    if not oa or not ob:
        return [0, dropped, 0, 0, len(ob) - len(oa), 99]
    d = min(((int(y) - int(x)) for x in oa for y in ob), key=abs)
    return [1, dropped, int(d in DECOY_STEPS), int(1 <= d <= 25), len(ob) - len(oa), max(-60, min(60, d))]

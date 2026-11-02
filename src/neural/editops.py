"""Edit-operation features for uncertain pairs: do they separate true matches from decoys better than p100k?"""
import json, re, sqlite3, collections, unicodedata
import numpy as np
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

LEGAL = {'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private', 'llp', 'lp', 'pllc',
         'pc', 'pa', 'plc', 'incorporated', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci'}
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's'})
GENERIC = {'group', 'center', 'centre', 'service', 'services', 'enterprises', 'enterprise', 'ventures', 'solutions',
           'partners', 'associates', 'holdings', 'international', 'india', 'global', 'systems', 'technologies', 'the',
           'and', 'of', 'sri', 'shree', 'dr', 'new', 'foods', 'exports', 'traders', 'industries', 'consultants'}

def fold(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()

def name_tokens(s):
    toks = re.findall(r'[a-z0-9@$]+', fold(s))
    return [t.translate(LEET) if not t.isdigit() else t for t in toks]

def nums(s):
    return re.findall(r'\d+', s or '')

def house_rel(x, ys):
    if x in ys: return 0
    for y in ys:
        if len(y) == len(x) - 1 and (x[1:] == y or x[:-1] == y): return 1
        if len(y) == len(x) + 1 and (y[1:] == x or y[:-1] == x): return 2
        if len(y) == len(x) and sum(a != b for a, b in zip(x, y)) == 1:
            pos = next(i for i, (a, b) in enumerate(zip(x, y)) if a != b)
            return 3 if pos < len(x) - 1 else 4
    for y in ys:
        if abs(int(x) - int(y)) <= 25: return 5
    return 6

def feats(a_name, a_addr, b_name, b_addr):
    A, B = name_tokens(a_name), name_tokens(b_name)
    Al, Bl = [t for t in A if t in LEGAL], [t for t in B if t in LEGAL]
    A, B = [t for t in A if t not in LEGAL], [t for t in B if t not in LEGAL]
    exact = sum(1 for t in A if t in B)
    ua = [t for t in A if t not in B]; ub = [t for t in B if t not in A]
    typo = sum(1 for t in ua if any(fuzz.ratio(t, u) >= 75 for u in ub))
    ua_left = len(ua) - typo
    ub_left = len(ub) - sum(1 for u in ub if any(fuzz.ratio(t, u) >= 75 for t in ua))
    ub_generic = sum(1 for u in ub if u in GENERIC)
    ua_generic = sum(1 for t in ua if t in GENERIC)
    order = [B.index(t) for t in A if t in B]
    reordered = int(order != sorted(order))
    dup = int(len(B) != len(set(B)))
    short_a = [t for t in re.findall(r'\b[A-Z]{2,4}\b', a_name)]
    initial_edit = int(any(fold(t) not in fold(b_name) for t in short_a))
    native = int(any(ord(c) > 0x900 for c in b_name))
    ta = b_addr.strip()
    b_empty = int(not ta or ta.upper() in ('NULL', '<NULL>'))
    na, nb = nums(a_addr), nums(b_addr)
    hr = 7 if not na else (8 if not nb else house_rel(na[0], nb))
    added = sum(1 for y in nb if y not in na); removed = sum(1 for x in na if x not in nb)
    extra = int(bool(re.search(r'pmb|www|\.com|\bid\b|#\d', b_addr.lower() + ' ' + b_name.lower())))
    sa = set(re.findall(r'[a-z]+', fold(a_addr))); sb = set(re.findall(r'[a-z]+', fold(b_addr)))
    street_j = len(sa & sb) / max(1, len(sa | sb))
    return [exact, typo, ua_left, ub_left, ub_generic, ua_generic, reordered, dup, initial_edit, native,
            int(bool(Al)), int(bool(Bl)), int(bool(set(Al) & set(Bl))), b_empty, hr, added, removed, extra, street_j,
            fuzz.token_set_ratio(fold(a_name), fold(b_name)), len(A), len(B)]

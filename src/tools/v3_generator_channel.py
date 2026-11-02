"""Observable edit evidence and train-only empirical corruption likelihoods.

No test labels, record-ID features, or universal numeric accept/reject rules.
Raw text remains available to the matcher. All string views are supplemental.
"""
from collections import Counter
from functools import lru_cache
import math
import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from v3_structural_features import LEGAL, address_view, house_number, street_component, single_digit_change

NATIVE = re.compile(r'[\u0900-\u0dff]')
WEB = re.compile(r'(?:www\.|\.(?:com|in|net|org|co|fr)\b|^@)', re.I)
GENERIC = frozenset('group holdings exports services service global international india enterprise enterprises solutions partners associates systems technologies industries traders trading company corp corporation ltd limited llc private pvt'.split())


def normalize(s):
    # Preserve Indic combining marks rather than ASCII-ignore or accent stripping.
    s = unicodedata.normalize('NFC', s or '').casefold()
    return ' '.join(re.sub(r'[^\w\u0300-\u036f\u0900-\u0dff]+', ' ', s).split())


def make_dictionary(counts):
    result = {}
    for token, votes in counts.items():
        word, n = Counter(votes).most_common(1)[0]
        total = sum(votes.values())
        if n >= 2 and n / total >= .4:
            result[token] = (word, n / total, total)
    return result


class Channel:
    def __init__(self, dictionary=None):
        self.dictionary = dictionary or {}

    @lru_cache(maxsize=16384)
    def name(self, s):
        raw = normalize(s)
        words = []
        confidences = []
        support = []
        unknown = 0
        for token in re.split(r'[\s,()\[\]/&]+', s or ''):
            if not token:
                continue
            if NATIVE.search(token):
                hit = self.dictionary.get(token)
                if hit:
                    token, confidence, n = hit
                    confidences.append(confidence)
                    support.append(n)
                else:
                    unknown += 1
            words.extend(normalize(token).split())
        core = [w for w in words if w not in LEGAL]
        leet = ' '.join(w.translate(str.maketrans('013458', 'oieasb'))
                        if re.search('[a-z]', w) and re.search('[0-9]', w) else w for w in core)
        return (raw, tuple(words), tuple(core), ' '.join(core), leet,
                min(confidences, default=0), math.log1p(min(support, default=0)),
                len(confidences), unknown)

    @lru_cache(maxsize=16384)
    def address(self, text, country):
        clean = '' if (text or '').strip().lower() in ('null', '<null>', 'nan') else text or ''
        nums = re.findall(r'\d+', clean)
        return (address_view(clean, country), street_component(clean, country),
                house_number(clean, country), nums[0] if nums else '', tuple(nums),
                bool(re.search(r'\d\s*[/\-]\s*\d', clean)),
                bool(re.search(r'\b(?:suite|ste|unit|flat|floor|apt|apartment)\b', clean, re.I)))

    def extract(self, row):
        # Some rival caches append provenance columns after the shared schema.
        _, target, _, country, _, an, aa, bn, ba = row[:9]
        a, b = self.name(an), self.name(bn)
        av, bv = self.address(aa, country), self.address(ba, country)
        aset, bset = set(a[2]), set(b[2])
        added, dropped = bset - aset, aset - bset
        legal_a, legal_b = set(a[1]) & LEGAL, set(b[1]) & LEGAL
        hr = number_relation(av[2], bv[2])
        first = number_relation(av[3], bv[3])
        off = int(bv[2]) - int(av[2]) if av[2] and bv[2] else None
        firstoff = int(bv[3]) - int(av[3]) if av[3] and bv[3] else None
        street = fuzz.ratio(av[1], bv[1]) / 100 if av[1] and bv[1] else 0
        name = fuzz.ratio(a[3], b[3]) / 100 if a[3] and b[3] else 0
        web = bool(WEB.search(bn))
        stem = re.sub(r'^(?:www\.|@)', '', bn.lower())
        stem = re.split(r'\.(?:com|in|net|org|co|fr)\b', stem)[0]
        stem = re.sub(r'[^a-z0-9]', '', stem)
        edits = Counter(e.tag for e in Levenshtein.editops(a[4], b[4]))
        values = [name, fuzz.ratio(a[0], b[0])/100, fuzz.ratio(a[4], b[4])/100,
                  fuzz.token_sort_ratio(a[3], b[3])/100, fuzz.token_set_ratio(a[3], b[3])/100,
                  len(aset & bset)/max(1,len(aset | bset)),
                  min(len(added),6), min(len(dropped),6),
                  len(added & GENERIC), len(dropped & GENERIC),
                  bool(legal_a and legal_b and legal_a != legal_b), bool(legal_a & legal_b),
                  bool(NATIVE.search(an)), bool(NATIVE.search(bn)),
                  b[5], b[6], b[7], b[8], web,
                  fuzz.ratio(''.join(a[2]),stem)/100 if web else 0,
                  int(bool(stem) and stem == ''.join(w[:1] for w in a[2])) if web else 0,
                  street, fuzz.token_sort_ratio(av[1],bv[1])/100 if av[1] and bv[1] else 0,
                  fuzz.ratio(av[0],bv[0])/100 if av[0] and bv[0] else 0,
                  bool(av[0]), bool(bv[0]), bool(av[1] and bv[1]),
                  bool(av[2] and bv[2]), hr, first,
                  signed_log(off), signed_log(firstoff),
                  av[5], bv[5], av[6], bv[6],
                  edits['insert']/max(1,len(a[4])), edits['delete']/max(1,len(a[4])),
                  edits['replace']/max(1,len(a[4])),
                  bool(set(av[4]) & set(bv[4])), len(av[4]), len(bv[4]),
                  min(abs(off),100) if off is not None else -1,
                  min(abs(firstoff),100) if firstoff is not None else -1]
        # No entire business strings or country identifiers are signature keys.
        nb = min(4,int(name*5)); sb = min(4,int(street*5))
        missing = int(not bv[0]); source = target[:2]
        legal = int(bool(legal_a and legal_b and legal_a != legal_b))
        ops = [('house',str(hr)),('firstnum',str(first)),
               ('offset',str(off) if off is not None and abs(off)<=50 else 'unknown_or_large'),
               ('firstoffset',str(firstoff) if firstoff is not None and abs(firstoff)<=50 else 'unknown_or_large'),
               ('name',f'{min(len(added),3)}:{min(len(dropped),3)}:{legal}:{nb}'),
               ('street_house',f'{sb}:{hr}:{legal}'),
               ('joint',f'{nb}:{sb}:{hr}:{missing}:{legal}'),
               ('source_joint',f'{source}:{nb}:{sb}:{hr}:{missing}:{legal}'),
               ('script',f'{int(bool(NATIVE.search(bn)))}:{min(b[8],2)}:{nb}:{sb}:{missing}')]
        for t in sorted(added & GENERIC): ops.append(('add_generic',f'{t}:{sb}:{hr}'))
        for t in sorted(dropped & GENERIC): ops.append(('drop_generic',f'{t}:{sb}:{hr}'))
        return values, [f'{k}={v}' for k,v in ops]


def signed_log(x):
    return math.copysign(math.log1p(abs(x)),x) if x is not None else 0.


def number_relation(a,b):
    if not a or not b: return 0
    if int(a)==int(b): return 1
    if single_digit_change(a,b): return 2 if len(a)>len(b) else 3
    if len(a)==len(b):
        diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        if len(diff)==1:return 4
        if len(diff)==2 and a[diff[0]]==b[diff[1]] and a[diff[1]]==b[diff[0]]:return 5
    if abs(int(a)-int(b))<=30:return 6
    return 7


class Likelihoods:
    def __init__(self):
        self.counts={}; self.totals=[0,0]

    def add(self, events, y):
        self.totals[int(y)]+=1
        for event in set(events):
            self.counts.setdefault(event,[0,0])[int(y)]+=1

    def transform(self,events):
        prior=self.totals[1]/max(1,sum(self.totals))
        out=[]
        for event in events[:9]:
            a,b=self.counts.get(event,[0,0]); n=a+b
            # Shrink empirical posterior to the global training prevalence.
            post=(b+32*prior)/(n+32)
            odds=math.log(max(1e-9,post)/max(1e-9,1-post))-math.log(prior/(1-prior))
            out.extend([max(-8,min(8,odds)),math.log1p(n)])
        extras=[]
        for event in events[9:]:
            a,b=self.counts.get(event,[0,0]);post=(b+32*prior)/(a+b+32)
            extras.append(math.log(max(1e-9,post)/max(1e-9,1-post))-math.log(prior/(1-prior)))
        out.extend([min(extras,default=0),max(extras,default=0),sum(extras)/max(1,len(extras))])
        return out

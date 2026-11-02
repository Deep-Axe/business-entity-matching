"""Label-free CPU features appended to unchanged frozen pair/context columns."""
from dataclasses import dataclass
from functools import lru_cache
import math
import re
import unicodedata

from anyascii import anyascii
from rapidfuzz import fuzz
import tantivy

from ber.text import normalize


WEIGHTED_METRICS = ('weighted_jaccard', 'reference_coverage', 'target_coverage',
                    'max_shared_idf', 'max_reference_only_idf', 'max_target_only_idf')
EXTRA_NAMES = [f'v2_{view}_{metric}' for view in ('name', 'ascii_name', 'address')
               for metric in WEIGHTED_METRICS] + [
    'v2_number_jaccard', 'v2_number_reference_coverage', 'v2_number_target_coverage',
    'v2_short_number_overlap', 'v2_short_number_disjoint',
    'v2_long_number_overlap', 'v2_long_number_disjoint', 'v2_number_one_side_missing',
    'v2_suffix_stripped_ascii_ratio', 'v2_suffix_stripped_ascii_token_set',
    'v2_accent_stripped_name_ratio', 'v2_accent_stripped_address_ratio']

SUFFIXES = frozenset(('private', 'limited', 'corporation', 'incorporated', 'company',
                     'llc', 'llp', 'plc', 'sarl', 'eurl', 'sas', 'sa'))


def strip_suffix(name):
    words = name.split()
    while len(words) > 1 and words[-1] in SUFFIXES:
        words.pop()
    return ' '.join(words)


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if not unicodedata.combining(c))


def numeric_tokens(address):
    # Unordered tokens, not a claim that the first number is the house number.
    result = set()
    for value, suffix in re.findall(r'\b([0-9]+)([a-z]?)\b', address):
        result.add((value.lstrip('0') or '0') + suffix)
    return frozenset(result)


@dataclass(slots=True)
class View:
    name: str
    address: str
    name_tokens: frozenset
    address_tokens: frozenset
    ascii_tokens: frozenset
    suffix_name: str
    accent_name: str
    accent_address: str
    numbers: frozenset

    @classmethod
    def from_row(cls, row):
        name = normalize(row['business_name'])
        address = normalize(row['business_address'], True)
        ascii_name = normalize(anyascii(row['business_name']))
        ascii_address = normalize(anyascii(row['business_address']), True)
        return cls(name, address, frozenset(name.split()), frozenset(address.split()),
                   frozenset(ascii_name.split()), strip_suffix(ascii_name),
                   strip_accents(name), strip_accents(address), numeric_tokens(ascii_address))


class Weighting:
    def __init__(self, index):
        self.index = tantivy.Index.open(str(index / 'tantivy'))
        self.searcher = self.index.searcher()
        self.documents = self.searcher.num_docs
        self.scale = math.log1p(self.documents)
        self.idf = lru_cache(maxsize=150000)(self._idf)

    def _idf(self, field, token):
        # Unseen terms get finite max weight; matching/OOV text is not a label.
        return math.log((self.documents + 1) / (self.searcher.doc_freq(field, token) + 1)) + 1


def weighted_overlap(left, right, field, weighting):
    union = left | right
    if not union:
        return [0.] * 6
    weights = {token: weighting.idf(field, token) for token in union}
    shared = left & right
    common = sum(weights[t] for t in shared)
    total = sum(weights.values())
    left_total = sum(weights[t] for t in left)
    right_total = sum(weights[t] for t in right)
    rare = lambda values: max((weights[t] for t in values), default=0.) / (weighting.scale + 1)
    return [common / total, common / left_total if left_total else 0.,
            common / right_total if right_total else 0., rare(shared),
            rare(left - right), rare(right - left)]


def number_features(left, right):
    both = left & right
    union = left | right
    short = lambda tokens: {t for t in tokens if len(re.sub('[a-z]', '', t)) <= 4}
    ls, rs = short(left), short(right)
    ll, rl = left - ls, right - rs
    return [len(both) / len(union) if union else 0., len(both) / len(left) if left else 0.,
            len(both) / len(right) if right else 0., float(bool(ls & rs)),
            float(bool(ls) and bool(rs) and not ls & rs), float(bool(ll & rl)),
            float(bool(ll) and bool(rl) and not ll & rl), float(bool(left) != bool(right))]


def extra_features(left, right, weighting):
    out = []
    for lhs, rhs, field in ((left.name_tokens, right.name_tokens, 'name_words'),
                            (left.ascii_tokens, right.ascii_tokens, 'ascii_words'),
                            (left.address_tokens, right.address_tokens, 'address_words')):
        out.extend(weighted_overlap(lhs, rhs, field, weighting))
    out.extend(number_features(left.numbers, right.numbers))
    ratio = lambda fn, a, b: fn(a, b) / 100 if a and b else 0.
    out.extend((ratio(fuzz.ratio, left.suffix_name, right.suffix_name),
                ratio(fuzz.token_set_ratio, left.suffix_name, right.suffix_name),
                ratio(fuzz.ratio, left.accent_name, right.accent_name),
                ratio(fuzz.ratio, left.accent_address, right.accent_address)))
    assert len(out) == len(EXTRA_NAMES)
    return out

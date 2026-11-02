"""Additive, country-aware views for an isolated cached-pair reranker.

These are observable heuristics, not generator truth. Original text/features
are retained. Unknown house/city/state roles are explicitly represented.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re

from anyascii import anyascii
from rapidfuzz import fuzz

LEGAL = frozenset('private pvt limited ltd llc llp inc incorporated corp corporation company co plc sarl sas sasu eurl sa sci ei'.split())
HONORIFIC = frozenset('shri sri smt dr mr mrs ms monsieur madame docteur'.split())
US_STATES = dict(x.split(':', 1) for x in (
    'AL:alabama|AK:alaska|AZ:arizona|AR:arkansas|CA:california|CO:colorado|CT:connecticut|DE:delaware|'
    'FL:florida|GA:georgia|HI:hawaii|ID:idaho|IL:illinois|IN:indiana|IA:iowa|KS:kansas|KY:kentucky|'
    'LA:louisiana|ME:maine|MD:maryland|MA:massachusetts|MI:michigan|MN:minnesota|MS:mississippi|'
    'MO:missouri|MT:montana|NE:nebraska|NV:nevada|NH:new hampshire|NJ:new jersey|NM:new mexico|'
    'NY:new york|NC:north carolina|ND:north dakota|OH:ohio|OK:oklahoma|OR:oregon|PA:pennsylvania|'
    'RI:rhode island|SC:south carolina|SD:south dakota|TN:tennessee|TX:texas|UT:utah|VT:vermont|'
    'VA:virginia|WA:washington|WV:west virginia|WI:wisconsin|WY:wyoming|DC:district of columbia'
).split('|'))
INDIA_STATES = {
    'UP': ('uttar pradesh', 'उत्तर प्रदेश'), 'MH': ('maharashtra', 'महाराष्ट्र'),
    'KA': ('karnataka', 'ಕರ್ನಾಟಕ'), 'TN': ('tamil nadu', 'தமிழ்நாடு'),
    'DL': ('delhi', 'new delhi', 'दिल्ली'), 'WB': ('west bengal', 'পশ্চিমবঙ্গ'),
    'GJ': ('gujarat', 'ગુજરાત'), 'RJ': ('rajasthan', 'राजस्थान'),
    'MP': ('madhya pradesh', 'मध्य प्रदेश'), 'AP': ('andhra pradesh', 'ఆంధ్ర ప్రదేశ్'),
    'TS': ('telangana', 'తెలంగాణ'), 'TG': ('telangana', 'తెలంగాణ'),
    'KL': ('kerala', 'കേരളം'), 'PB': ('punjab', 'ਪੰਜਾਬ'), 'HR': ('haryana', 'हरियाणा'),
    'BR': ('bihar', 'बिहार'), 'OD': ('odisha', 'orissa', 'ଓଡ଼ିଶା'),
    'OR': ('odisha', 'orissa'), 'JH': ('jharkhand', 'झारखंड'),
    'CG': ('chhattisgarh', 'छत्तीसगढ़'), 'CT': ('chhattisgarh',),
    'UK': ('uttarakhand', 'उत्तराखण्ड', 'उत्तराखंड'), 'UT': ('uttarakhand',),
    'HP': ('himachal pradesh', 'हिमाचल प्रदेश'), 'AS': ('assam', 'অসম'),
    'GA': ('goa',), 'CH': ('chandigarh',), 'PY': ('puducherry', 'pondicherry'),
    'JK': ('jammu and kashmir',), 'LA': ('ladakh',), 'AN': ('andaman and nicobar islands',),
    'AR': ('arunachal pradesh',), 'MN': ('manipur',), 'ML': ('meghalaya',),
    'MZ': ('mizoram',), 'NL': ('nagaland',), 'SK': ('sikkim',), 'TR': ('tripura',),
}
US_STREET = dict(x.split(':') for x in 'rd:road st:street ave:avenue av:avenue blvd:boulevard dr:drive ln:lane ct:court cir:circle pl:place hwy:highway pkwy:parkway trl:trail ter:terrace'.split())
FR_STREET = dict(x.split(':') for x in 'r:rue av:avenue ave:avenue bd:boulevard blvd:boulevard chem:chemin all:allee pl:place psg:passage rte:route imp:impasse fbg:faubourg crs:cours'.split())
ORDINALS = dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(), (f'{i}' for i in range(1, 21))))


def folded(text):
    return ' '.join(re.findall(r'[a-z0-9]+', anyascii(text or '').lower()))


def core_name(text):
    return ' '.join(sorted(set(folded(text).split()) - LEGAL - HONORIFIC))


def leet_view(text):
    # Only mixed letters/digits, never alter standalone business/address numbers.
    table = str.maketrans('013458', 'oieasb')
    return ' '.join(w.translate(table) if re.search('[a-z]', w) and re.search('[0-9]', w) else w
                    for w in folded(text).split())


@lru_cache(maxsize=8)
def state_aliases(country):
    aliases = {}
    if country == 'US':
        for code, name in US_STATES.items(): aliases.update({code.lower(): name, name: name})
    elif country == 'India':
        for code, names in INDIA_STATES.items():
            for word in (code, *names): aliases[folded(word)] = names[0]
    return aliases


def state_name(address, country):
    aliases = state_aliases(country)
    # Only a comma component consisting of a state (+ optional postal digits)
    # qualifies. Avoid interpreting "OR", "IN", "UP" inside street text.
    found = set()
    for part in address.split(','):
        key = ' '.join(w for w in folded(part).split() if not w.isdigit())
        if key in aliases: found.add(aliases[key])
    return next(iter(found)) if len(found) == 1 else ''


def address_view(address, country):
    mapping = FR_STREET if country == 'France' else US_STREET
    words = []
    for word in folded(address).split():
        word = mapping.get(word, word)
        word = ORDINALS.get(word, word)
        word = re.sub(r'^(\d+)(st|nd|rd|th)$', r'\1', word)
        words.append(word)
    return ' '.join(words)


def house_number(address, country):
    """Conservative component-role proxy; compound India identifiers stay unknown."""
    street_words = set(US_STREET.values()) | set(FR_STREET.values()) | {'street', 'road', 'marg', 'nagar'}
    candidates = []
    for part in address.split(','):
        raw = anyascii(part).strip().lower()
        words = address_view(part, country).split()
        match = re.match(r'^(?:(?:house|plot|door|h\.?\s*no\.?|no\.?)\s*[-:]?\s*)?(\d{1,5})(?:\s|$)', raw)
        if match and (street_words.intersection(words) or country == 'India' and raw.strip().isdigit()):
            candidates.append(str(int(match[1])))
    return candidates[0] if len(candidates) == 1 else ''


def street_component(address, country):
    types=set(US_STREET.values())|set(FR_STREET.values())|{'marg'}
    pieces=[address_view(part,country) for part in address.split(',')]
    found=[part for part in pieces if set(part.split())&types]
    # Do not reward matching city/region text as though it were a street.
    if len(found)!=1:return ''
    return ' '.join(w for w in found[0].split() if not re.search(r'\d',w))


def city_proxy(address, country):
    pieces = [folded(x) for x in address.split(',')]
    states = state_name(address, country)
    candidates = [x for x in pieces if x and not re.search(r'\d', x)
                  and not state_name(x, country) and x != states
                  and not set(address_view(x, country).split()) & (set(US_STREET.values()) | set(FR_STREET.values()))]
    if country == 'France':
        regions = {'nouvelle aquitaine','hauts de france','ile de france','auvergne rhone alpes',
                   'provence alpes cote d azur','occitanie','bretagne','normandie','grand est',
                   'pays de la loire','centre val de loire','bourgogne franche comte','corse'}
        candidates = [x for x in candidates if x not in regions]
    return candidates[0] if len(candidates) == 1 else ''


def single_digit_change(a, b):
    if not a or not b or abs(len(a)-len(b)) != 1: return False
    short, long = sorted((a,b), key=len)
    return any(long[:i]+long[i+1:] == short for i in range(len(long)))


def learn_fragments(records, train):
    unusual, ordinary = Counter(), Counter()
    unusual_ids, ordinary_ids = set(), set()
    for i in range(len(train['y'])):
        if train['y'][i] != 1: continue
        l, r = int(train['left'][i]), int(train['right'][i])
        if records[l]['fold'] != 'train' or records[r]['fold'] != 'train':
            raise ValueError('Fragment inventory must use training-fold positives only')
        pool = unusual_ids if fuzz.ratio(folded(records[l]['name']),folded(records[r]['name'])) < 30 else ordinary_ids
        pool.update((l,r))
    for ids, counter in ((unusual_ids,unusual),(ordinary_ids,ordinary)):
        for i in ids:
            grams = set()
            for token in folded(records[i]['name']).split():
                if token.isalpha() and len(token) >= 7:
                    grams.update(token[j:j+4] for j in range(len(token)-3))
            counter.update(grams)
    ranked = [(n/(ordinary[g]+1),n,g) for g,n in unusual.items() if n >= 3 and n/(ordinary[g]+1) >= 2]
    return {'fragments':[g for _,_,g in sorted(ranked,reverse=True)[:64]],
            'unusual_training_records':len(unusual_ids),'ordinary_training_records':len(ordinary_ids),
            'description':'Train-positive low-name-similarity enriched 4-grams; NOT a known generator syllable inventory'}


def pseudo_score(name, fragments):
    word = folded(name).replace(' ', '')
    covered = set(); hits = 0
    for g in fragments:
        start = word.find(g)
        if start >= 0: hits += 1; covered.update(range(start,start+len(g)))
    return len(covered)/max(1,len(word)) if hits >= 2 else 0.


@dataclass(frozen=True)
class View:
    core: str
    name: str
    leet: str
    address: str
    street: str
    state: str
    house: str
    city: str
    pseudo: float
    country: str

    @classmethod
    def make(cls, name, address, country, fragments=()):
        view = address_view(address,country)
        return cls(core_name(name),folded(name),leet_view(name),view,
                   street_component(address,country),
                   state_name(address,country),house_number(address,country),city_proxy(address,country),
                   pseudo_score(name,fragments),country)


FEATURE_GROUPS = {
    'house': ['house_both_known','house_exact','house_single_digit_change','house_small_offset',
              'house_log_offset','house_other_conflict','house_one_missing','house_both_unknown'],
    'address': ['street_ratio','street_token_sort','street_jaccard','normalized_address_ratio',
                'address_one_missing','address_both_missing'],
    'state': ['state_both_known','state_equal','state_conflict'],
    'name': ['core_name_ratio','leet_name_ratio','leet_gain','honorific_core_gain','pseudo_left','pseudo_right'],
    'sharing': ['log_s1_core_name_count','log_s1_city_core_count','city_proxy_known'],
}
NAMES = sum(FEATURE_GROUPS.values(), [])


def ratio(a,b): return fuzz.ratio(a,b)/100 if a and b else 0.


def pair_features(l, r, name_count=0, city_count=0):
    both = bool(l.house and r.house)
    offset = abs(int(l.house)-int(r.house)) if both else 0
    digit = single_digit_change(l.house,r.house)
    ls,rs = set(l.street.split()),set(r.street.split())
    rawname = ratio(l.name,r.name); leet = ratio(l.leet,r.leet)
    values = [float(both),float(both and offset==0),float(digit),float(both and 0<offset<=30),
              math.log1p(offset),float(both and offset>0 and not digit and offset>30),
              float(bool(l.house)!=bool(r.house)),float(not l.house and not r.house),
              ratio(l.street,r.street),fuzz.token_sort_ratio(l.street,r.street)/100 if ls and rs else 0.,
              len(ls&rs)/max(1,len(ls|rs)),ratio(l.address,r.address),
              float(bool(l.address)!=bool(r.address)),float(not l.address and not r.address),
              float(bool(l.state and r.state)),float(bool(l.state) and l.state==r.state),
              float(bool(l.state and r.state) and l.state!=r.state),ratio(l.core,r.core),leet,
              leet-rawname,ratio(l.core,r.core)-rawname,l.pseudo,r.pseudo,
              math.log1p(name_count),math.log1p(city_count),float(bool(l.city))]
    assert len(values)==len(NAMES)
    return values

"""How many uncertain French test pairs become near-exact once French abbreviations are expanded?"""
import glob, random, re, sqlite3, sys, unicodedata, collections
import numpy as np
from rapidfuzz import fuzz
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

FR_ADDR = {'r': 'rue', 'av': 'avenue', 'ave': 'avenue', 'bd': 'boulevard', 'bld': 'boulevard', 'blvd': 'boulevard',
           'imp': 'impasse', 'crs': 'cours', 'chem': 'chemin', 'ch': 'chemin', 'all': 'allee', 'pl': 'place',
           'psg': 'passage', 'rte': 'route', 'fbg': 'faubourg', 'sq': 'square', 'qu': 'quai', 'res': 'residence',
           'no': '', 'n': '', 'nº': '', 'n°': ''}
REGIONS = {'hauts de france', 'nouvelle aquitaine', 'pays de la loire', 'nord', 'pas de calais', 'loire atlantique',
           'gironde', 'ile de france', 'france'}

def fold(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()

def base(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', fold(s)).split())

def fr(s):
    s = base(s)
    for r in sorted(REGIONS, key=len, reverse=True):
        s = re.sub(rf'\b{r}\b', ' ', s)
    s = re.sub(r'\b\d{5}\b', ' ', s)
    return ' '.join(t for t in (FR_ADDR.get(t, t) for t in s.split()) if t)

def nums(s):
    return re.findall(r'\d+', s)

test = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)
shards = sorted(glob.glob('artifacts/improvement_v2/inference/submission_controller_v1/collected/submission_100k_*/shards/*/'))
random.seed(0)
stats = collections.Counter(); examples = []
for sd in random.sample(shards, 60):
    z = np.load(sd + 'pair_ids.npz', allow_pickle=True); p = np.load(sd + 'probabilities.npy')
    for i in np.where((p >= 0.1) & (p < 0.76))[0]:
        q = z['query_ids'][i]
        if str(z['countries'][q]) != 'France':
            continue
        a = test.execute('select name,address from anchors where entity_id=?', (str(z['reference_ids'][q]),)).fetchone()
        b = test.execute('select name,address from records where entity_id=?', (str(z['target_ids'][i]),)).fetchone()
        stats['pairs'] += 1
        before = fuzz.token_set_ratio(base(a[1]), base(b[1]))
        after = fuzz.token_set_ratio(fr(a[1]), fr(b[1]))
        same_name = core(a[0]) == core(b[0])
        same_house = bool(nums(a[1])) and bool(nums(b[1])) and nums(a[1])[0] == nums(b[1])[0]
        if after >= 95 and before < 95:
            stats['address becomes near-exact only after French expansion'] += 1
            if same_name:
                stats['  ...and core name identical'] += 1
            if len(examples) < 12:
                examples.append((float(p[i]), a, b))
        if after >= 95 and same_name and same_house:
            stats['name identical + address near-exact + same house number (after expansion)'] += 1
for k, v in stats.items():
    print(f'{v:6d}  {k}')
for p, a, b in examples:
    print(f'{p:.2f} {a[0]} | {a[1]}\n      -> {b[0]} | {b[1]}')

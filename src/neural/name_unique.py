import json, re, sqlite3, collections, unicodedata

LEGAL = {'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private', 'llp', 'lp',
         'pllc', 'pc', 'pa', 'plc', 'group', 'the', 'and', 'of'}

def core(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = s.replace('0', 'o').replace('1', 'l').replace('8', 'b')
    toks = [t for t in re.findall(r'[a-z]+', s) if t not in LEGAL]
    return ' '.join(sorted(set(toks)))

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
cnt = collections.Counter()
for country, name in con.execute('select country, name from anchors'):
    cnt[(country, core(name))] += 1
print('anchors', sum(cnt.values()))

rows = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
for kind in ['retrieved_false_negative', 'retrieval_miss', 'false_positive']:
    c = collections.Counter()
    for r in rows:
        if r['kind'] != kind:
            continue
        ta = r['target']['business_address'].strip()
        if ta and ta.upper() not in ('NULL', '<NULL>'):
            continue
        n = cnt[(r['reference']['country'], core(r['reference']['business_name']))]
        c['unique' if n == 1 else 'shared'] += 1
    print(kind, 'empty-address targets, claimant core-name among all S1:', dict(c))

# Base rate over labelled empty-address targets
lab = collections.Counter()
q = '''select a.country, a.name from records r join labels l on l.target_id = r.entity_id
       join anchors a on a.entity_id = l.reference_id
       where trim(r.address) = '' limit 200000'''
for country, name in con.execute(q):
    lab['unique' if cnt[(country, core(name))] == 1 else 'shared'] += 1
print('all labelled empty-address true matches, owner core-name:', dict(lab))

"""Read-only probes of missing dictionary and leave-one-target-out peer evidence.

Uses the existing development cache; this is exploratory feature diagnosis,
not training, a new evaluation split, or permission to accept a route wholesale.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
from rapidfuzz import fuzz

from v3_structural_features import folded, LEGAL, HONORIFIC
from v3_fundamental_audit import OUT, ROOT


def run():
    result_path = OUT / 'evidence_gaps.json'
    if result_path.exists(): raise FileExistsError(result_path)
    base = ROOT/'artifacts/improvement_v3/transfer_pilot_v1/data'
    with np.load(base/'dev.npz', allow_pickle=False) as z:
        a = {k:z[k] for k in ('left','right','query_ids','reference_ids','y','baseline_100000')}
    needed = set(map(int, np.r_[a['left'],a['right']]))
    records = {}
    with (base/'records.jsonl').open() as f:
        for i,line in enumerate(f):
            if i in needed: records[i] = json.loads(line)
    mapping = json.loads((ROOT/'artifacts/training_lexicon_v1/lexicon.json').read_text())
    removed = json.loads((ROOT/'artifacts/improvement_v3/structural_v1/owner_error_analysis_v1/removed_claims.json').read_text())
    if isinstance(removed,dict): raise ValueError('Unexpected removed-claim schema')
    removed_pairs = {(x['owner'],x['target']) for x in removed}
    p = a['baseline_100000']; accepted = p >= .76
    for i,(q,r) in enumerate(zip(a['query_ids'],a['right'])):
        if (str(a['reference_ids'][q]),records[int(r)]['id']) in removed_pairs: accepted[i] = False
    seeds = defaultdict(list)
    for i in np.flatnonzero(accepted & (p >= .98)):
        seeds[int(a['query_ids'][i])].append(i)
    for q in seeds: seeds[q] = sorted(seeds[q],key=lambda i:p[i],reverse=True)[:5]

    @lru_cache(maxsize=200000)
    def name(i): return folded(records[i]['name'])

    @lru_cache(maxsize=200000)
    def tokens(i): return set(name(i).split())

    stats = defaultdict(Counter); examples = defaultdict(list)
    for i,(li,ri) in enumerate(zip(a['left'],a['right'])):
        li,ri = int(li),int(ri);q=int(a['query_ids'][i]);l,r=records[li],records[ri]
        label=bool(a['y'][i]);selected=bool(accepted[i])
        routes=[]
        if l['country']=='India' and any('\u0900'<=c<='\u0dff' for c in r['name']):
            routes.append('indic_all')
            source = tokens(li)-LEGAL-HONORIFIC
            target = tokens(ri)
            direct=sum(word in target for word in source)
            aligned=sum(bool(({word}|{v for v,count in mapping.get(word,[])}) & target) for word in source)
            if source and aligned>direct and aligned/len(source)>=.8:
                routes.append('indic_existing_dictionary_coverage80_new_evidence')
                if len(source)>=2:routes.append('indic_existing_dictionary_coverage80_min2coretokens')
                if label and not selected and len(examples['dictionary_rejected_true'])<8:
                    examples['dictionary_rejected_true'].append(dict(reference=l,target=r,p=float(p[i]),
                        source_core=sorted(source),target_tokens=sorted(target),direct=direct,aligned=aligned))
        if not selected:
            direct_name=fuzz.token_set_ratio(name(li),name(ri))
            other=[j for j in seeds.get(q,[]) if int(a['right'][j])!=ri]
            sibling=max((fuzz.token_set_ratio(name(int(a['right'][j])),name(ri)) for j in other),default=0.)
            if sibling>=90 and sibling-direct_name>=20:
                routes.append('strong_seed_name_gain20_best90_rejected')
                cross=[j for j in other if records[int(a['right'][j])]['id'][:2]!=r['id'][:2]]
                best_cross=max((fuzz.token_set_ratio(name(int(a['right'][j])),name(ri)) for j in cross),default=0.)
                if best_cross>=90 and best_cross-direct_name>=20:
                    routes.append('cross_source_strong_seed_name_gain20_best90_rejected')
        for route in routes:
            b=stats[route];b['pairs']+=1;b['positive' if label else 'negative']+=1
            b[('TP' if label else 'FP') if selected else ('FN' if label else 'TN')]+=1
            if not selected and p[i]>=.2:
                b['unselected_p20_positive' if label else 'unselected_p20_negative']+=1
    result=dict(complete=True,scope='All 173401 cached dev candidate pairs, existing 100k owner policy replay.',
        dictionary_source='Existing training-only 128-word/288-mapping lexicon; no new labels fitted.',
        sibling_seeds='Retained owner-assigned edges with original p>=.98, at most five per S1; candidate excluded from its own support; no seed selection by labels.',
        groups=dict(stats),examples=dict(examples),
        caution='Route counts are not deployable acceptance rules; shared names, ambiguous alignments and erroneous seeds remain possible. Development has been reused.')
    result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2),flush=True)


if __name__=='__main__': run()

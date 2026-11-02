"""Explain fixed-cohort retrieval misses and retain honest extra-candidate text.

No re-retrieval, positive injection, or fake first-stage probabilities. The raw
CE-ready rows deliberately use None for unavailable p100k; they must never be
fed to the production CPU/CE blend without a separately validated recipe.
"""
from collections import Counter,defaultdict
from pathlib import Path
import pickle

from rapidfuzz import fuzz

from ber.store import connect,lookup
from v3_neural_test_pipeline import read,resources
from v3_structural_features import folded,single_digit_change,house_number
from v3_transfer_data import ROOT,digest,write


def run():
    origin=ROOT/'artifacts/improvement_v3/retrieval_recovery_400_v1'
    output=ROOT/'artifacts/improvement_v3/retrieval_error_slices_v1';output.mkdir(parents=True,exist_ok=False)
    report=read(origin/'report.json')
    if not report['complete']:raise ValueError('Original retrieval pilot incomplete')
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    results={};bound={}
    for split in ('tune','dev'):
        counters=defaultdict(Counter);rows=[];owners=0
        for path in sorted((origin/split).glob('[0-9]*.json')):
            resources();r=read(path);owners+=1;owner=r['owner'];base=set(r['baseline']);truth=set(r['truth'])
            recipes={name:set(ids) for name,ids in r['channels'].items()}
            structured=set().union(*(v for k,v in recipes.items() if not k.startswith('existing_')))
            union=structured|recipes.get('existing_deeper_top100',set());missing=truth-base
            left=lookup(db,'anchors','entity_id',[owner])[owner]
            rights=lookup(db,'records','entity_id',sorted(union|missing))
            if any(right['country']!=left['country'] for right in rights.values()):raise ValueError('Cross-country retrieval')
            for target in missing:
                right=rights[target];country=left['country'];ln,rn=folded(left['name']),folded(right['name'])
                flags=['all',country]
                if not right['address'].strip():flags.append('empty_target_address')
                if any('\u0900'<=c<='\u0dff' for c in right['name']):flags.append('indic_target_name')
                if fuzz.ratio(ln,rn)<30:flags.append('low_name_similarity')
                if single_digit_change(house_number(left['address'],country),house_number(right['address'],country)):
                    flags.append('house_digit_deletion')
                for flag in flags:
                    counters[flag]['missed_by_baseline']+=1
                    counters[flag]['recovered_top50']+=target in recipes.get('existing_deeper_top50',set())
                    counters[flag]['recovered_top100']+=target in recipes.get('existing_deeper_top100',set())
                    counters[flag]['recovered_structured']+=target in structured
                    counters[flag]['recovered_union']+=target in union
            for target in sorted(union-base):
                right=rights[target]
                rows.append((owner,target,int(target in truth),left['country'],None,
                    left['name'],left['address'],right['name'],right['address']))
        if owners!=400:raise ValueError('Fixed retrieval cohort changed')
        dest=output/f'{split}_extra.pkl'
        with dest.open('wb') as handle:pickle.dump(rows,handle)
        bound[dest.name]=digest(dest)
        results[split]=dict(owners=owners,extra_candidates=len(rows),extra_positive=sum(r[2] for r in rows),
            slices=dict(counters),p100k_available=False)
    db.close();write(output/'report.json',dict(complete=True,source_report_sha256=digest(origin/'report.json'),
        splits=results,outputs_sha256=bound,code_sha256=digest(__file__),
        quality_evaluated=False,ready_for_submission=False,
        note='Overlapping descriptive error flags; all additional candidates retained, not just positives. No first-stage context/probability invented. Small US/India cohort; no French recall measurement.'))


if __name__=='__main__':run()

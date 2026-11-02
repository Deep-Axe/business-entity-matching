"""Check whether existing small retrieval pilots add beyond selected broad search."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ber.data import write_json
from ber.shards import file_hash


def cached_pairs(path):
    with np.load(path,allow_pickle=False) as z:
        refs=z['reference_ids']
        pairs={(str(refs[q]),str(t)) for q,t in zip(z['query_ids'],z['target_ids'])}
        gold={(str(refs[q]),str(t)) for q,t,y in zip(z['query_ids'],z['target_ids'],z['y']) if y==1}
    return set(map(str,refs)),pairs,gold


def run(comparison,knn,alias,output):
    comparison,knn,alias,output=map(Path,(comparison,knn,alias,output))
    reports={};sources={}
    broad={s:cached_pairs(comparison/f'gate0_{s}.npz') for s in ('tune','dev')}
    for s in ('tune','dev'):sources[str(comparison/f'gate0_{s}.npz')]=file_hash(comparison/f'gate0_{s}.npz')
    neighbors=set();ngold=set();refs=set()
    for channel in ('word_combined','char_combined'):
        path=knn/f'{channel}_neighbors.tsv';sources[str(path)]=file_hash(path)
        with path.open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                refs.add(row['source1_entity_id'])
                if int(row['rank'])>20:continue
                pair=(row['source1_entity_id'],row['target_id']);neighbors.add(pair)
                if int(row['is_gold']):ngold.add(pair)
    assert refs<=broad['tune'][0], 'kNN references outside comparison'
    missing=neighbors-broad['tune'][1]
    reports['knn_word_address_plus_char_address_top20']=dict(references=len(refs),
        additional_candidates_vs_gate0=len(missing),additional_true_pairs_vs_gate0=len(missing & ngold),
        new_true_pairs=sorted(missing & ngold),
        caveat='Optimistic100k pool includes all positives; not full-gallery recall or classification gain.')
    for split in ('tune','dev'):
        path=alias/f'{split}_rows.jsonl';sources[str(path)]=file_hash(path)
        added=set();correct=set();accepted=set();refs=set()
        with path.open() as f:
            for line in f:
                row=json.loads(line);rid=row['reference_id'];refs.add(rid)
                new={(rid,t) for t in row['new_candidates']}
                added.update(new)
                correct.update(new & {(rid,t) for t in row['gold']})
                accepted.update(new & {(rid,t) for t in row['additive']})
        assert refs<=broad[split][0], 'Alias references outside comparison'
        extra=added-broad[split][1]
        reports['alias_'+split]=dict(references=len(refs),pilot_added_candidates=len(added),
            pilot_added_true_pairs=len(correct),additional_candidates_vs_gate0=len(extra),
            additional_true_pairs_vs_gate0=len(extra & correct),new_true_pairs=sorted(extra & correct),
            pilot_accepted_true_still_missing_from_gate0=len(accepted & correct & extra),
            caveat='Old gate1-seeded alias pilot only; not a new gate0 alias rerun.')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(results=reports,source_sha256=sources,audit_used=False)
    write_json(output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('comparison','knn','alias','output'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))

"""Expand the frozen native-alias hypothesis to all 3k tune and 3k dev owners.

Reuse the first 400 queries verbatim. Retain all new candidate text, never just
positives, for an independent post-submission CE diagnostic. No production edits.
"""
from collections import Counter,defaultdict
from pathlib import Path
import pickle
import time

import numpy as np

from ber.disk_index import DiskRetriever
from ber.store import connect,lookup,record_row
from ber.text import Record
from v3_native_alias_retrieval import queries
from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,PILOT,digest,write


def run():
    output=ROOT/'artifacts/improvement_v3/native_alias_recovery_3k_v1';output.mkdir(parents=True,exist_ok=True)
    pilot=ROOT/'artifacts/improvement_v3/native_alias_recovery_400_v1';dictionary=pilot/'dictionary.json'
    if not read(pilot/'report.json')['complete']:raise ValueError('Initial native alias pilot not complete')
    identity=dict(dictionary_sha256=digest(dictionary),code_sha256=digest(__file__),
        query_code_sha256=digest(ROOT/'tools/v3_native_alias_retrieval.py'),owners_per_split=3000,
        hypothesis='Frozen union of four native word-family queries, top20 each',
        evaluation_labels_used_for_queries=False,dictionary_learned_from_training_labels=True)
    if (output/'identity.json').exists():
        if read(output/'identity.json')!=identity:raise ValueError('Full native pilot identity changed')
    else:write(output/'identity.json',identity)
    mapping=read(dictionary)['mapping'];db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    index=DiskRetriever(ROOT/'artifacts/train_index_v2',transliteration=True,max_df_ratio=None)
    import json
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    start=time.monotonic();splits={};bound={}
    for split,fold in (('tune','tune'),('dev','eval')):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','left','right','query_ids','true_counts')}
        refs=list(map(str,a['reference_ids']));base=defaultdict(set)
        if len(refs)!=3000:raise ValueError('Full native cohort changed')
        for l,r in zip(a['left'],a['right']):base[ids[int(l)]].add(ids[int(r)])
        stats=Counter();patterns=Counter();rows=[];bycountry=defaultdict(Counter)
        for i,owner in enumerate(refs):
            resources();left=lookup(db,'anchors','entity_id',[owner])[owner]
            if left['fold']!=fold:raise ValueError('Evaluation owner overlaps dictionary training')
            path=output/split/f'{i:04d}.json';original=pilot/split/f'{i:04d}.json'
            if path.exists():result=read(path)
            elif original.exists():
                result=read(original);write(path,result)
            else:
                channels={}
                for name,query in queries(index,Record.from_row(record_row(left)),mapping).items():
                    hits=index.searcher.search(query,limit=20,count=False).hits
                    records=lookup(db,'records','row_id',index.searcher.fast_field_values('row_id',[doc for _,doc in hits]))
                    if any(r['country']!=left['country'] for r in records.values()):raise ValueError('Cross-country native query')
                    channels[name]=sorted(r['entity_id'] for r in records.values())
                result=dict(owner=owner,channels=channels);write(path,result)
            if result['owner']!=owner:raise ValueError('Resumed native query owner changed')
            found=set().union(*(set(v) for v in result['channels'].values())) if result['channels'] else set()
            extras=found-base[owner]
            # Ranking is frozen before inspecting any evaluation ownership label.
            truth={r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(owner,))}
            if len(truth)!=a['true_counts'][i]:raise ValueError('Truth-count mismatch')
            right=lookup(db,'records','entity_id',sorted(extras));correct=extras&truth
            for target in sorted(extras):
                r=right[target];rows.append((owner,target,int(target in truth),left['country'],None,
                    left['name'],left['address'],r['name'],r['address']))
            for target in correct:
                r=right[target];patterns['indic_target_name']+=any('\u0900'<=c<='\u0dff' for c in r['name'])
                patterns['empty_target_address']+=not r['address'].strip()
            stats.update(owners=1,extra_candidates=len(extras),extra_positive=len(correct),baseline_retrieval_misses=len(truth-base[owner]))
            bycountry[left['country']].update(owners=1,extra=len(extras),correct=len(correct))
            if (i+1)%100==0:write(output/'progress.json',dict(stage='full_native_retrieval',split=split,**stats,seconds=time.monotonic()-start))
        dest=output/f'{split}_extra.pkl'
        with dest.open('wb') as handle:pickle.dump(rows,handle)
        bound[dest.name]=digest(dest);splits[split]=dict(totals=stats,overlapping_positive_patterns=patterns,countries=dict(bycountry))
    db.close();write(output/'report.json',dict(complete=True,splits=splits,outputs_sha256=bound,
        identity_sha256=digest(output/'identity.json'),seconds=time.monotonic()-start,ready_for_submission=False,
        note='Full existing tune/development reference sets, reused exploratory evaluation. All extra pairs retained with no invented p100k. Recall-only; no matching gain yet.'))


if __name__=='__main__':run()

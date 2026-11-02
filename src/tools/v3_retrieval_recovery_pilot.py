"""Bounded full-gallery retrieval-only pilot on fixed tune/development owners.

Compare deeper existing search with conjunctive name/address and structured
number/street queries. Labels are consulted only after label-free retrieval.
No injected positives, classifier claims, submission edits, or index changes.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import tantivy

from ber.data import read_tsv
from ber.disk_index import DiskRetriever
from ber.retrieval import field_terms
from ber.store import connect, lookup, record_row
from ber.text import Record
from v3_transfer_data import ROOT, PILOT, digest, write


def custom_queries(retriever, record):
    schema=retriever.schema;terms=field_terms(record)
    field={f:retriever._field_parts(f,terms[f])[0] for f in ('name_words','ascii_words','name_grams','address_words')}
    result={}
    def must(parts):
        return tantivy.Query.boolean_query([(tantivy.Occur.Must,q) for q in parts]) if all(q is not None for q in parts) else None
    for name in ('name_words','ascii_words','name_grams'):
        result[name+'_AND_address']=must([field[name],field['address_words']])
    address_terms=terms['address_words']
    numbers=sorted((t for t in address_terms if t.isdigit()),key=lambda x:(retriever.frequency('address_words',x),x))[:3]
    streets=sorted((t for t in address_terms if t.isalpha() and len(t)>=3),key=lambda x:(retriever.frequency('address_words',x) or 10**20,x))[:3]
    number_variants={v for n in numbers for v in (n,str(int(n)),str(int(n)).zfill(len(str(int(n)))+1),str(int(n)).zfill(len(str(int(n)))+2))}
    def either(field,values):
        return tantivy.Query.boolean_query([(tantivy.Occur.Should,tantivy.Query.term_query(schema,field,v,'freq'))
            for v in sorted(values)],minimum_number_should_match=1) if values else None
    result['number_AND_street']=must([either('address_words',number_variants),either('address_words',streets)])
    country=tantivy.Query.const_score_query(tantivy.Query.term_query(schema,'country',record.country,'basic'),0.)
    return {name:must([query,country]) for name,query in result.items() if query is not None}


def run(output,owners_per_split=400):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=dict(owners_per_split=owners_per_split,index='artifacts/train_index_v2',full_gallery=True,
        deeper_k=100,structured_k=20,labels_used_for_retrieval=False,selection='First fixed N existing tune/dev owners',
        code_sha256=digest(__file__),index_config_sha256=digest(ROOT/'artifacts/train_index_v2/config.json'))
    if (output/'config.json').exists():
        if json.loads((output/'config.json').read_text())!=config:raise ValueError('Retrieval pilot identity changed')
    else:write(output/'config.json',config)
    index=DiskRetriever(ROOT/config['index'],transliteration=True,max_df_ratio=None)
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    splits={};start=time.monotonic()
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','left','right','y','query_ids','true_counts')}
        refs=list(map(str,a['reference_ids'][:owners_per_split]));baseline=defaultdict(set)
        baseline_gold=defaultdict(set)
        for l,r,y,q in zip(a['left'],a['right'],a['y'],a['query_ids']):
            if q>=owners_per_split:continue
            key=ids[int(l)];baseline[key].add(ids[int(r)])
            if y:baseline_gold[key].add(ids[int(r)])
        anchors=lookup(db,'anchors','entity_id',refs);summary=defaultdict(Counter);recovered=defaultdict(list)
        for n,owner in enumerate(refs):
            destination=output/split/f'{n:04d}.json'
            if destination.exists():r=json.loads(destination.read_text())
            else:
                row=anchors[owner];record=Record.from_row(record_row(row));stamp=time.monotonic()
                channels=defaultdict(set)
                for rid,hits in index.candidates(record,100):
                    for name,(_,inv_rank) in hits.items():
                        rank=round(1/inv_rank)
                        for k in (50,100):
                            if rank<=k:channels[f'existing_deeper_top{k}'].add(rid)
                for name,query in custom_queries(index,record).items():
                    hits=index.searcher.search(query,limit=20,count=False).hits
                    channels[name].update(index.searcher.fast_field_values('row_id',[addr for _,addr in hits]))
                byrow=lookup(db,'records','row_id',list(set().union(*channels.values())))
                found={name:sorted(byrow[rid]['entity_id'] for rid in rids) for name,rids in channels.items()}
                if any(t['country']!=row['country'] for t in byrow.values()):raise ValueError('Cross-country retrieval')
                # Only now look at ground truth: no effect on channels/ranking.
                truth={r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(owner,))}
                if len(truth)!=a['true_counts'][n]:raise ValueError('Truth count mismatch')
                r=dict(owner=owner,country=row['country'],channels=found,baseline=sorted(baseline[owner]),
                    truth=sorted(truth),seconds=time.monotonic()-stamp)
                write(destination,r)
            if r['owner']!=owner or set(r['baseline'])!=baseline[owner]:raise ValueError('Resumed query identity changed')
            truth=set(r['truth']);base=set(r['baseline'])
            recipes={name:set(c) for name,c in r['channels'].items()}
            recipes['structured_union']=set().union(*(v for k,v in recipes.items() if not k.startswith('existing_')))
            for name,candidates in recipes.items():
                extra=candidates-base;correct=extra&truth;c=summary[name]
                c.update(owners=1,additional_candidates=len(extra),additional_correct=len(correct),
                         baseline_correct=len(base&truth),truth=len(truth),owners_helped=bool(correct))
                recovered[name].extend((owner,t) for t in sorted(correct))
            if (n+1)%20==0:
                write(output/'progress.json',dict(split=split,owners_done=n+1,owners_total=owners_per_split,
                    seconds=time.monotonic()-start));print(split,n+1,flush=True)
        splits[split]=dict(recipes=dict(summary),additional_correct_pairs=dict(recovered))
    db.close();write(output/'report.json',dict(complete=True,config=config,splits=splits,seconds=time.monotonic()-start,
        caveat='Retrieval recall/cost only; no classification gain established. Small fixed cohort; no French labels. Added candidates need honest scoring/training context.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owners-per-split',type=int,default=400);a=p.parse_args()
    if not 1<=a.owners_per_split<=3000:p.error('Bounded cohort 1–3000 owners per split required')
    run(**vars(a))

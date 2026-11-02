"""Train-only native-word aliases, then a full-gallery conjunctive pilot.

Unlike romanized OR expansion, query actual Unicode name tokens and require
multiple source-word families. This is retrieval evidence, never a merge rule.
Production indexes/dictionaries and submissions stay unchanged.
"""
from collections import Counter,defaultdict
from itertools import islice
import math
import time

from anyascii import anyascii
import tantivy

from ber.data import read_tsv
from ber.disk_index import DiskRetriever
from ber.shards import gold_for
from ber.store import connect,lookup,record_row
from ber.text import Record,normalize,SUFFIXES
from learned_query_expansion import align
from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,digest,write

OUT=ROOT/'artifacts/improvement_v3/native_alias_recovery_400_v1'
SOURCE=ROOT/'artifacts/improvement_v2/selection/inputs_v1/train_200k/references.tsv'


def native_align(left,right):
    native=defaultdict(set)
    for token in right.name.split():
        if token.isascii():continue
        roman=normalize(anyascii(token)).split()
        if len(roman)==1:native[roman[0]].add(token)
    return {(a,raw) for a,b in align(left.ascii_name.split(),right.ascii_name.split()) for raw in native.get(b,())}


def learn():
    identity=dict(references_sha256=digest(SOURCE),code_sha256=digest(__file__),
        alignment_sha256=digest(ROOT/'tools/learned_query_expansion.py'),train_only=True,minimum_owners=2,max_variants=16)
    if (OUT/'identity.json').exists():
        if read(OUT/'identity.json')!=identity:raise ValueError('Native alias experiment identity changed')
    else:write(OUT/'identity.json',identity)
    if (OUT/'dictionary.json').exists():
        m=read(OUT/'dictionary.json')
        if not m['complete']:raise ValueError('Incomplete native dictionary')
        return m['mapping']
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    iterator=iter(read_tsv(SOURCE));evidence=Counter();n=used=aligned=0;start=time.monotonic();seen=set()
    while True:
        resources();rows=list(islice(iterator,1000))
        if not rows:break
        ids=[r['entity_id'] for r in rows];anchors=lookup(db,'anchors','entity_id',ids)
        if len(anchors)!=len(ids) or seen.intersection(ids) or any(a['fold']!='train' for a in anchors.values()):
            raise ValueError('Dictionary fit escaped unique train owners')
        if any(record_row(anchors[r['entity_id']])!=r for r in rows):raise ValueError('Training text changed')
        seen.update(ids);selected=[r for r in rows if r['country']=='India' and r['business_name'].isascii()]
        gold=gold_for(db,[r['entity_id'] for r in selected]);targets=lookup(db,'records','entity_id',[t for g in gold.values() for t in g])
        for row in selected:
            left=Record.from_row(row);current=set()
            for target in gold[left.entity_id]:
                raw=targets[target]
                if raw['fold']!='train' or raw['country']!='India':raise ValueError('Nontraining/cross-country alias label')
                if raw['name'].isascii():continue
                current.update(native_align(left,Record.from_row(record_row(raw))));used+=1
            evidence.update(current);aligned+=bool(current)
        n+=len(rows)
        if n%10000==0:write(OUT/'progress.json',dict(stage='train_native_aliases',owners=n,seconds=time.monotonic()-start))
    if n!=200000 or digest(SOURCE)!=identity['references_sha256']:raise ValueError('Wrong native dictionary training selection')
    candidates=defaultdict(list)
    for (left,right),count in evidence.items():
        if count>=2:candidates[left].append((right,count))
    mapping={w:sorted(v,key=lambda x:(-x[1],x[0]))[:16] for w,v in sorted(candidates.items())}
    db.close();write(OUT/'dictionary.json',dict(complete=True,mapping=mapping,training_owners=n,aligned_owners=aligned,
        cross_script_positive_pairs=used,expanded_words=len(mapping),variant_mappings=sum(map(len,mapping.values())),
        seconds=time.monotonic()-start,identity_sha256=digest(OUT/'identity.json'),
        caveat='Same conservative >=3-character/ratio>=30 alignment as original learner; short acronyms and compound-word alignments remain limitations.'))
    return mapping


def queries(index,record,mapping):
    if record.country!='India':return {}
    words=sorted(set(record.ascii_name.split()))
    if not any(mapping.get(w) for w in words):return {}
    core=[];legal=[]
    for word in words:
        variants={word,*(v for v,_ in mapping.get(word,()))}
        terms=[(tantivy.Occur.Should,tantivy.Query.term_query(index.schema,'name_words',v,'freq')) for v in sorted(variants)]
        family=tantivy.Query.boolean_query(terms,minimum_number_should_match=1)
        weight=math.log1p(index.searcher.num_docs/(index.frequency('ascii_words',word)+1))
        query=tantivy.Query.const_score_query(family,weight)
        (legal if word in SUFFIXES else core).append((tantivy.Occur.Should,query))
    if not core:return {}
    address,_=index._field_parts('address_words',record.address_tokens)
    country=tantivy.Query.const_score_query(tantivy.Query.term_query(index.schema,'country','India','basic'),0.)
    result={}
    for label,minimum in [('all_core',len(core)),('one_core_missing',max(1,len(core)-1))]:
        names=tantivy.Query.boolean_query(core,minimum_number_should_match=minimum)
        for use_address in (False,True):
            clauses=[(tantivy.Occur.Must,names),(tantivy.Occur.Must,country),*legal]
            if use_address and address is not None:clauses.append((tantivy.Occur.Should,address))
            result[label+('_address' if use_address else '_name_only')]=tantivy.Query.boolean_query(clauses)
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=True);mapping=learn();origin=ROOT/'artifacts/improvement_v3/retrieval_recovery_400_v1'
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    index=DiskRetriever(ROOT/'artifacts/train_index_v2',transliteration=True,max_df_ratio=None)
    splits={};start=time.monotonic()
    for split,fold in (('tune','tune'),('dev','eval')):
        totals=defaultdict(Counter);n=0
        for path in sorted((origin/split).glob('[0-9]*.json')):
            resources();old=read(path);owner=old['owner'];left=lookup(db,'anchors','entity_id',[owner])[owner]
            if left['fold']!=fold:raise ValueError('Pilot owner belongs to wrong fold')
            dest=OUT/split/path.name
            if dest.exists():row=read(dest)
            else:
                channels={}
                for name,query in queries(index,Record.from_row(record_row(left)),mapping).items():
                    hits=index.searcher.search(query,limit=20,count=False).hits
                    byrow=lookup(db,'records','row_id',index.searcher.fast_field_values('row_id',[doc for _,doc in hits]))
                    if any(r['country']!=left['country'] for r in byrow.values()):raise ValueError('Cross-country native retrieval')
                    channels[name]=sorted(r['entity_id'] for r in byrow.values())
                row=dict(owner=owner,channels=channels);write(dest,row)
            if row['owner']!=owner:raise ValueError('Resumed native query owner changed')
            # Ranking is now finished; labels are used only for measurement.
            base=set(old['baseline']);truth=set(old['truth']);deeper=set().union(*(set(v) for v in old['channels'].values()))
            sets={k:set(v) for k,v in row['channels'].items()};sets['union']=set().union(*sets.values()) if sets else set()
            for name,found in sets.items():
                extra=found-base;correct=extra&truth
                totals[name].update(owners=1,extra_candidates=len(extra),extra_correct=len(correct),beyond_deeper=len(correct-deeper))
            n+=1
            if n%25==0:write(OUT/'progress.json',dict(stage='full_gallery_native_queries',split=split,owners=n,seconds=time.monotonic()-start))
        if n!=400:raise ValueError('Fixed native retrieval cohort changed')
        splits[split]=dict(owners=n,channels=dict(totals))
    db.close();write(OUT/'report.json',dict(complete=True,splits=splits,seconds=time.monotonic()-start,
        dictionary_sha256=digest(OUT/'dictionary.json'),source_report_sha256=digest(origin/'report.json'),
        ready_for_submission=False,external_data_used=False,
        note='Only additional candidate recall/cost. Baseline retained. No classifier gain measured; no French labels or score forecast.'))


if __name__=='__main__':run()

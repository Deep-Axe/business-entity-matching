"""Bounded website-style name retrieval over the existing supplied-data index.

No web/domain lookup is made. These are deterministic alternate query strings,
not asserted business aliases. Ranking never reads evaluation labels. All extra
pairs are retained for a later precision check; no submission is modified.
"""
from collections import Counter,defaultdict
from pathlib import Path
import pickle
import re
import time

from anyascii import anyascii
import numpy as np
import tantivy

from ber.disk_index import DiskRetriever
from ber.store import connect,lookup,record_row
from ber.text import Record,SUFFIXES
from v3_neural_test_pipeline import resources,read
from v3_transfer_data import ROOT,PILOT,digest,write

LEGAL=set(SUFFIXES)|{'inc','corp','co','ltd','pvt','llp','llc','sarl','sas','sasu','sci','eurl','sa'}
STOP={'and','the','of'}


def variants(record):
    raw=[w for w in re.findall(r'[a-z0-9]+',anyascii(record.original_name).lower()) if w not in STOP]
    normalized=[w for w in record.ascii_name.split() if w not in STOP]
    full=set();initials=set()
    for words in (raw,normalized,[w for w in raw if w not in LEGAL],[w for w in normalized if w not in LEGAL]):
        if not words:continue
        # Up to three trailing legal tokens may disappear in website renderings.
        choices=[words]
        reduced=list(words)
        for _ in range(3):
            if len(reduced)<2 or reduced[-1] not in LEGAL:break
            reduced=reduced[:-1];choices.append(reduced)
        for choice in choices:
            full.add(''.join(choice))
            if len(choice)>1:
                initials.add(''.join(w[0] for w in choice[:-1])+choice[-1])
                initials.add(''.join(w[0] for w in choice))
    valid=lambda items:sorted(t for t in items if 4<=len(t)<=96 and not t.isdigit())[:24]
    return {'joined':valid(full),'initials_last_word':valid(initials-full)}


def queries(index,record):
    country=tantivy.Query.const_score_query(tantivy.Query.term_query(index.schema,'country',record.country,'basic'),0.)
    def either(values):
        return tantivy.Query.boolean_query([(tantivy.Occur.Should,tantivy.Query.term_query(index.schema,'name_words',v,'freq'))
            for v in values],minimum_number_should_match=1)
    suffix=either(['com','net','org','in'])
    address,_=index._field_parts('address_words',record.address_tokens)
    result={}
    for name,words in variants(record).items():
        if not words:continue
        base=[(tantivy.Occur.Must,either(words)),(tantivy.Occur.Must,suffix),(tantivy.Occur.Must,country)]
        result[name]=tantivy.Query.boolean_query(base)
        if address is not None:
            result[name+'_address_bonus']=tantivy.Query.boolean_query([*base,(tantivy.Occur.Should,address)])
    return result


def run():
    out=ROOT/'artifacts/improvement_v3/website_alias_recovery_3k_v1';out.mkdir(parents=True,exist_ok=True)
    identity=dict(code_sha256=digest(__file__),index_config_sha256=digest(ROOT/'artifacts/train_index_v2/config.json'),
        owners_per_split=3000,k=20,evaluation_labels_used_for_queries=False,
        hypothesis='Website-concatenated names and initial+last-word aliases; proposed after earlier dev error inspection')
    if (out/'identity.json').exists():
        if read(out/'identity.json')!=identity:raise ValueError('Website alias retrieval identity changed')
    else:write(out/'identity.json',identity)
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    index=DiskRetriever(ROOT/'artifacts/train_index_v2',transliteration=True,max_df_ratio=None)
    import json
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    start=time.monotonic();splits={};hashes={}
    for split,fold in (('tune','tune'),('dev','eval')):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','left','right','true_counts')}
        refs=list(map(str,a['reference_ids']));base=defaultdict(set)
        for l,r in zip(a['left'],a['right']):base[ids[int(l)]].add(ids[int(r)])
        if len(refs)!=3000:raise ValueError('Website pilot owner cohort changed')
        stats=Counter();countries=defaultdict(Counter);rows=[];channels_stats=defaultdict(Counter)
        for i,o in enumerate(refs):
            resources();left=lookup(db,'anchors','entity_id',[o])[o]
            if left['fold']!=fold:raise ValueError('Wrong evaluation fold')
            path=out/split/f'{i:04d}.json'
            if path.exists():result=read(path)
            else:
                channels={}
                for name,query in queries(index,Record.from_row(record_row(left))).items():
                    hits=index.searcher.search(query,limit=20,count=False).hits
                    found=lookup(db,'records','row_id',index.searcher.fast_field_values('row_id',[doc for _,doc in hits]))
                    if any(r['country']!=left['country'] for r in found.values()):raise ValueError('Cross-country website query')
                    channels[name]=sorted(r['entity_id'] for r in found.values())
                result=dict(owner=o,channels=channels);write(path,result)
            if result['owner']!=o:raise ValueError('Resumed query owner changed')
            extra=set().union(*(set(v) for v in result['channels'].values()))-base[o] if result['channels'] else set()
            # Evaluate only after query ranking has been fixed and cached.
            truth={r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))}
            if len(truth)!=a['true_counts'][i]:raise ValueError('Truth count mismatch')
            right=lookup(db,'records','entity_id',sorted(extra))
            for t in sorted(extra):
                r=right[t];rows.append((o,t,int(t in truth),left['country'],None,left['name'],left['address'],r['name'],r['address']))
            stats.update(owners=1,extra_candidates=len(extra),extra_positive=len(extra&truth),baseline_retrieval_misses=len(truth-base[o]))
            countries[left['country']].update(extra=len(extra),correct=len(extra&truth))
            for name,found in result['channels'].items():
                new=set(found)-base[o];channels_stats[name].update(extra=len(new),correct=len(new&truth))
            if (i+1)%100==0:write(out/'progress.json',dict(stage='website_query_pilot',split=split,**stats,seconds=time.monotonic()-start))
        dest=out/f'{split}_extra.pkl'
        with dest.open('wb') as handle:pickle.dump(rows,handle)
        hashes[dest.name]=digest(dest);splits[split]=dict(totals=stats,countries=dict(countries),channels=dict(channels_stats))
    db.close();write(out/'report.json',dict(complete=True,splits=splits,outputs_sha256=hashes,identity_sha256=digest(out/'identity.json'),
        seconds=time.monotonic()-start,ready_for_submission=False,external_data_used=False,
        note='Retrieval recall only; extra pairs retain no invented p100k. No website lookup or French score claim.'))


if __name__=='__main__':run()

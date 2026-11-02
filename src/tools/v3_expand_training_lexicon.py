"""Stream the established token-alignment learner over 200k train-only S1s.

Keep the production 12k dictionary untouched. Each owner votes once for each
mapping, even with multiple duplicate targets. SQLite batch commits support
safe resume; no tune/eval/test labels are read when fitting this dictionary.
"""
from collections import Counter,defaultdict
import fcntl
from itertools import islice
from pathlib import Path
import sqlite3
import time

from ber.data import read_tsv,write_tsv
from ber.store import connect,lookup,record_row
from ber.shards import gold_for
from ber.text import Record
from learned_query_expansion import align
from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,digest,write

OUT=ROOT/'artifacts/improvement_v3/lexicon200k_v1'
SOURCE=ROOT/'artifacts/improvement_v2/selection/inputs_v1/train_200k/references.tsv'


def run():
    OUT.mkdir(parents=True,exist_ok=True);lease=(OUT/'lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(references_sha256=digest(SOURCE),code_sha256=digest(__file__),
        alignment_code_sha256=digest(ROOT/'tools/learned_query_expansion.py'),expected=200000,
        training_only=True,minimum_distinct_owners=2,variants_per_word=6)
    if (OUT/'identity.json').exists():
        if read(OUT/'identity.json')!=identity:raise ValueError('Dictionary identity changed')
    else:write(OUT/'identity.json',identity)
    if (OUT/'manifest.json').exists():
        saved=read(OUT/'manifest.json')
        if not saved['complete'] or digest(OUT/'lexicon.json')!=saved['lexicon_sha256']:raise ValueError('Completed dictionary changed')
        return
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    work=sqlite3.connect(OUT/'evidence.sqlite3');work.execute('PRAGMA cache_size=-4096')
    work.executescript('''CREATE TABLE IF NOT EXISTS evidence(left_word TEXT,right_word TEXT,n INTEGER NOT NULL,
        PRIMARY KEY(left_word,right_word)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS owners(owner TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,n INTEGER NOT NULL);''')
    state=dict(work.execute('SELECT key,n FROM state'))
    processed=state.get('processed',0);aligned=state.get('aligned',0);used=state.get('used',0)
    iterator=iter(read_tsv(SOURCE));start=time.monotonic()
    for _ in range(processed):next(iterator)
    while True:
        resources();rows=list(islice(iterator,1000))
        if not rows:break
        ids=[r['entity_id'] for r in rows];anchors=lookup(db,'anchors','entity_id',ids)
        if len(anchors)!=len(ids) or any(a['fold']!='train' for a in anchors.values()):raise ValueError('Dictionary fit escaped unique train owners')
        if any(record_row(anchors[r['entity_id']])!=r for r in rows):raise ValueError('Training text does not match store')
        selected=[r for r in rows if r['country']=='India' and r['business_name'].isascii()]
        gold=gold_for(db,[r['entity_id'] for r in selected])
        targets=lookup(db,'records','entity_id',[t for g in gold.values() for t in g])
        evidence=Counter()
        for row in selected:
            left=Record.from_row(row);current=set()
            for target in gold[left.entity_id]:
                raw=targets[target]
                if raw['fold']!='train' or raw['country']!='India':raise ValueError('Cross-fold/country dictionary label')
                if raw['name'].isascii():continue
                right=Record.from_row(record_row(raw));current.update(align(left.ascii_name.split(),right.ascii_name.split()));used+=1
            evidence.update(current);aligned+=bool(current)
        processed+=len(rows)
        with work:
            work.executemany('INSERT INTO owners VALUES(?)',[(o,) for o in ids])
            work.executemany('INSERT INTO evidence VALUES(?,?,?) ON CONFLICT(left_word,right_word) DO UPDATE SET n=n+excluded.n',
                [(a,b,n) for (a,b),n in evidence.items()])
            work.executemany('INSERT OR REPLACE INTO state VALUES(?,?)',[('processed',processed),('aligned',aligned),('used',used)])
        write(OUT/'progress.json',dict(stage='train_only_alignment',owners=processed,expected=200000,aligned_owners=aligned,
            cross_script_positive_pairs=used,seconds=time.monotonic()-start))
    if processed!=200000 or work.execute('SELECT count(*) FROM owners').fetchone()[0]!=200000:raise ValueError('Incomplete dictionary owner selection')
    candidates=defaultdict(list)
    for a,b,n in work.execute('SELECT left_word,right_word,n FROM evidence WHERE n>=2 AND left_word!=right_word'):
        candidates[a].append((b,n))
    lexicon={word:sorted(values,key=lambda x:(-x[1],x[0]))[:6] for word,values in sorted(candidates.items())}
    write(OUT/'lexicon.json',lexicon)
    write_tsv(OUT/'training_reference_ids.tsv',({'entity_id':r[0]} for r in work.execute('SELECT owner FROM owners ORDER BY owner')),['entity_id'])
    if digest(SOURCE)!=identity['references_sha256']:raise ValueError('Training selection changed during alignment')
    write(OUT/'manifest.json',dict(complete=True,training_references=processed,aligned_references=aligned,
        positive_cross_script_pairs_used=used,expanded_words=len(lexicon),variant_mappings=sum(map(len,lexicon.values())),
        min_distinct_training_references=2,max_variants_per_word=6,external_data_used=False,
        identity_sha256=digest(OUT/'identity.json'),lexicon_sha256=digest(OUT/'lexicon.json'),
        seconds=time.monotonic()-start,warning='Heuristic training-token correspondences, not a verified linguistic dictionary. Not deployed.'))
    db.close();work.close()


if __name__=='__main__':run()

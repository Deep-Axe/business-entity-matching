"""Disk-backed, resumable S1 name/city counts for structural CE rescoring.

Reads only provided S1 text. No labels, target data, or external data. Counts
all S1s, including owners without routed candidates. Checkpoints each batch
transactionally; a partial batch is never counted twice on resume.
"""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
from itertools import islice
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from ber.data import read_tsv
from v3_neural_test_pipeline import DATA,resources
from v3_structural_features import core_name,city_proxy,View,pair_features,NAMES
from v3_transfer_data import ROOT,digest,write


def open_db(path,readonly=False):
    path=Path(path)
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True) if readonly else sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    if not readonly:
        db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS names(country TEXT,core TEXT,n INTEGER NOT NULL,
            PRIMARY KEY(country,core)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS cities(country TEXT,core TEXT,city TEXT,n INTEGER NOT NULL,
            PRIMARY KEY(country,core,city)) WITHOUT ROWID;
        ''')
    return db


def add_batch(db,rows,processed):
    names=Counter();cities=Counter()
    for row in rows:
        name=row['name'] if 'name' in row else row['business_name']
        address=row['address'] if 'address' in row else row['business_address']
        key=(row['country'],core_name(name))
        names[key]+=1;city=city_proxy(address,row['country'])
        if city:cities[(*key,city)]+=1
    with db:
        db.executemany('INSERT INTO names VALUES(?,?,?) ON CONFLICT(country,core) DO UPDATE SET n=n+excluded.n',
                       [(*key,n) for key,n in names.items()])
        db.executemany('INSERT INTO cities VALUES(?,?,?,?) ON CONFLICT(country,core,city) DO UPDATE SET n=n+excluded.n',
                       [(*key,n) for key,n in cities.items()])
        db.execute("INSERT OR REPLACE INTO meta VALUES('processed',?)",(str(processed),))


class Counts:
    def __init__(self,path):self.db=open_db(path,True)

    @lru_cache(maxsize=30000)
    def get(self,country,core,city):
        row=self.db.execute('SELECT n FROM names WHERE country=? AND core=?',(country,core)).fetchone()
        if row is None:raise ValueError('S1 name absent from complete text counts')
        ncity=self.db.execute('SELECT n FROM cities WHERE country=? AND core=? AND city=?',(country,core,city)).fetchone() if city else None
        return row[0],ncity[0] if ncity else 0

    def close(self):self.get.cache_clear();self.db.close()


def matrix(rows,counts,fragments):
    @lru_cache(maxsize=60000)
    def view(name,address,country):return View.make(name,address,country,fragments)
    x=np.empty((len(rows),len(NAMES)),np.float32)
    for i,r in enumerate(rows):
        left,right=view(r[5],r[6],r[3]),view(r[7],r[8],r[3])
        n,ncity=counts.get(r[3],left.core,left.city)
        x[i]=pair_features(left,right,n,ncity)
    if not np.isfinite(x).all():raise ValueError('Invalid structural matrix')
    return x


def run(output,source=DATA/'test_source1.tsv',expected=1732544):
    output=Path(output);source=Path(source);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(source=str(source),source_sha256=digest(source),expected=expected,
        script_sha256=digest(__file__),features_sha256=digest(ROOT/'tools/v3_structural_features.py'))
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text())!=identity:raise ValueError('Name-count identity changed')
    else:write(output/'identity.json',identity)
    if (output/'completion.json').exists():
        done=json.loads((output/'completion.json').read_text())
        if done['database_sha256']!=digest(output/'counts.sqlite3'):raise ValueError('Complete name counts changed')
        return
    db=open_db(output/'counts.sqlite3');row=db.execute("SELECT value FROM meta WHERE key='processed'").fetchone()
    processed=int(row[0]) if row else 0;start=time.monotonic();iterator=iter(read_tsv(source))
    for _ in range(processed):next(iterator)
    while True:
        resources();batch=list(islice(iterator,5000))
        if not batch:break
        processed+=len(batch);add_batch(db,batch,processed)
        write(output/'progress.json',dict(stage='counting_s1_text',processed=processed,expected=expected,seconds=time.monotonic()-start))
    if processed!=expected or db.execute('SELECT sum(n) FROM names').fetchone()[0]!=expected:
        raise ValueError('Incomplete S1 name-count coverage')
    stats={table:db.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ('names','cities')}
    db.close()
    if digest(source)!=identity['source_sha256']:raise ValueError('S1 input changed while counting')
    write(output/'completion.json',dict(complete=True,references=processed,identity_sha256=digest(output/'identity.json'),
        database_sha256=digest(output/'counts.sqlite3'),groups=stats,seconds=time.monotonic()-start,
        source='All provided S1 text only; no labels; unknown city is not a match/conflict'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=DATA/'test_source1.tsv');p.add_argument('--expected',type=int,default=1732544)
    run(**vars(p.parse_args()))

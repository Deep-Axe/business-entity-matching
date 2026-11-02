"""Check numerical identity before claiming any structural-feature speed gain."""
import argparse
from pathlib import Path
import pickle
import resource
import time

import numpy as np

from v3_neural_name_counts import Counts,matrix as original
from v3_structural_fast import matrix as fast
from v3_neural_test_pipeline import SCRATCH,read,resources
from v3_transfer_data import ROOT,digest,write


def run(output,limit=10000):
    resources();counts_root=ROOT/'artifacts/improvement_v3/test_s1_structural_counts_v1'
    done=read(counts_root/'completion.json')
    if not done['complete'] or digest(counts_root/'counts.sqlite3')!=done['database_sha256']:
        raise ValueError('Full S1 counts changed')
    rows=pickle.load((SCRATCH/'test/part_000.pkl').open('rb'))
    ix=np.linspace(0,len(rows)-1,min(len(rows),limit),dtype=int);rows=[rows[i] for i in ix]
    fragments=read(ROOT/'artifacts/improvement_v3/structural_v1/feature_pilot_v2/fragments.json')['fragments']
    counts=Counts(counts_root/'counts.sqlite3');start=time.monotonic();a=original(rows,counts,fragments);t1=time.monotonic()-start;counts.close()
    counts=Counts(counts_root/'counts.sqlite3');start=time.monotonic();b=fast(rows,counts,fragments);t2=time.monotonic()-start;counts.close()
    if not np.array_equal(a,b):raise ValueError('Memoized features differ from canonical values')
    write(output,dict(complete=True,pairs=len(rows),dimensions=a.shape[1],bitwise_equal=True,
        max_absolute_delta=float(np.max(np.abs(a-b))),original_seconds=t1,memoized_seconds=t2,
        wall_speedup=t1/t2,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256=digest(ROOT/'tools/v3_structural_features.py'),optimized_sha256=digest(ROOT/'tools/v3_structural_fast.py'),
        note='Small equal-input test under concurrent load. No accuracy claim or change to canonical source.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10000);a=p.parse_args()
    if not 1<=a.limit<=20000:p.error('Bounded parity benchmark requires 1 to 20000 rows')
    run(**vars(a))

"""Small CPU-only timing/coverage benchmark, never a submission or score claim."""
import argparse
from pathlib import Path
import pickle
import resource
import time

import joblib
import numpy as np

from v3_neural_name_counts import Counts,matrix
from v3_neural_test_pipeline import SCRATCH,read,resources,feature_functions,edit_matrix,lg
from v3_transfer_data import ROOT,digest,write


def run(counts_root,output,limit=5000):
    counts_root,output=Path(counts_root),Path(output)
    done=read(counts_root/'completion.json')
    if not done['complete'] or done['database_sha256']!=digest(counts_root/'counts.sqlite3'):
        raise ValueError('Full S1 text counts not ready')
    resources();counts=Counts(counts_root/'counts.sqlite3')
    stats=[dict(country=c,references=int(n),shared_core_references=int(s),max_core_group=int(m))
        for c,n,s,m in counts.db.execute('SELECT country,sum(n),sum(CASE WHEN n>1 THEN n ELSE 0 END),max(n) FROM names GROUP BY country')]
    source=SCRATCH/'test/part_000.pkl';all_rows=pickle.load(source.open('rb'))
    ix=np.linspace(0,len(all_rows)-1,min(limit,len(all_rows)),dtype=int);rows=[all_rows[i] for i in ix];del all_rows
    fragments=read(ROOT/'artifacts/improvement_v3/structural_v1/feature_pilot_v2/fragments.json')['fragments']
    start=time.monotonic();ops=edit_matrix(rows,feature_functions());edit_seconds=time.monotonic()-start
    start=time.monotonic();extra=matrix(rows,counts,fragments);structural_seconds=time.monotonic()-start
    model=joblib.load(ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1/evaluation/ops_ce_extra.joblib')
    # Neutral CE values test dimensional compatibility and speed only; never
    # expose these predictions as real matching results or a quality metric.
    x=np.column_stack((ops,lg([r[4] for r in rows]),np.zeros(len(rows)),extra)).astype(np.float32)
    start=time.monotonic();pred=model.predict_proba(x)[:,1];predict_seconds=time.monotonic()-start
    if not np.isfinite(pred).all():raise ValueError('Invalid benchmark predictions')
    counts.close();write(output,dict(complete=True,pairs=len(rows),dimensions=x.shape[1],
        seconds=dict(edit=edit_seconds,structural=structural_seconds,predict=predict_seconds),
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,full_test_name_counts=stats,
        counts_manifest_sha256=digest(counts_root/'completion.json'),source_sha256=digest(source),
        quality_evaluated=False,ready_for_submission=False,
        note='Neutral neural input for speed/dimension test only. Wall time includes contention with other bounded jobs; full-test time also includes I/O, assignment, assembly and validators.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=5000);args=p.parse_args()
    if not 1<=args.limit<=20000:p.error('Small bounded benchmark: 1 to 20000 rows')
    run(**vars(args))

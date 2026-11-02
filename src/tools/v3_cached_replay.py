"""Reconstruct68 deployment-safe baseline features from saved IDs, without search."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from ber.features import pair_features,FEATURE_NAMES
from ber.text import Record
from ber.store import connect,lookup,record_row
from v2_cpu_features import Weighting
from v2_run_pipeline import enhanced_matrix,NAMES
from v3_transfer_data import ROOT,digest,write

REPLAY_COLUMNS=[i for i,n in enumerate(NAMES) if n not in set(FEATURE_NAMES[30:])]


def replay(a,rows,target_rows,weighting):
    left=[Record.from_row(r) for r in rows]
    right={eid:Record.from_row(record_row(r)) for eid,r in target_rows.items()}
    # Empty hits are temporary placeholders ONLY for excluded retrieval columns;
    # no first-stage model is ever evaluated on these placeholders.
    X=np.array([pair_features(left[int(q)],right[str(e)],{}) for q,e in zip(a['query_ids'],a['target_ids'])],dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
    reconstructed=dict(a,X=X)
    return enhanced_matrix(reconstructed,rows,target_rows,weighting)[:,REPLAY_COLUMNS]


def check(output):
    prepared=ROOT/'artifacts/improvement_v2/cpu/prepared_gate0_full_tune_v1'
    m=json.loads((prepared/'manifest.json').read_text());s=m['shards'][0]
    path=Path(m['source'])/'shards'/s['shard']/'features.npz'
    if digest(path)!=s['features_sha256']:raise ValueError('Cached candidates changed')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in ('query_ids','reference_ids','target_ids','countries')}
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-4096')
    anchors=lookup(db,'anchors','entity_id',a['reference_ids'].tolist())
    targets=lookup(db,'records','entity_id',np.unique(a['target_ids']).tolist())
    weighting=Weighting(ROOT/'artifacts/train_index_v2');start=time.monotonic()
    actual=replay(a,[record_row(anchors[str(r)]) for r in a['reference_ids']],targets,weighting)
    seconds=time.monotonic()-start
    expected=np.load(prepared/'X.npy',mmap_mode='r')[s['pair_start']:s['pair_end'],REPLAY_COLUMNS]
    delta=float(np.max(np.abs(actual-expected))) if actual.size else 0.
    if not np.array_equal(actual,expected):raise ValueError(f'Cached-only feature replay differs: max{delta}')
    result={'passed':True,'columns':len(REPLAY_COLUMNS),'pairs':len(actual),'owners':len(a['reference_ids']),
            'max_absolute_difference':delta,'seconds_excluding_initial_lookup':seconds,
            'retrieval_performed':False,'first_stage_prediction_on_placeholder_features':False,
            'source_features_sha256':digest(path),'feature_names':[NAMES[i] for i in REPLAY_COLUMNS],
            'note':'Timing is a local sample, not a full-test throughput guarantee; new structural/global-rival features are additional.'}
    write(output,result);print(json.dumps(result,indent=2));db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/improvement_v3/structural_v1/cached_replay_parity.json')
    check(p.parse_args().output)

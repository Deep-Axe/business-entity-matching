"""Replay one collected cloud test shard locally, without consulting any labels."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from ber.data import FIELDS,read_tsv,write_tsv
from ber.shards import file_hash
from v2_prediction_integrity import read_cpu_release,validate_config,validate_shard

def run(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    cfg=json.loads((source/'config.json').read_text())
    release=read_cpu_release('artifacts/improvement_v2/cpu/release_gate0_100k_cpu_v1.json')
    validate_config(cfg,release,audit=False)
    candidates=sorted((source/'shards').glob('[0-9]*/done.json'))
    if not candidates:raise ValueError('No committed cloud shard collected yet')
    folder=candidates[0].parent;shard=int(folder.name);size=cfg['query_batch_size']
    references=Path('6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv')
    if file_hash(references)!=cfg['references_sha256']:raise ValueError('Reference hash differs')
    rows=list(itertools.islice(read_tsv(references),shard*size,(shard+1)*size))
    done,arrays,probabilities,_=validate_shard(folder,rows,cfg,file_hash(source/'config.json'))
    output.mkdir(parents=True)
    selected=output/'references.tsv';write_tsv(selected,rows,FIELDS)
    local=output/'local_predictions'
    argv=[sys.executable,'tools/v2_run_pipeline.py','--index','artifacts/test_index_v2',
        '--store','artifacts/test_store_v2','--references',str(selected),'--lexicon','artifacts/training_lexicon_v1',
        '--model','artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm',
        '--model-report','artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_report.json',
        '--training-manifest','artifacts/improvement_v2/cpu/prepared_gate0_train_100k_v1/manifest.json',
        '--output',str(local),'--threshold','.76','--weighting-policy','forward_gallery_index_idf_v1',
        '--gate-terms','0','--workers','1']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');env.pop('PYTHONPATH',None)
    subprocess.run(argv,env=env,check=True)
    lc=json.loads((local/'config.json').read_text())
    _,la,lp,_=validate_shard(local/'shards/000000',rows,lc,file_hash(local/'config.json'))
    if any(not np.array_equal(arrays[k],la[k]) for k in arrays):raise ValueError('Candidate/reference arrays differ crosshost')
    delta=float(np.max(np.abs(probabilities-lp))) if len(lp) else 0.
    identical=np.array_equal(probabilities,lp)
    if delta>1e-12 or not np.array_equal(probabilities>=.76,lp>=.76):
        raise ValueError('Crosshost prediction discrepancy')
    for name in ('candidate_pairs.tsv','matching_results.tsv'):
        if file_hash(folder/name)!=file_hash(local/'shards/000000'/name):raise ValueError('Crosshost TSV differs')
    report=dict(complete=True,labels_accessed=False,source=str(source),shard=shard,references=len(rows),
        candidate_pairs=len(lp),probabilities_bit_identical=identical,max_absolute_probability_delta=delta,
        tsvs_bit_identical=True,cloud_done_sha256=file_hash(folder/'done.json'),
        local_done_sha256=file_hash(local/'shards/000000/done.json'))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))

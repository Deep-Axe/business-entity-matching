"""Combine two independently checked feature blocks in an isolated CPU pilot.

Reuse fixed owner fitting/calibration splits, report development, do not alter
the running 56-feature deployment. No GPU, retrieval, test scoring or upload.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

from v3_neural_test_pipeline import read, resources
from v3_transfer_data import ROOT, digest, write


def aligned_join(a,b,rows):
    if a.ndim!=2 or b.ndim!=2 or len(a)!=rows or len(b)!=rows:
        raise ValueError('Feature block row count or shape differs')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite feature value')
    return np.column_stack((a,b)).astype(np.float32)


def run():
    work=ROOT/'artifacts/improvement_v3/structural_neural_extra_target_frequency_v1'
    work.mkdir(parents=True,exist_ok=False); resources()
    primary=ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1'
    secondary=ROOT/'artifacts/improvement_v3/target_frequency_peerhalf_v1'
    manifests=[read(p/'manifest.json') for p in (primary,secondary)]
    if not all(m['complete'] for m in manifests):raise ValueError('Feature components incomplete')
    names=manifests[0]['names']+manifests[1]['names']
    if len(names)!=27 or len(set(names))!=len(names):raise ValueError('Unexpected combined feature schema')
    bound={}
    for split in ('tune','dev','rivals','train5k'):
        left,right=[m['inputs'][split] for m in manifests]
        if left['sha256']!=right['sha256'] or left['pairs']!=right['pairs']:
            raise ValueError('Feature components refer to different raw rows')
        if digest(left['path'])!=left['sha256']:raise ValueError('Raw feature rows changed')
        arrays=[]
        for root,info in zip((primary,secondary),(left,right)):
            path=root/f'{split}.npy'
            if digest(path)!=info['features_sha256']:raise ValueError('Feature component changed')
            arrays.append(np.load(path,allow_pickle=False))
        values=aligned_join(*arrays,left['pairs']); target=work/f'{split}.npy'; np.save(target,values)
        bound[split]=dict(path=left['path'],sha256=left['sha256'],pairs=left['pairs'],features_sha256=digest(target))
    write(work/'manifest.json',dict(complete=True,names=names,inputs=bound,
        components=[dict(path=str(p/'manifest.json'),sha256=digest(p/'manifest.json')) for p in (primary,secondary)],
        code_sha256=digest(__file__),labels_used_to_build_features=False,
        caveat='Target ASCII-core frequency collapses native-only names; keep this semantic limitation visible.'))
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    results={}
    for fresh in (False,True):
        label='fresh5k' if fresh else 'same_training'; audit=work/label
        write(work/'progress.json',dict(stage='cpu_audit',fit=label,ready_for_submission=False)); resources()
        args=[sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
            '--ce-root',str(peer/'half_calibration'),'--rival-scores',str(peer/'half_calibration/rivals.npy'),
            '--extra-root',str(work),'--output',str(audit)]
        if fresh:args+=['--training-root','artifacts/improvement_v3/fresh_stack_train_v1']
        with (work/(label+'.log')).open('a') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        r=read(audit/'report.json'); results[label]=dict(audit=str(audit),report_sha256=digest(audit/'report.json'),
            selected=r['selected'],development=r['models'][r['selected']['name']]['development'])
    write(work/'report.json',dict(complete=True,variants=results,ready_for_submission=False,
        full_test_scored=False,note='Feature experiment only; not the already selected/running structural deployment. No model chosen by maximizing development.'))
    write(work/'progress.json',dict(stage='complete',ready_for_submission=False,variants=results))


if __name__=='__main__':run()

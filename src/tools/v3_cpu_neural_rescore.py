"""Apply an independently audited CPU stacker to retained, verified CE scores.

No CUDA, retrieval, external data, or portal uploads. Separate output only.
Supports the original 30 inputs and the 26 independently tested structural
inputs. Full S1 name/city counts are on disk. All 128 partitions and official
ID validation are mandatory before a ready receipt is written.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np

from v3_neural_name_counts import Counts
from v3_structural_fast import matrix
from v3_neural_test_pipeline import (SCRATCH,CONTROLLER,STRUCT,DATA,read,lg,resources,
    feature_functions,edit_matrix,partition_inputs,verify_route,assemble)
from v3_owner_assignment import open_claims,load_text
from v3_peer_neural_submission import valid_probabilities
from v3_structural_features import NAMES
from v3_transfer_data import ROOT,digest,write


def supported_columns(report):
    name=report['selected']['name']
    if name.startswith('blend_'):return None
    columns=report['feature_columns'][name]
    if columns not in (list(range(30)),list(range(30+len(NAMES)))):
        raise ValueError('Unsupported audited feature set')
    if len(columns)>30 and report.get('extra_features')!=NAMES:
        raise ValueError('Structural feature names/order differ from audit')
    return columns


def run(audit,source,output,counts_root=None,minutes=360,check_only=False):
    audit,source,output=map(Path,(audit,source,output))
    output.mkdir(parents=True,exist_ok=True)
    lease=(output/'lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60;report=read(audit/'report.json');selected=report['selected']
    if not report['complete']:raise ValueError('Incomplete audit')
    columns=supported_columns(report);metrics=report['models'][selected['name']]['development']
    if metrics['macro_f05']<=report['baseline_replay']['macro_f05']:
        raise ValueError('Audited model is not better than existing 100k owner baseline')
    calibration=source/'half_calibration';parity=read(calibration/'parity.json')
    if not parity['complete'] or parity['test_replay_max_absolute_delta']>2e-4:
        raise ValueError('Source neural scores lack independent precision replay')
    if Path(report['ce_root']).resolve()!=calibration.resolve():
        raise ValueError('CPU calibration is not the retained test score precision')
    for path,value in report['inputs_sha256'].items():
        if digest(path)!=value:raise ValueError('Audited input changed: '+path)
    for name,value in parity['scores_sha256'].items():
        if digest(calibration/name)!=value:raise ValueError('Source CE calibration changed')
    model_path=audit/(selected['name']+'.joblib')
    model=joblib.load(model_path) if columns is not None else None
    functions=feature_functions();extra=columns is not None and len(columns)>30
    fragments_path=ROOT/'artifacts/improvement_v3/structural_v1/feature_pilot_v2/fragments.json'
    fragments=read(fragments_path)['fragments'];counts=None
    if extra:
        if counts_root is None:raise ValueError('Full-gallery structural counts required')
        counts_root=Path(counts_root);done=read(counts_root/'completion.json');count_identity=read(counts_root/'identity.json')
        if (not done['complete'] or done['references']!=1732544
                or done['database_sha256']!=digest(counts_root/'counts.sqlite3')
                or done['identity_sha256']!=digest(counts_root/'identity.json')
                or count_identity['source_sha256']!=digest(DATA/'test_source1.tsv')
                or count_identity['features_sha256']!=digest(ROOT/'tools/v3_structural_features.py')):
            raise ValueError('Incomplete or changed full-test structural counts')
        # The fragment inventory and definitions must be the training recipe.
        candidates=[Path(p) for p in report['inputs_sha256'] if Path(p).name=='manifest.json']
        manifests=[read(p) for p in candidates]
        bound=[m for m in manifests if m.get('names')==NAMES]
        if len(bound)!=1 or bound[0]['fragments_sha256']!=digest(fragments_path) or bound[0]['features_code_sha256']!=digest(ROOT/'tools/v3_structural_features.py'):
            raise ValueError('Structural training/deployment recipe differs')
        counts=Counts(counts_root/'counts.sqlite3')
        feature_parity=read(counts_root/'fast_parity.json')
        if (not feature_parity['complete'] or not feature_parity['bitwise_equal']
                or feature_parity['max_absolute_delta']!=0 or feature_parity['pairs']<10000
                or feature_parity['source_sha256']!=digest(ROOT/'tools/v3_structural_features.py')
                or feature_parity['optimized_sha256']!=digest(ROOT/'tools/v3_structural_fast.py')):
            raise ValueError('Memoized structural implementation lacks exact feature replay')
    # Independently replay the 28 raw edit features used when fitting.
    for split in ('tune','dev'):
        rows=pickle.load((SCRATCH/f'{split}.pkl').open('rb'))
        if not np.array_equal(edit_matrix(rows,functions),np.load(SCRATCH/f'{split}.pkl.ops.npy')):
            raise ValueError('Edit feature definitions do not reproduce fitting inputs')
    del rows
    identity=dict(audit_sha256=digest(audit/'report.json'),source_producer_sha256=digest(source/'producer.json'),
        parity_sha256=digest(calibration/'parity.json'),model_sha256=digest(model_path) if model else None,
        counts_sha256=digest(counts_root/'completion.json') if extra else None,
        selected=selected,columns=columns,script_sha256=digest(__file__),
        functions=functions[2],structural_sha256=digest(ROOT/'tools/v3_structural_features.py'),
        counts_code_sha256=digest(ROOT/'tools/v3_neural_name_counts.py'),
        optimized_sha256=digest(ROOT/'tools/v3_structural_fast.py'),
        feature_parity_sha256=digest(counts_root/'fast_parity.json') if extra else None,
        assembly_sha256=digest(ROOT/'tools/v3_neural_test_pipeline.py'),
        owner_code_sha256=digest(ROOT/'tools/v3_owner_assignment.py'),
        owner_policy_sha256=digest(STRUCT/'owner_eval/policy.json'),text_sha256=digest(STRUCT/'text_scorer.json'))
    if (output/'identity.json').exists():
        if read(output/'identity.json')!=identity:raise ValueError('Rescore identity changed')
    else:write(output/'identity.json',identity)
    reproduction=output/'reproduction';reproduction.mkdir(exist_ok=True)
    files=[audit/'report.json',Path(__file__),ROOT/'tools/v3_neural_name_counts.py',ROOT/'tools/v3_structural_features.py',ROOT/'tools/v3_structural_fast.py']
    if model is not None:files.append(model_path)
    for path in files:
        dest=reproduction/path.name
        if not dest.exists():shutil.copy2(path,dest)
        if digest(path)!=digest(dest):raise ValueError('Retained CPU recipe changed')
    def progress(stage,**kw):
        write(output/'progress.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),
            ready_for_submission=False,uploaded_to_portal=False,**kw));print(stage,kw,flush=True)
    def alive():
        if time.monotonic()>=end:raise TimeoutError('Finite CPU-rescore window expired')
        resources()
    if check_only:
        progress('preflight_passed',uses_gpu=False,structural=extra)
        if counts:counts.close()
        return
    queue=read(CONTROLLER/'queue.json')
    if set(queue['complete'])!=set(map(str,range(128))) or queue['assigned']:
        raise ValueError('Incomplete original candidate coverage')
    db=open_claims(output/'cache');scores_dir=output/'scores';scores_dir.mkdir(exist_ok=True)
    for part in range(128):
        alive();source_rows=SCRATCH/'test'/f'part_{part:03d}.pkl'
        bound=read(source/'scores'/f'{part:03d}.json');cep=source/'scores'/f'{part:03d}.ce.npy'
        if (not bound['source_pair_route_verified'] or bound['partition']!=part
                or digest(source_rows)!=bound['input_sha256'] or digest(cep)!=bound['ce_sha256']):
            raise ValueError('Retained source route or neural scores changed')
        recipe=json.dumps(dict(part=bound,identity_sha256=digest(output/'identity.json')),sort_keys=True)
        prior=db.execute('SELECT identity FROM partitions WHERE partition=?',(part,)).fetchone()
        if prior:
            if prior[0]!=recipe:raise ValueError('Completed partition identity changed')
            continue
        expected,owners=partition_inputs(part,queue['complete'][str(part)])
        rows=pickle.load(source_rows.open('rb'));verify_route(rows,expected,owners);del expected
        ce=np.load(cep,allow_pickle=False);base=np.asarray([r[4] for r in rows])
        if not valid_probabilities(ce,len(rows)):raise ValueError('Invalid retained neural probabilities')
        start=time.monotonic()
        if model is None:
            w=float(selected['name'].split('_')[1]);scores=1/(1+np.exp(-((1-w)*lg(base)+w*lg(ce))))
        else:
            x=np.column_stack((edit_matrix(rows,functions),lg(base),lg(ce))).astype(np.float32)
            if extra:x=np.column_stack((x,matrix(rows,counts,fragments)))
            scores=model.predict_proba(x)[:,1]
            del x
        if not valid_probabilities(scores,len(rows)):raise ValueError('Invalid rescored probabilities')
        array=scores_dir/f'{part:03d}.npy';np.save(array,scores)
        write(scores_dir/f'{part:03d}.json',dict(recipe=recipe,score_sha256=digest(array),pairs=len(rows)))
        with db:
            db.executemany('INSERT INTO owners VALUES(?,?)',owners.items())
            db.executemany('INSERT INTO claims VALUES(?,?,?)',[(rows[i][1],rows[i][0],float(scores[i])) for i in np.flatnonzero(scores>=selected['threshold'])])
            db.execute('INSERT INTO partitions VALUES(?,?,?)',(part,recipe,len(owners)))
        progress('cpu_rescoring',completed_partitions=part+1,partitions=128,pairs=len(rows),seconds=time.monotonic()-start)
    if counts:counts.close()
    progress('owner_assignment_and_assembly')
    applied=assemble(db,output,selected['threshold'],read(STRUCT/'owner_eval/policy.json')['selected'],load_text(STRUCT/'text_scorer.json'))
    db.close();write(output/'application.json',applied)
    for stage,args in [
        ('stream_validation',['tools/validate_full_submission.py','--test-dir',DATA,'--output',output,'--report',output/'stream_validation.json']),
        ('official_validation',['tools/package_v2_200k_submission.py','official-validate','--validator',DATA.parent.parent/'utils/validate_submission.py',
            '--test-dir',DATA,'--output',output,'--report',output/'official_matching_validation.json'])]:
        alive();progress(stage)
        with (output/(stage+'.log')).open('a') as handle:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
    stream,official=read(output/'stream_validation.json'),read(output/'official_matching_validation.json')
    if not (stream['valid'] and stream['references']==1732544 and official['valid'] and official['check_ids'] and not official['errors'] and not official['warnings']):
        raise ValueError('CPU-rescored submission validation failed')
    if not all(r['sha256']['matching_results.tsv']==digest(output/'matching_results.tsv') for r in (stream,official)):
        raise ValueError('Validated TSV changed')
    write(output/'completion.json',dict(ready_for_submission=True,uploaded_to_portal=False,references=1732544,
        matching=str(output/'matching_results.tsv'),output_sha256=stream['sha256'],development=metrics,
        identity_sha256=digest(output/'identity.json'),note='CPU-only rescore of verified neural probabilities; no public/French score claim'))
    progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--counts-root',type=Path);p.add_argument('--minutes',type=float,default=360)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.minutes<=0:p.error('Finite positive duration required')
    try:run(**vars(a))
    except Exception as error:
        write(a.output/'failure.json',dict(error=str(error),utc=datetime.now(timezone.utc).isoformat(),ready_for_submission=False));raise

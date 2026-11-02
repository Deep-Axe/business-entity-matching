"""Share storage only if assembled 200k candidates exactly equal validated 100k.

Every path and byte is preserved. Predictions, models and manifests are untouched.
The 200k assembler must finish first; no partially written file is ever replaced.
"""
import fcntl
import json
import os
from pathlib import Path
import time

from v3_transfer_data import ROOT,digest,write


def share(master,target,expected):
    master,target=Path(master),Path(target)
    if master.is_symlink() or target.is_symlink():raise ValueError('Regular immutable candidate files required')
    a,b=master.stat(),target.stat()
    if a.st_dev!=b.st_dev:raise ValueError('Different filesystems')
    if digest(master)!=expected or digest(target)!=expected:raise ValueError('Candidate content differs')
    if a.st_ino==b.st_ino:return dict(already_shared=True,size=b.st_size)
    temporary=target.with_name(target.name+'.shared-pending')
    if temporary.exists():raise FileExistsError(temporary)
    if target.stat().st_ino!=b.st_ino or target.stat().st_mtime_ns!=b.st_mtime_ns:
        raise ValueError('Candidate file changed during verification')
    os.link(master,temporary);os.replace(temporary,target)
    if digest(target)!=expected:raise ValueError('Shared candidate hash differs')
    return dict(already_shared=False,size=b.st_size,old_link_count=b.st_nlink,sha256=expected)


def run():
    work=ROOT/'artifacts/improvement_v3/shared_200k_candidates_v1';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (work/'report.json').exists():return
    old=ROOT/'artifacts/duf-fel_full_output_v2_100k'
    new=ROOT/'artifacts/duf-fel_full_output_v2_200k'
    receipt=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/stream_validation.json'
    end=time.monotonic()+3*3600
    while not (new/'assembly.json').exists():
        if time.monotonic()>=end:raise TimeoutError('Finite candidate-sharing wait expired')
        time.sleep(10)
    assembly=json.loads((new/'assembly.json').read_text());validated=json.loads(receipt.read_text())
    if not assembly['complete_submission'] or assembly['references']!=1732544 or not validated['valid']:
        raise ValueError('Complete assembly and existing valid candidate receipt required')
    expected=assembly['output_sha256']['candidate_pairs.tsv']
    if expected!=validated['sha256']['candidate_pairs.tsv']:
        write(work/'report.json',dict(complete=True,shared=False,reason='Candidate bytes differ; both files preserved'))
        return
    result=share(old/'candidate_pairs.tsv',new/'candidate_pairs.tsv',expected)
    write(work/'report.json',dict(complete=True,shared=True,result=result,assembly_sha256=digest(new/'assembly.json'),
        candidate_paths_removed=0,predictions_modified=False,
        note='Completed immutable candidate files have identical bytes and share storage. Neither path may be overwritten in place.'))


if __name__=='__main__':run()

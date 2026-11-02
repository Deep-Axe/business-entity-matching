"""Deduplicate explicit immutable checkpoint snapshots, retaining every path.

Never touch a training checkpoint or peer-owned file. Replace a duplicate only
after full-byte hash equality against a retained reference; content and all
artifact receipt hashes stay unchanged. Future consumers must treat retained
reproduction weights as immutable (as the deployment receipts already require).
"""
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil

from v3_transfer_data import ROOT,digest,write

GROUPS=(
    ('artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/reproduction/model.pt',(
        'artifacts/duf-fel_full_output_v4_neural1300k_ops_v1/reproduction/model.pt',
        'artifacts/improvement_v3/neural_followup1300k_v1/model.pt')),
    ('artifacts/duf-fel_full_output_v4_neural400k_ops_v2/reproduction/model.pt',(
        'artifacts/duf-fel_full_output_v4_neural400k_ops_v1/reproduction/model.pt',)),
)


def run():
    report=ROOT/'artifacts/improvement_v3/retained_weight_dedup_v1/report.json'
    if report.exists():raise FileExistsError('Already recorded; no blind repeat')
    before=shutil.disk_usage(ROOT).free;actions=[]
    for master_name,duplicates in GROUPS:
        master=ROOT/master_name
        if master.is_symlink() or not master.is_file():raise ValueError('Expected explicit retained regular checkpoint')
        h=digest(master);ms=master.stat()
        for name in duplicates:
            p=ROOT/name
            if p.is_symlink() or not p.is_file():raise ValueError('Expected explicit retained regular duplicate')
            s=p.stat()
            if digest(p)!=h:
                actions.append(dict(path=name,action='preserved_nonidentical',sha256=digest(p)));continue
            if s.st_ino==ms.st_ino:
                actions.append(dict(path=name,action='already_shared',sha256=h));continue
            if s.st_dev!=ms.st_dev or (s.st_mode&0o777)!=(ms.st_mode&0o777):
                raise ValueError('Checkpoint filesystem or permissions differ')
            temp=p.with_name(p.name+'.dedup-pending')
            if temp.exists():raise FileExistsError(temp)
            # Verify the old inode did not change between hashing and replacement.
            if p.stat().st_ino!=s.st_ino or p.stat().st_mtime_ns!=s.st_mtime_ns:
                raise ValueError('Retained checkpoint changed during verification')
            os.link(master,temp)
            if digest(temp)!=h or digest(master)!=h:raise ValueError('Master changed during deduplication')
            os.replace(temp,p)
            if digest(p)!=h:raise ValueError('Deduplicated checkpoint content differs')
            actions.append(dict(path=name,master=master_name,action='shared_identical_retained_inode',
                sha256=h,size=s.st_size,previous_link_count=s.st_nlink))
            write(report.with_name('progress.json'),dict(actions=actions))
    write(report,dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),actions=actions,
        free_bytes_before=before,free_bytes_after=shutil.disk_usage(ROOT).free,
        artifact_paths_removed=0,checkpoint_contents_removed=0,
        note='Every path and byte hash retained. Immutable deployment snapshots now share storage; no training or peer-owned file modified.'))


if __name__=='__main__':run()

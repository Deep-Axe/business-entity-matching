"""Remove four explicitly reviewed regenerable caches, never models or data.

User previously authorized removal of unused earlier experiment artifacts.
Source matrices (some losslessly archived), labels, borders and manifests stay.
Reject any unexpected checksum, link, open file, or missing source artifact.
"""
from pathlib import Path
import subprocess

from v3_transfer_data import ROOT, digest, write
from v3_neural_test_pipeline import read

FOLDERS=('quantized_50k_v1','quantized_gate0_20k_v1',
         'quantized_gate0_50k_v1','quantized_gate0_100k_v1')


def run():
    out=ROOT/'artifacts/improvement_v3/pruned_old_quantized_v1'
    out.mkdir(parents=True,exist_ok=False)
    reviewed=[]
    for name in FOLDERS:
        folder=ROOT/'artifacts/improvement_v2/cpu'/name/'enhanced'
        path=folder/'train.cbq';m=read(folder/'manifest.json');prepared=Path(m['prepared'])
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink!=1:
            raise ValueError('Expected one-link regular quantized cache')
        if not m['complete'] or digest(path)!=m['quantized_sha256']:
            raise ValueError('Cache differs from completed-training manifest')
        if digest(prepared/'manifest.json')!=m['prepared_manifest_sha256']:
            raise ValueError('Preserved preparation manifest differs')
        if not ((prepared/'X.npy').exists() or (prepared/'X.npy.gz').exists()):
            raise ValueError('No original or reversible source matrix')
        if digest(prepared/'y.npy')!=m['labels_sha256'] or digest(folder/'train_borders.tsv')!=m['borders_sha256']:
            raise ValueError('Preserved labels or quantization borders changed')
        checked=subprocess.run(['fuser',str(path)],capture_output=True,text=True,timeout=20)
        if checked.returncode!=1 or checked.stdout.strip() or checked.stderr.strip():
            raise ValueError('Cache may be in use or open-file check failed')
        stat=path.stat()
        reviewed.append(dict(path=str(path),sha256=m['quantized_sha256'],bytes=stat.st_size,
            inode=stat.st_ino,mtime_ns=stat.st_mtime_ns,source=str(prepared),
            manifest_sha256=digest(folder/'manifest.json')))
    write(out/'review.json',dict(files=reviewed,code_sha256=digest(__file__),
        source_data_preserved=True,models_preserved=True,recovery='Regenerate from retained matrix/labels and quantization borders'))
    for item in reviewed:
        path=Path(item['path']);stat=path.stat()
        if stat.st_ino!=item['inode'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Cache changed after review')
        checked=subprocess.run(['fuser',str(path)],capture_output=True,text=True,timeout=20)
        if checked.returncode!=1 or checked.stdout.strip() or checked.stderr.strip():raise ValueError('Cache opened after review')
        path.unlink()
        item['removed']=True
        write(out/'progress.json',dict(files=reviewed))
    write(out/'completion.json',dict(complete=True,files=reviewed,bytes_recovered=sum(i['bytes'] for i in reviewed),
        review_sha256=digest(out/'review.json'),models_removed=0,original_data_removed=0,
        note='Binary training caches deleted, not archived. Regenerable from retained original/archived matrices, labels and borders.'))


if __name__=='__main__':run()

"""Independently replay a peer's provided-gallery target-name count feature.

The feature source reads names/countries, not labels. Recount the complete train
gallery, require exact agreement with every existing peer feature, then run our
owner-held-out stacker audit. No GPU, test rescore, peer writes or promotion.
ASCII-only normalization collapses native-only names; document that limitation.
"""
from collections import Counter
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from ber.store import connect
from v3_neural_test_pipeline import SCRATCH, read, resources
from v3_transfer_data import ROOT, digest, write


def run():
    work=ROOT/'artifacts/improvement_v3/target_frequency_peerhalf_v1'
    work.mkdir(parents=True,exist_ok=False); resources()
    source=Path('src/neural/name_unique_lib.py')
    spec=importlib.util.spec_from_file_location('reviewed_peer_ascii_core',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    core=module.core
    raw={s:SCRATCH/f'{s}.pkl' for s in ('tune','dev','rivals')}
    raw['train5k']=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1/train5k.pkl'
    rows={s:pickle.load(p.open('rb')) for s,p in raw.items()}
    keys={s:[(r[3],core(r[7])) for r in rr] for s,rr in rows.items()}
    interested=set(k for kk in keys.values() for k in kk); counts=Counter()
    db=connect(ROOT/'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-8192')
    scanned=0
    for scanned,r in enumerate(db.execute('SELECT name,country FROM records'),1):
        key=(r['country'],core(r['name']))
        if key in interested:counts[key]+=1
        if scanned%500000==0:
            resources(); write(work/'progress.json',dict(stage='recount_train_gallery',records=scanned))
    db.close(); bound={}; native={}
    for split,path in raw.items():
        values=np.asarray([[np.log1p(counts[k])] for k in keys[split]],np.float32)
        if any(counts[k]<1 for k in keys[split]):raise ValueError('Target absent from its own gallery count')
        if split!='train5k':
            peer=Path(str(path)+'.nf2.npy'); old=np.load(peer,allow_pickle=False)
            if not np.array_equal(old,values):
                raise ValueError('Independently recounted target frequency differs from peer: '+split)
        dest=work/f'{split}.npy'; np.save(dest,values)
        bound[split]=dict(path=str(path),sha256=digest(path),pairs=len(values),features_sha256=digest(dest))
        native[split]=sum(not k[1] for k in keys[split])
    write(work/'manifest.json',dict(complete=True,names=['log_gallery_target_ascii_core_frequency'],inputs=bound,
        source_code_sha256=digest(source),builder_sha256=digest(__file__),records_scanned=scanned,
        store_config_sha256=digest(ROOT/'artifacts/train_store_v2/config.json'),empty_ascii_core_rows=native,
        independent_peer_feature_replay=True,labels_used_to_build_features=False,
        caveat='ASCII-only core drops native scripts. Native-only names share an empty key; this is not genuine semantic name uniqueness.'))
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    if not read(peer/'half_calibration/parity.json')['complete']:raise ValueError('Matched neural scores unavailable')
    write(work/'progress.json',dict(stage='cpu_independent_audit'))
    with (work/'audit.log').open('a') as log:
        subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
            '--ce-root',str(peer/'half_calibration'),'--rival-scores',str(peer/'half_calibration/rivals.npy'),
            '--extra-root',str(work),'--output',str(work/'evaluation')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            timeout=600,check=True)
    r=read(work/'evaluation/report.json')
    write(work/'progress.json',dict(stage='complete',selected=r['selected'],development=r['models'][r['selected']['name']]['development'],
        ready_for_submission=False,full_test_scored=False))


if __name__=='__main__':run()

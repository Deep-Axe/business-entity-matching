"""Fit the four predeclared fresh-owner heads without evaluating calibration.

Only fit_existing/fit_new labels enter fitting. Save aligned candidate scores
for later full-rival calibration; do not inspect cal_new labels or choose a
model/threshold here. No test scoring, deployment or peer-file mutation.
"""
import fcntl
import pickle
import time

import joblib
import numpy as np

from v3_neural_independent_audit import lg
from v3_neural_test_pipeline import SCRATCH, read, resources
from v3_public0976_error_audit import model, features
from v3_transfer_data import ROOT, digest, write

WORK=ROOT/'artifacts/improvement_v3/fresh_scale_fit_v1'
DATA=ROOT/'artifacts/improvement_v3/fresh_scale_features_v1'
AUDIT=ROOT/'artifacts/improvement_v3/public0976_error_audit_v2'
ROLES=ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1'
VARIANTS=('parent33','dual34','structural59','structural_dual60')


def variant_matrix(base, extra, later_logit, name):
    if base.ndim!=2 or base.shape[1]!=33 or extra.shape!=(len(base),26) or later_logit.shape!=(len(base),):
        raise ValueError('Fresh feature dimensions differ')
    if name not in VARIANTS:raise ValueError('Undeclared recipe')
    cols=[base]
    if name in ('structural59','structural_dual60'):cols.append(extra)
    if name in ('dual34','structural_dual60'):cols.append(later_logit[:,None])
    matrix=np.column_stack(cols).astype(np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Invalid feature value')
    return matrix


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+6*3600
    def progress(stage,**kw):write(WORK/'progress.json',dict(stage=stage,calibration_metrics_inspected=False,
        model_selected=False,ready_for_submission=False,**kw))
    while not (DATA/'completion.json').exists():
        if time.monotonic()>=end:raise TimeoutError('Finite fresh model waiting window expired')
        progress('waiting_for_complete_fresh_features');time.sleep(30)
    resources();done=read(DATA/'completion.json');roles=read(ROLES/'manifest.json')
    if (not done['complete'] or done['fit_owners']!=25000 or done['calibration_owners']!=5000
            or done['role_manifest_sha256']!=digest(ROLES/'manifest.json')
            or roles['owner_roles_sha256']!=digest(ROLES/'owner_roles.tsv')):
        raise ValueError('Fresh fit/calibration contract changed')
    for name,h in done['outputs_sha256'].items():
        if digest(DATA/name)!=h:raise ValueError('Fresh feature output changed')
    identity=dict(code_sha256=digest(__file__),feature_completion_sha256=digest(DATA/'completion.json'),
        role_manifest_sha256=digest(ROLES/'manifest.json'),contract_sha256=digest(ROOT/'FRESH_CALIBRATION_ROLES.md'),
        parent_audit_model_sha256=digest(AUDIT/'reconstructed_final.joblib'),
        recipe_code_sha256=digest(ROOT/'tools/v3_public0976_error_audit.py'),
        variants=list(VARIANTS),threshold_grid=[.60,.70,.75,.80,.85,.90],
        owner_policy=dict(alpha=.25,floor=0,margin=.1),threshold_or_model_selection_performed=False)
    if (WORK/'identity.json').exists():
        if read(WORK/'identity.json')!=identity:raise ValueError('Frozen fitting identity changed')
    else:write(WORK/'identity.json',identity)
    if (WORK/'completion.json').exists():return
    rows=pickle.load((DATA/'rows.pkl').open('rb'))
    with np.load(DATA/'roles.npz',allow_pickle=False) as z:
        fit=z['pair_fit'];cal=z['pair_cal'];owner_fit=z['owner_fit'];owner_cal=z['owner_cal'];owners=z['reference_ids']
    if fit.dtype!=bool or cal.dtype!=bool or not np.all(fit^cal) or len(fit)!=len(rows):raise ValueError('Invalid pair role masks')
    if owner_fit.sum()!=25000 or owner_cal.sum()!=5000 or not np.all(owner_fit^owner_cal):raise ValueError('Invalid owner role masks')
    owner_role={str(o):bool(v) for o,v in zip(owners,owner_fit)}
    if not np.array_equal(fit,[owner_role[r[0]] for r in rows]):raise ValueError('Pair roles differ from owners')
    # Read labels only after filtering to fitting rows. Calibration labels stay unused.
    Y=np.asarray([rows[i][2] for i in np.flatnonzero(fit)],np.int8)
    if set(np.unique(Y))!={0,1}:raise ValueError('Invalid fitting labels')
    base=features(rows,np.load(DATA/'ops28.npy'),np.load(DATA/'parent_pce.npy'),np.load(DATA/'name_freq3.npy'))
    extra=np.load(DATA/'structural26.npy');later=lg(np.load(DATA/'round3_pce.npy'))
    parent=joblib.load(AUDIT/'reconstructed_final.joblib')
    # Audit the frozen comparison model against original deployed probabilities.
    test=SCRATCH/'test/part_000.pkl';test_rows=pickle.load(test.open('rb'))
    tbase=features(test_rows,np.load(str(test)+'.ops.npy'),np.load(str(test)+'.e5_cont1300k.npy'),
        np.column_stack((np.load(str(test)+'.nf.npy'),np.load(str(test)+'.nf2.npy'))))
    saved=np.load(str(test)+'.stack_e5_cont1300k_nocty_nf2.npy')
    if not np.array_equal(parent.predict_proba(tbase)[:,1].astype(np.float32),saved):raise ValueError('Public-best comparison model failed parity')
    del test_rows,tbase,saved
    np.save(WORK/'frozen_public33.npy',parent.predict_proba(base)[:,1].astype(np.float32));del parent
    receipts={}
    for name in VARIANTS:
        resources();progress('fitting_training_owners_only',variant=name)
        receipt=WORK/(name+'.json')
        if receipt.exists():
            r=read(receipt)
            if r['identity_sha256']!=digest(WORK/'identity.json'):raise ValueError('Resumed model identity differs')
            for file,h in r['outputs_sha256'].items():
                if digest(WORK/file)!=h:raise ValueError('Resumed fit output changed')
            receipts[name]=r;continue
        matrix=variant_matrix(base,extra,later,name);fitted=model().fit(matrix[fit],Y)
        joblib.dump(fitted,WORK/(name+'.joblib'));np.save(WORK/(name+'.npy'),fitted.predict_proba(matrix)[:,1].astype(np.float32))
        r=dict(complete=True,features=matrix.shape[1],fit_pairs=int(fit.sum()),
            fit_owner_pool=25000,effective_fit_owners=len({rows[i][0] for i in np.flatnonzero(fit)}),
            calibration_labels_used=False,identity_sha256=digest(WORK/'identity.json'),
            outputs_sha256={file:digest(WORK/file) for file in (name+'.joblib',name+'.npy')})
        write(receipt,r);receipts[name]=r;del matrix,fitted
    if digest(DATA/'completion.json')!=identity['feature_completion_sha256']:raise ValueError('Feature receipt changed during fitting')
    write(WORK/'completion.json',dict(complete=True,models=receipts,
        frozen_public_scores_sha256=digest(WORK/'frozen_public33.npy'),identity_sha256=digest(WORK/'identity.json'),
        calibration_metrics_inspected=False,model_selected=False,ready_for_submission=False,
        next_required='Build genuine full-context reverse rivals, then evaluate fixed candidate models on cal_new only; reserved audit stays untouched.'))
    progress('complete_models_only')


if __name__=='__main__':run()

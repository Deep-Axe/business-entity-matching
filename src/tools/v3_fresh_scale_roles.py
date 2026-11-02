"""Freeze roles for a possible fresh-data scale experiment; no jobs launched.

Keep the original 5k fitting owners, add 20k, reserve the remaining 5k from the
selected train-fold pool for independent stacker calibration. These labels
must not be passed into fitting. The eval-fold reporting reservation is separate.
"""
from collections import Counter
import json

from ber.data import read_tsv, write_tsv
from v3_transfer_data import ROOT, digest, write


def roles(n):
    if n != 30000:
        raise ValueError('Expected the frozen 30k-owner pool')
    return ['fit_existing']*5000 + ['fit_new']*20000 + ['cal_new']*5000


def run():
    source=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
    cloud=ROOT/'artifacts/improvement_v3/fresh_cloud25k_v1'
    output=ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1'; output.mkdir(parents=True,exist_ok=True)
    selection=json.loads((source/'selection.json').read_text()); plan=json.loads((cloud/'plan.json').read_text())
    for name,expected in selection['output_sha256'].items():
        if digest(source/name)!=expected:raise ValueError('Original owner selection changed')
    if digest(cloud/'references.tsv')!=plan['references_sha256']:
        raise ValueError('Cloud extraction references changed')
    rows=list(read_tsv(source/'references_30k.tsv')); old=list(read_tsv(source/'references_5k.tsv'))
    if rows[:5000]!=old or rows[5000:]!=list(read_tsv(cloud/'references.tsv')):
        raise ValueError('Local/cloud selections are not the exact original nesting')
    ids=[r['entity_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Repeated owner')
    assignments=[dict(entity_id=o,role=role) for o,role in zip(ids,roles(len(rows)))]
    path=output/'owner_roles.tsv'; manifest=output/'manifest.json'
    if manifest.exists():
        old_manifest=json.loads(manifest.read_text())
        if digest(path)!=old_manifest['owner_roles_sha256'] or list(read_tsv(path))!=assignments:
            raise ValueError('Frozen role assignments changed')
        return
    write_tsv(path,assignments,['entity_id','role'])
    write(manifest,dict(complete=True,jobs_launched=False,labels_used_for_role_selection=False,
        owners=len(rows),roles=dict(Counter(r['role'] for r in assignments)),fit_owners=25000,calibration_owners=5000,
        country_counts={role:dict(Counter(row['country'] for row,r in zip(rows,assignments) if r['role']==role))
                        for role in ('fit_existing','fit_new','cal_new')},
        owner_roles_sha256=digest(path),selection_sha256=digest(source/'selection.json'),cloud_plan_sha256=digest(cloud/'plan.json'),
        code_sha256=digest(__file__),
        fit_allowed_roles=['fit_existing','fit_new'],calibration_allowed_roles=['cal_new'],
        requirements=['Wait for complete 200k test queue, idle nodes, budget and resource checks before extraction.',
            'Filter compact exports by these roles; their training_only metadata denotes train-fold provenance, not permission to fit calibration rows.',
            'Freeze retrieval/model candidates before looking at the new calibration labels; never fit on cal_new.',
            'The separate eval-fold reserved 2k stays reporting-only after final recipe freeze.'],
        caveat='All owners are outside the first-stage/CE 200k left-owner fit, but provided gallery/alias resources may have seen their text or labels. Not a pristine end-to-end holdout.'))
    print('Fixed 25k fitting / 5k calibration roles. No cloud job or model run.')


if __name__=='__main__':run()

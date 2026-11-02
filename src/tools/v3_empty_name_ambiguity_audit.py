"""Check literal raw-text ambiguity within the existing training-only pool.

Owner ID fixes the left record; target source and raw name are retained, and
every target address in this pool is genuinely missing. This is not a test
accuracy estimate or permission to use IDs as model features.
"""
from collections import Counter, defaultdict
import pickle

from v3_neural_test_pipeline import read, resources
from v3_transfer_data import ROOT, digest, write


def groups_report(rows, key):
    groups = defaultdict(Counter)
    for row in rows:
        groups[key(row)][int(row[2])] += 1
    mixed = [v for v in groups.values() if v[0] and v[1]]
    return dict(groups=len(groups), mixed_groups=len(mixed),
                positive_rows_in_mixed_groups=sum(v[1] for v in mixed),
                negative_rows_in_mixed_groups=sum(v[0] for v in mixed),
                minimum_pair_errors_if_same_decision_within_group=sum(min(v[0],v[1]) for v in mixed))


def run():
    source=ROOT/'artifacts/improvement_v3/empty_name_specialist_data_v1'
    out=ROOT/'artifacts/improvement_v3/empty_name_ambiguity_v1';out.mkdir(parents=True,exist_ok=False)
    resources();receipt=read(source/'completion.json')
    if not receipt['training_only'] or digest(source/'pairs.pkl')!=receipt['pairs_sha256']:
        raise ValueError('Training-only name pool changed')
    rows=pickle.load((source/'pairs.pkl').open('rb'))
    results={
        'same_name_only_input':groups_report(rows,lambda r:(r[4],r[5])),
        'same_owner_raw_target_name':groups_report(rows,lambda r:(r[0],r[5])),
        'same_owner_source_raw_target_name':groups_report(rows,lambda r:(r[0],r[1].split('-')[0],r[5])),
    }
    if results['same_name_only_input']['mixed_groups']!=receipt['exact_input_groups_with_mixed_labels']:
        raise ValueError('Original specialist ambiguity census does not replay')
    write(out/'report.json',dict(complete=True,training_only=True,rows=len(rows),groups=results,
        input_sha256=receipt['pairs_sha256'],code_sha256=digest(__file__),ready_for_submission=False,
        limitations=['Selected old CE training pool, not all provided records and not test.',
            'Source/owner group is literal raw-name ambiguity with missing address; no ID used as a feature.',
            'Minimum pair errors assumes identical grouped inputs receive one decision; not a macro-F0.5 or graph oracle.',
            'Global context or distinct missing-value serialization may distinguish some groups. No public ceiling claimed.']))


if __name__=='__main__':run()

"""Freeze the parent-approved simpler CPU policy before audit scores are read."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ber.shards import file_hash


def freeze(experimental_release, output):
    experimental_release, output = Path(experimental_release), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(experimental_release.read_text())
    for name, artifact in source['artifacts'].items():
        if file_hash(artifact['path']) != artifact['sha256']:
            raise ValueError('Changed original release artifact: ' + name)
    comparison = json.loads(Path(source['artifacts']['comparison_report']['path']).read_text())
    control = comparison['threshold_only_selected_on_tune']
    if control['allowance'] is not None or control['threshold'] != .76:
        raise ValueError('Expected tuning-selected threshold-only 0.76 policy')
    if comparison['fixed_threshold_veto_selected_on_tune']['allowance'] is not None:
        raise ValueError('Fixed-threshold tuning no longer prefers no veto')
    source.update(release_candidate='gate0_enhanced100k_cpu_v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), threshold=.76,
        threshold_selection='threshold-only full tuning maximum; parent simplicity/runtime decision before audit scores',
        status='production CPU-only policy frozen; fresh CPU-policy audit and full-test validation pending',
        owner_policy=dict(enabled=False, allowance=None,
                          multiple_targets_per_reference_allowed=True, unmatched_targets_allowed=True),
        production_selection=control,
        decision=dict(authority='parent explicit instruction before any audit scores were inspected',
            rationale='Owner tuning gain +0.000037 is negligible and uncertain, adds full-test reverse-index runtime; fixed 0.76 tuning prefers no veto',
            based_on='existing tune/development comparison and operational runtime, not fresh audit',
            experimental_owner_release=dict(path=str(experimental_release.resolve()), sha256=file_hash(experimental_release)),
            superseded_audit='inference/audit_100k_v1 at 0.70 stopped; partial artifacts retained; no score inspected',
            fresh_audit='inference/audit_100k_cpu_v1 at frozen 0.76; no subsequent policy tuning'),
        final_audit_used_for_selection=False)
    source['artifacts']['cpu_release_code'] = dict(path=str(Path(__file__).resolve()), sha256=file_hash(__file__))
    source['caveats'] = ['No hidden-test or French accuracy claim',
                         'Historical owner comparison is retained as evidence, not the production policy',
                         'Full-test coverage and submission validation remain required']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(source, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(dict(path=str(output.resolve()), sha256=file_hash(output), threshold=.76, owner_enabled=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimental-release', required=True)
    parser.add_argument('--output', required=True)
    freeze(**vars(parser.parse_args()))

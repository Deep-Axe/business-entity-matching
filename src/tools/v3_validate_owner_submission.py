"""Validate a deletion-only owner variant, without loading the full graph in RAM.

Requires separate successful full-file and organizer validation receipts. Checks
every baseline edge against the score cache and every output edge against the
decision table. Exhaustive conflict coverage then proves target exclusivity.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sqlite3
import time

from ber.data import read_tsv
from v3_transfer_data import ROOT, digest, write


def id_set(row):
    if set(row) != {'source1_entity_id', 'matched_entity_ids'}:
        raise ValueError('Incorrect matching columns')
    ids = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate match in row')
    return set(ids)


def check_rows(db, baseline_rows, output_rows, references, expected_references):
    conflicts = db.execute('SELECT count(*) FROM (SELECT target FROM claims '
                           'GROUP BY target HAVING count(*)>1)').fetchone()[0]
    decisions, abstained = db.execute('SELECT count(*),sum(owner IS NULL) FROM decisions.keep').fetchone()
    if decisions != conflicts:
        raise ValueError('Decision table must cover every conflicting target')
    invalid = db.execute('SELECT count(*) FROM decisions.keep k WHERE '
                         '(SELECT count(*) FROM claims c WHERE c.target=k.target)<2').fetchone()[0]
    if invalid:
        raise ValueError('Decision applied to a non-conflicting target')
    invalid = db.execute('SELECT count(*) FROM decisions.keep k LEFT JOIN claims c '
                         'ON c.target=k.target AND c.owner=k.owner '
                         'WHERE k.owner IS NOT NULL AND c.owner IS NULL').fetchone()[0]
    if invalid:
        raise ValueError('Winning owner was not an accepted claimant')
    totals, countries = Counter(), defaultdict(Counter)
    sql = ('SELECT c.target,k.target,k.owner FROM claims c LEFT JOIN decisions.keep k '
           'ON k.target=c.target WHERE c.owner=?')
    for reference, old, new in itertools.zip_longest(references, baseline_rows, output_rows):
        if reference is None or old is None or new is None:
            raise ValueError('Reference/output lengths differ')
        owner = reference['entity_id']
        if old['source1_entity_id'] != owner or new['source1_entity_id'] != owner:
            raise ValueError('Reference identity/order changed')
        original, actual = id_set(old), id_set(new)
        expected, cached = set(), set()
        for target, decision_target, winner in db.execute(sql, (owner,)):
            cached.add(target)
            if decision_target is None or winner == owner:
                expected.add(target)
        if original != cached:
            raise ValueError('Accepted score cache differs from baseline TSV: ' + owner)
        if not actual <= original:
            raise ValueError('Owner assignment added a match: ' + owner)
        if actual != expected:
            raise ValueError('Output does not exactly implement conflict decisions: ' + owner)
        count = dict(references=1, baseline_pairs=len(original), predicted_pairs=len(actual),
                     removed_pairs=len(original-actual), changed_owners=int(original != actual),
                     baseline_empty=int(not original), output_empty=int(not actual),
                     newly_empty=int(bool(original) and not actual))
        totals.update(count)
        countries[reference['country']].update(count)
        if totals['references'] % 100000 == 0:
            print(json.dumps({'stage': 'structural_validation', **totals}), flush=True)
    if totals['references'] != expected_references:
        raise ValueError('Incomplete reference coverage')
    if totals['baseline_pairs'] != db.execute('SELECT count(*) FROM claims').fetchone()[0]:
        raise ValueError('Accepted graph contains extra or missing baseline edges')
    return dict(totals=totals, countries=dict(countries), conflicting_targets_before=conflicts,
                abstained_conflicting_targets=abstained or 0, conflicting_targets_after=0)


def run(cache, baseline_output, output, policy, text_scorer):
    cache, baseline_output, output, policy, text_scorer = map(
        Path, (cache, baseline_output, output, policy, text_scorer))
    receipt_path = output/'completion.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    started = time.monotonic()
    load = lambda path: json.loads(path.read_text())
    result = load(output/'report.json')
    stream = load(output/'stream_validation.json')
    official = load(output/'official_matching_validation.json')
    rule = load(policy)
    baseline = load(baseline_output/'assembly.json')
    if not result.get('complete') or result['references'] != 1732544:
        raise ValueError('Owner application incomplete')
    if (not rule.get('enabled') or not rule.get('frozen_on_tune') or
            not rule.get('development_reported') or
            result['policy_sha256'] != digest(policy) or
            rule['text_scorer_sha256'] != digest(text_scorer)):
        raise ValueError('Frozen policy/scorer binding mismatch')
    if not baseline.get('complete_submission') or baseline['references'] != 1732544:
        raise ValueError('Baseline assembly incomplete')
    if not stream.get('valid') or stream['references'] != 1732544:
        raise ValueError('Full output validation incomplete')
    if (not official.get('valid') or official.get('errors') or official.get('warnings') or
            not official.get('check_ids') or official.get('scope') != 'complete matching file'):
        raise ValueError('Organizer validation failed')
    test = ROOT/'6ab10eb3b23ba_student_resource/student_resource/dataset/test'
    inputs = {'matching_results.tsv': output/'matching_results.tsv',
              'validator': test.parent.parent/'utils/validate_submission.py',
              **{f'test_source{i}.tsv': test/f'test_source{i}.tsv' for i in (1, 2, 3)}}
    for name, path in inputs.items():
        if official['sha256'].get(name) != digest(path):
            raise ValueError('Organizer-validated input changed: ' + name)
    before = {}
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        before[name] = digest(output/name)
        if before[name] != stream['sha256'][name]:
            raise ValueError('Stream-validated output changed')
        if digest(baseline_output/name) != baseline['output_sha256'][name]:
            raise ValueError('Original baseline changed')
    if (before['matching_results.tsv'] != result['matching_sha256'] or
            before['candidate_pairs.tsv'] != baseline['output_sha256']['candidate_pairs.tsv']):
        raise ValueError('Output identity or unchanged candidate assertion failed')
    db = sqlite3.connect((cache/'claims.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('ATTACH DATABASE ? AS decisions',
               ((output/'decisions.sqlite3').resolve().as_uri()+'?mode=ro',))
    try:
        checked = check_rows(db, read_tsv(baseline_output/'matching_results.tsv'),
                             read_tsv(output/'matching_results.tsv'),
                             read_tsv(test/'test_source1.tsv'), 1732544)
    finally:
        db.close()
    if (checked['totals']['removed_pairs'] != result['removed_edges'] or
            checked['abstained_conflicting_targets'] != result['abstained_targets'] or
            checked['totals']['predicted_pairs'] != stream['predicted_pairs']):
        raise ValueError('Application/validation counts disagree')
    for name, hashed in before.items():
        if digest(output/name) != hashed:
            raise ValueError('Output changed during structural validation')
    receipt = dict(ready_for_submission=True, uploaded_to_portal=False,
                   utc=datetime.now(timezone.utc).isoformat(),
                   matching=str((output/'matching_results.tsv').resolve()),
                   method='Frozen CPU baseline + tune-frozen probability/text owner assignment; no reranker',
                   base_model_sha256=rule['model_sha256'],threshold=rule.get('threshold',.76),
                   output_sha256=before, policy_sha256=digest(policy),
                   text_scorer_sha256=digest(text_scorer),
                   validation_sha256={p.name: digest(p) for p in (
                       output/'stream_validation.json', output/'official_matching_validation.json',
                       output/'report.json')},
                   structural_validator_sha256=digest(Path(__file__)),
                   unchanged_candidates=True, nonconflicting_predictions_unchanged=True,
                   no_added_matches=True, **checked, seconds=time.monotonic()-started,
                   note='Structural/format validation only; this variant has no measured leaderboard score')
    write(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'baseline-output', 'output', 'policy', 'text-scorer'):
        p.add_argument('--'+name, required=True, type=Path)
    run(**vars(p.parse_args()))

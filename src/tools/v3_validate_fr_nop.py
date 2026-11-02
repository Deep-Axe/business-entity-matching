"""Independently validate the other session's France-only variant, read-only.

Check full coverage and organizer IDs, and prove US/India lines are unchanged
from the submitted 0.976 baseline. No quality claim for unlabeled France.
"""
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
from pathlib import Path
import subprocess
import sys

from v3_neural_test_pipeline import DATA, read, resources
from v3_transfer_data import ROOT, digest, write

SOURCE=Path('work/ce/sub_v5_fr_nop')
BASE=ROOT/'artifacts/public_best_0976_20260926/matching_results.tsv'
WORK=ROOT/'artifacts/improvement_v3/peer_fr_nop_validation_v1'
EXPECTED='443ddeaf96415f955268cddb68d1b011ee0e8f8bae1d6ce1cfa8e778f91534ce'


def run():
    WORK.mkdir(parents=True,exist_ok=False);resources()
    original=read(SOURCE/'report.json')
    identity=dict(code_sha256=digest(__file__),source_report_sha256=digest(SOURCE/'report.json'),
        matching_sha256=digest(SOURCE/'matching_results.tsv'),baseline_sha256=digest(BASE))
    if (identity['matching_sha256']!=EXPECTED or original['matching_sha256']!=EXPECTED
            or original['references']!=1732544 or identity['baseline_sha256']!='5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118'):
        raise ValueError('Frozen comparison identities differ')
    write(WORK/'identity.json',identity)
    def progress(stage):write(WORK/'progress.json',dict(stage=stage,accuracy_claimed=False,uploaded_to_portal=False))
    for stage,args in (
        ('stream',['tools/validate_full_submission.py','--test-dir',DATA,'--output',SOURCE,'--report',WORK/'stream.json']),
        ('official',['tools/package_v2_200k_submission.py','official-validate','--validator',DATA.parent.parent/'utils/validate_submission.py',
            '--test-dir',DATA,'--output',SOURCE,'--report',WORK/'official.json'])):
        progress(stage);resources()
        with (WORK/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    s,o=read(WORK/'stream.json'),read(WORK/'official.json')
    if not (s['valid'] and s['references']==1732544 and o['valid'] and o['check_ids'] and not o['errors'] and not o['warnings']):
        raise ValueError('Full-file validator failed')
    if any(r['sha256']['matching_results.tsv']!=EXPECTED for r in (s,o)):raise ValueError('Validated file hash differs')
    progress('checking_exact_country_isolation')
    with (DATA/'test_source1.tsv').open(newline='') as handle:
        french={r['entity_id'] for r in csv.DictReader(handle,delimiter='\t') if r['country']=='France'}
    if len(french)!=259452:raise ValueError('French reference count differs')
    counts=defaultdict(Counter);seen=0
    with BASE.open(newline='') as left,(SOURCE/'matching_results.tsv').open(newline='') as right:
        for a,b in zip_longest(csv.DictReader(left,delimiter='\t'),csv.DictReader(right,delimiter='\t')):
            if a is None or b is None or a['source1_entity_id']!=b['source1_entity_id']:raise ValueError('Comparison row alignment differs')
            seen+=1;country='France' if a['source1_entity_id'] in french else 'US/India'
            aa=set(filter(None,a['matched_entity_ids'].split(',')));bb=set(filter(None,b['matched_entity_ids'].split(',')))
            counts[country].update(owners=1,changed_owners=int(aa!=bb),added=len(bb-aa),removed=len(aa-bb),before=len(aa),after=len(bb))
            if country=='US/India' and a!=b:raise ValueError('Purported France-only variant changes another country')
    if seen!=1732544 or digest(SOURCE/'matching_results.tsv')!=EXPECTED:raise ValueError('Final coverage/hash changed')
    write(WORK/'completion.json',dict(complete=True,structural_validation_passed=True,exact_us_india_unchanged=True,
        matching=str(SOURCE/'matching_results.tsv'),matching_sha256=EXPECTED,identity_sha256=digest(WORK/'identity.json'),
        references=seen,changes=counts,stream_sha256=digest(WORK/'stream.json'),official_sha256=digest(WORK/'official.json'),
        ready_as_experimental_submission=True,accuracy_independently_measured=False,uploaded_to_portal=False,
        note='French head omits first-stage p100 feature. No French labels, no measured improvement over public 0.976. Structural validity alone is not a quality recommendation.'))
    progress('complete')


if __name__=='__main__':run()

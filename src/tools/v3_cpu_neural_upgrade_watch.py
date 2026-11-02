"""Select a follow-on CPU rescore on held-out calibration, with quality gates.

No baseline overwrite, no GPU work, no upload. Development is a reused screening
set, not a pristine confirmation set. A failed chosen-model gate means no new
deployment, not trying runners-up until one looks good on development.
"""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time

from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,digest,write


def run():
    work=ROOT/'artifacts/improvement_v3/neural_cpu_upgrade_v1';work.mkdir(parents=True,exist_ok=True)
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    fresh=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_v1/evaluation'
    extra=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_extra_v1/evaluation'
    wider=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1/evaluation'
    structural=ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1'
    own=work/'same_training_half_extra';end=time.monotonic()+12*3600
    def progress(stage,**kw):
        write(work/'progress.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),ready_for_submission=False,**kw))
    def command(args,log):
        resources()
        if time.monotonic()>=end:raise TimeoutError('Finite CPU upgrade window expired')
        with (work/log).open('a') as handle:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                check=True,timeout=end-time.monotonic())
    while not (peer/'evaluation/report.json').exists():
        if time.monotonic()>=end:raise TimeoutError('Peer precision audit unavailable')
        progress('waiting_for_peer_precision_audit');time.sleep(30)
    if not (own/'report.json').exists():
        progress('precision_matched_structural_audit')
        command(['tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
            '--ce-root',peer/'half_calibration','--rival-scores',peer/'half_calibration/rivals.npy',
            '--extra-root',structural,'--output',own],'same_training_extra.log')
    while not all(p.exists() for p in (peer/'completion.json',fresh/'report.json',extra/'report.json',wider/'report.json')):
        if time.monotonic()>=end:raise TimeoutError('Critical neural follow-ups did not finish in finite window')
        progress('waiting_for_complete_submission_and_followups');time.sleep(30)
    baseline=read(peer/'evaluation/report.json');baseline_choice=baseline['selected']
    baseline_metric=baseline['models'][baseline_choice['name']]['development']
    paths=[peer/'evaluation',own,fresh,extra];options=[]
    for path in paths:
        r=read(path/'report.json')
        if not r['complete'] or Path(r['ce_root']).resolve()!=(peer/'half_calibration').resolve():
            raise ValueError('Incompatible candidate audit')
        if r.get('widened_rival_coverage'):raise ValueError('Cannot compare calibration with different rival coverage')
        options.append(dict(audit=str(path),report_sha256=digest(path/'report.json'),selected=r['selected']))
    # Select only by the SAME held-out calibration owners/metric. Prefer the
    # already finished baseline on ties; don't maximize development over trials.
    best=max(options,key=lambda x:(x['selected']['calibration_macro_f05'],x['audit']==str(peer/'evaluation')))
    write(work/'selection.json',dict(options=options,selected=best,selection='Calibration only; baseline preferred on ties',
        development_used_for_selection=False,development_reused_for_quality_gate=True,
        wider_audit_sha256=digest(wider/'report.json')))
    chosen=read(Path(best['audit'])/'report.json');metric=chosen['models'][best['selected']['name']]['development']
    cal_gain=best['selected']['calibration_macro_f05']-baseline_choice['calibration_macro_f05']
    dev_gain=metric['macro_f05']-baseline_metric['macro_f05']
    country_delta={c:metric['countries'][c]-v for c,v in baseline_metric['countries'].items()}
    accepted=best['audit']!=str(peer/'evaluation') and cal_gain>=.0002 and dev_gain>=.0001 and min(country_delta.values())>=-.002
    write(work/'quality_gate.json',dict(accepted=accepted,calibration_gain=cal_gain,development_gain=dev_gain,country_delta=country_delta,
        note='Exploratory screen, not a significance claim. No French labels. Wider rival audit is reported separately and is not silently treated as equivalent coverage.'))
    if not accepted:
        progress('complete_no_upgrade',reason='Calibration winner did not clear the preset quality gate; original submission preserved')
        return
    # Do not merely wait for wider competition: explicitly replay both frozen
    # policies against it. No new fitting/threshold search and no runner-up
    # selection using this reused development check.
    robustness=work/'frozen_widened_comparison'
    if not (robustness/'report.json').exists():
        progress('frozen_wider_competition_gate')
        command(['tools/v3_frozen_widened_audit.py','--baseline',peer/'evaluation',
            '--candidate',best['audit'],'--widened',wider.parent,'--output',robustness],
            'frozen_widened.log')
    check=read(robustness/'report.json')
    if not check['complete'] or check['models'][1]['report_sha256']!=best['report_sha256']:
        raise ValueError('Frozen widened comparison does not match selected candidate')
    delta=check['comparison']
    passed=(delta['calibration_gain']>=0 and delta['development_gain']>=.0001
        and min(delta['country_delta'].values())>=-.002)
    write(work/'widened_quality_gate.json',dict(accepted=passed,**delta,
        report_sha256=digest(robustness/'report.json'),fitted_or_retuned=False))
    if not passed:
        progress('complete_no_upgrade',reason='Frozen winner failed wider-rival robustness gate; no fallback search')
        return
    destination=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_cpu_upgrade_v1'
    if not (destination/'completion.json').exists():
        progress('cpu_only_rescore',audit=best['audit'])
        command(['tools/v3_cpu_neural_rescore.py','--audit',best['audit'],'--source',peer,'--output',destination,
            '--counts-root','artifacts/improvement_v3/test_s1_structural_counts_v1',
            '--minutes',max(1,int((end-time.monotonic())/60))],'cpu_rescore.log')
    if not read(destination/'completion.json')['ready_for_submission']:raise ValueError('CPU upgrade did not validate')
    progress('complete',submission=str(destination/'matching_results.tsv'))


if __name__=='__main__':run()

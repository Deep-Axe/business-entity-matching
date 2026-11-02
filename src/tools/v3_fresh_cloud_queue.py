"""Reuse eligible idle workers for the frozen eight-part 25k training export.

No provisioning, timer changes, fitting, test scoring, or uploads. Start only
after exact 200k completion, live Free Plan/credit checks, and per-node idle and
resource checks. Remote feature matrices stay remote; collect compact exports.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from aws_benchmark_ctl import aws
from aws_baseline_transfer import ssh
from v2_submission_controller import ROOT, REMOTE, host_for, invoke
from v3_transfer_data import digest, write

WORK = ROOT/'artifacts/improvement_v3/fresh_cloud25k_v1'
QUEUE = ROOT/'artifacts/improvement_v2/inference/submission_controller_200k_v1'
CONFIG = ROOT/'artifacts/improvement_v2/inference/submission_queue_config_200k_v1.json'
BUDGET = ROOT/'artifacts/improvement_v2/cloud/final_deadline_budget_20260927_v1.json'
TIMER = 'amazon-ml-v3-final-deadline-20260927.timer'
MODEL200 = 'artifacts/improvement_v2/cpu/deployed200k_cpu_v1/enhanced.cbm'
# These two inference-only machines lack the full training gallery/index.
# Preserve eight data partitions; six verified training-capable machines execute
# them, with the first two machines taking a second partition only when idle.
TRAIN_NODES = ('knn_fleet_v1_node0', 'knn_fleet_v1_node1', 'fleet_v1_node0',
               'fleet_v1_node1', 'fleet_v1_node2', 'fleet_v1_node3')


def read(path):
    return json.loads(Path(path).read_text())


def part_can_start(part, completed, assignment):
    return all(str(k) in completed for k in range(part) if assignment[k] == assignment[part])


def complete_handoff(manifest, queue):
    if not manifest.get('complete') or not manifest.get('full_test_coverage'):
        return False
    parts=queue.get('complete', {})
    if queue.get('assigned') or set(parts)!=set(map(str,range(128))):
        raise ValueError('200k handoff lacks exact completed/free-worker coverage')
    if sum(p['references'] for p in parts.values())!=1732544:
        raise ValueError('200k handoff reference count differs')
    if len({p['directory'] for p in parts.values()})!=128:
        raise ValueError('Repeated completed partition directory')
    return True


def budget_check(receipt, account, now):
    deadline=datetime.fromisoformat(receipt['deadline_utc'])
    b=receipt['budget']
    if not (receipt.get('complete') and receipt.get('timers_applied') and
            receipt.get('reviewed_budget') and b['no_plan_upgrade'] and
            b['total_authorized_usd']==120 and b['conservative_combined_usd']<=120):
        raise ValueError('Invalid reviewed budget/timer receipt')
    if account['accountPlanType']!='FREE' or account['accountPlanStatus']!='ACTIVE':
        raise ValueError('Active Free Plan required; no upgrade authorized')
    # Existing whole-fleet timer reserve already covers this extraction. Requiring
    # remaining credit to cover the entire conservative historical+future reserve
    # is deliberately stricter than charging this job an additional runtime.
    if float(account['accountPlanRemainingCredits']['amount'])<b['conservative_combined_usd']:
        raise ValueError('Confirmed credits below conservative whole-run reserve')
    if (deadline-now).total_seconds()<7200:
        raise ValueError('Less than two hours before absolute fleet shutdown')
    return deadline


def remote(node, command, timeout=120):
    run,host=host_for(node)
    result=subprocess.run(ssh(run,host)+['cd '+shlex.quote(REMOTE)+' && '+command],
                          capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(node['id']+': '+result.stderr[-1400:])
    return result.stdout


def preflight(node, deadline):
    code="""import json,os,shutil,subprocess
from pathlib import Path
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().decode().split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeDecodeError):continue
 if any(v in a for v in ('tools/v2_run_pipeline.py','tools/v2_sparse_fullgallery.py')):active.append(p.name)
mem=next(int(s.split()[1])/1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
t=subprocess.check_output(['systemctl','show','amazon-ml-v3-final-deadline-20260927.timer','-p','NextElapseUSecRealtime','--value'],text=True).strip()
stamp=int(subprocess.check_output(['date','-u','-d',t,'+%s'],text=True))
state=subprocess.check_output(['systemctl','is-active','amazon-ml-v3-final-deadline-20260927.timer'],text=True).strip()
print(json.dumps(dict(active=active,memory_mib=mem,disk_gib=shutil.disk_usage('.').free/1024**3,timer_epoch=stamp,timer_state=state)))
"""
    h=json.loads(remote(node,shlex.join(['.venv/bin/python','-c',code])))
    if h['active'] or h['memory_mib']<4096 or h['disk_gib']<6:
        raise ValueError(node['id']+': busy or resource reserve insufficient')
    if h['timer_state']!='active' or h['timer_epoch']!=int(deadline.timestamp()):
        raise ValueError(node['id']+': absolute shutdown timer differs')
    return h


def ship(node, plan):
    paths=[WORK/'references.tsv',WORK/'plan.json',ROOT/'tools/v3_fresh_cloud_data.py',ROOT/'tools/v3_transfer_data.py']
    if digest(paths[0])!=plan['references_sha256'] or digest(paths[2])!=plan['exporter_sha256']:
        raise ValueError('Frozen extraction payload changed')
    run,host=host_for(node);transport=shlex.join(ssh(run,host)[:-1])
    for path in paths:
        relative=path.relative_to(ROOT)
        remote(node,'mkdir -p '+shlex.quote(str(relative.parent)))
        subprocess.run(['rsync','-az','--ignore-existing','-e',transport,str(path),
                        'ubuntu@'+host['ip']+':'+REMOTE+'/'+str(relative)],check=True,timeout=90,
                       stdout=subprocess.DEVNULL)
    identities={str(p.relative_to(ROOT)):digest(p) for p in paths}
    identities['tools/v2_run_pipeline.py']=plan['runner_sha256']
    identities['artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm']=plan['models_sha256']['100']
    identities[MODEL200]=plan['models_sha256']['200']
    for relative in ('artifacts/train_index_v2/config.json', 'artifacts/train_store_v2/config.json'):
        identities[relative]=digest(ROOT/relative)
    # No existing remote file is overwritten. A differing dependency fails here.
    code='import json,hashlib; d=json.loads('+repr(json.dumps(identities))+'); '+\
        'assert all(hashlib.sha256(open(p,"rb").read()).hexdigest()==h for p,h in d.items()); print("verified")'
    remote(node,shlex.join(['.venv/bin/python','-c',code]))
    return identities


def collect_compact(node, part, plan):
    source=plan['specs'][part]['spec']['output']
    destination=f'artifacts/improvement_v3/fresh25k_compact_p{part}_v1'
    argv=['.venv/bin/python','tools/v3_fresh_cloud_data.py','export',
          '--plan',str((WORK/'plan.json').relative_to(ROOT)),'--part',str(part),
          '--source',source,'--output',destination,'--model200',
          MODEL200]
    remote(node,'env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(argv),timeout=300)
    run,host=host_for(node);local=WORK/'collected'/str(part);local.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-az','-e',shlex.join(ssh(run,host)[:-1]),
                    'ubuntu@'+host['ip']+':'+REMOTE+'/'+destination+'/',str(local)+'/'],check=True,timeout=150,
                   stdout=subprocess.DEVNULL)
    done=read(local/'completion.json')
    if not done['complete'] or done['partition']!=part or done['plan_sha256']!=digest(WORK/'plan.json'):
        raise ValueError('Collected compact partition identity differs')
    if done['owners']!=plan['specs'][part]['expected_references']:
        raise ValueError('Collected owner count differs')
    for name,h in done['output_sha256'].items():
        if Path(name).name!=name or digest(local/name)!=h:raise ValueError('Compact output checksum mismatch')
    return dict(directory=str(local),completion_sha256=digest(local/'completion.json'),owners=done['owners'])


def run(minutes=180):
    out=WORK/'controller';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60;plan=read(WORK/'plan.json');cfg=read(CONFIG)
    roles=read(ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1/manifest.json')
    if roles['cloud_plan_sha256']!=digest(WORK/'plan.json'):raise ValueError('Frozen role plan mismatch')
    nodes=[n for n in cfg['nodes'] if not n.get('local')]
    if len(nodes)!=8 or any(n['workers']!=2 for n in nodes):raise ValueError('Expected existing eight two-worker nodes')
    workers=[next(n for n in nodes if n['id']==name) for name in TRAIN_NODES]
    assignment=[workers[part%len(workers)]['id'] for part in range(8)]
    state_path=out/'queue.json'
    state=read(state_path) if state_path.exists() else dict(plan_sha256=digest(WORK/'plan.json'),
        code_sha256=digest(__file__),nodes=[n['id'] for n in nodes],launched={},complete={})
    if state['code_sha256']!=digest(__file__):
        previous='f419d7684fcae1f29a2e396e9993461ba8090a247281887c51931184a4c50c21'
        if state['code_sha256']!=previous or state['launched'] or state['complete']:
            raise ValueError('Only the known pre-launch path failure can migrate')
        state['prelaunch_migration']=dict(previous_code_sha256=previous,
            reason='Verified existing deployment path and six training-capable nodes; data plan unchanged')
        state['code_sha256']=digest(__file__)
    if state['plan_sha256']!=digest(WORK/'plan.json') or state['nodes']!=[n['id'] for n in nodes]:
        raise ValueError('Extraction controller identity changed')
    if state.get('worker_assignment',assignment)!=assignment:raise ValueError('Worker assignment changed')
    state['worker_assignment']=assignment
    write(state_path,state)
    while not (QUEUE/'manifest.json').exists() or not complete_handoff(read(QUEUE/'manifest.json'),read(QUEUE/'queue.json')):
        if time.monotonic()>=end:raise TimeoutError('Finite handoff wait expired')
        write(out/'progress.json',dict(stage='waiting_for_exact_200k_completion',jobs_launched=False));time.sleep(30)
    if read(QUEUE/'queue.json')['config_sha256']!=digest(CONFIG):raise ValueError('200k handoff config differs')
    # Fresh structural fitting improves the original neural calibration baseline;
    # this is a scale/calibration study, not a claim it beats the smaller upgrade.
    fresh=read(ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_extra_v1/evaluation/report.json')
    base=read(ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/evaluation/report.json')
    if fresh['selected']['calibration_macro_f05']<=base['selected']['calibration_macro_f05']:
        raise ValueError('Fresh-data pilot did not improve calibration')
    account=aws('freetier','get-account-plan-state',region='us-east-1')
    receipt=read(BUDGET);deadline=budget_check(receipt,account,datetime.now(timezone.utc))
    if set(n['id'] for n in nodes)!={n['id'] for n in receipt['nodes']}:raise ValueError('Fleet differs from reviewed budget')
    write(out/'review.json',dict(utc=datetime.now(timezone.utc).isoformat(),budget_sha256=digest(BUDGET),
        account_plan='FREE',remaining_credits_usd=float(account['accountPlanRemainingCredits']['amount']),
        no_new_instances=True,no_plan_upgrade=True,no_timer_change=True,
        role_manifest_sha256=digest(ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1/manifest.json'),
        note='Data extraction only; no model fitted and no calibration labels inspected.'))
    while len(state['complete'])<8:
        if time.monotonic()>=end:raise TimeoutError('Finite extraction controller window expired; remote states preserved')
        if shutil.disk_usage(ROOT).free<8*1024**3:
            write(out/'progress.json',dict(stage='waiting_for_local_8gib_reserve',
                jobs_launched=len(state['launched']),completed_partitions=len(state['complete'])))
            time.sleep(30);continue
        for part in range(8):
            node=workers[part%len(workers)]
            key=str(part);item=plan['specs'][part]['spec']
            if key in state['complete']:continue
            if not part_can_start(part,state['complete'],assignment):continue
            if key not in state['launched']:
                health=preflight(node,deadline);payload=ship(node,plan)
                state['launched'][key]=dict(node=node['id'],health=health,payload=payload)
                write(state_path,state)
            try:status=invoke(node,'status',item)
            except RuntimeError as error:
                if 'FileNotFoundError' not in str(error):raise
                # Crash-safe restart: still idle before starting a recorded assignment.
                preflight(node,deadline);invoke(node,'start',item);status=invoke(node,'status',item)
            health=status['health']
            if health['available_memory_mib']<1024 or health['free_disk_gib']<5:
                invoke(node,'stop',item);raise RuntimeError('Cloud resource guard stopped only its own extraction')
            if status['finished']:
                if status['exit']['returncode']!=0:raise RuntimeError('Cloud extraction failed: '+status.get('error_tail',''))
                state['complete'][key]=collect_compact(node,part,plan);write(state_path,state)
            elif not status['active']:raise RuntimeError('Cloud extraction disappeared without exit state')
        write(out/'progress.json',dict(stage='extracting',jobs_launched=len(state['launched']),completed_partitions=len(state['complete']),partitions=8))
        if len(state['complete'])<8:time.sleep(30)
    if sum(p['owners'] for p in state['complete'].values())!=25000:raise ValueError('Final fresh-owner count mismatch')
    write(out/'completion.json',dict(complete=True,owners=25000,partitions=state['complete'],
        plan_sha256=state['plan_sha256'],role_manifest_sha256=digest(ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1/manifest.json'),
        training_only=True,model_fitted=False,ready_for_submission=False))
    write(out/'progress.json',dict(stage='complete',owners=25000,model_fitted=False,ready_for_submission=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--minutes',type=float,default=180);a=p.parse_args()
    if a.minutes<=0:p.error('Positive finite duration required')
    try:run(a.minutes)
    except Exception as error:
        write(WORK/'controller/failure.json',dict(error=str(error),utc=datetime.now(timezone.utc).isoformat()));raise

"""Freeze complete128partition inference scheduling without reading test labels."""
import argparse
import json
from pathlib import Path
from ber.shards import file_hash

def prepare(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    root=Path('artifacts/improvement_v2')
    release=root/'cpu/release_gate0_100k_cpu_v1.json'
    if not release.exists():raise ValueError('Release must be frozen first')
    nodes=[dict(id='local',local=True,workers=4)]
    for name,count in [('knn_fleet_v1',2),('submission_fleet_v1',2),('fleet_v1',4)]:
        fleet=root/'cloud'/name
        for i in range(count):
            ready=[str(fleet/'submission_deployment.json')]
            if name=='fleet_v1':
                ready.append(str(root/'cpu/cloud200k_ready_for_inference.json') if i==0
                    else str(fleet/'collected'/f'v2_gate0_train200k_p{i}'/'manifest.json'))
            nodes.append(dict(id=f'{name}_node{i}',run=str(fleet),node=i,workers=2,ready_files=ready))
    config=dict(partitions=128,expected_references=1732544,threshold=.76,owner_policy='disabled',
        release=str(release),release_sha256=file_hash(release),
        model_sha256=file_hash(root/'cpu/fit_gate0_100k_v1/enhanced.cbm'),
        runner_sha256=file_hash('tools/v2_run_pipeline.py'),nodes=nodes)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(config=str(output),partitions=128,cloud_workers=8,local_workers=4,
        release_sha256=config['release_sha256'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    prepare(**vars(p.parse_args()))

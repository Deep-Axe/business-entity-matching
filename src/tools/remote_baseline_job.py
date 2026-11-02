"""Launch/status a bounded project job on a dedicated temporary baseline worker."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def run(action, job, spec=None):
    root = Path(__file__).resolve().parents[1]
    if not job.replace("_", "").isalnum():
        raise ValueError("Unsafe job name")
    directory = root / "artifacts/remote_jobs"
    directory.mkdir(parents=True, exist_ok=True)
    state_path, log_path = directory / (job+".json"), directory / (job+".log")
    if action == "start":
        if state_path.exists():
            raise ValueError("Job already exists; inspect rather than launching twice")
        configuration = json.loads(spec)
        argv = configuration["argv"]
        if argv[:2] != [".venv/bin/python", "tools/run_direct_pipeline.py"]:
            raise ValueError("Only the direct baseline driver is allowed")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with log_path.open("x") as log:
            child = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        start_ticks = Path(f"/proc/{child.pid}/stat").read_text().split()[21]
        state = dict(job=job, pid=child.pid, start_ticks=start_ticks, argv=argv, started=time.time())
        state_path.write_text(json.dumps(state)+"\n")
        print(json.dumps(dict(job=job, pid=child.pid, started=True)), flush=True)
        return
    state = json.loads(state_path.read_text())
    pid = state["pid"]
    process = Path(f"/proc/{pid}/stat")
    active = process.exists() and process.read_text().split()[21] == state["start_ticks"]
    if action == "stop" and active:
        if os.getsid(pid) != pid:
            raise ValueError("Process session ownership changed")
        os.killpg(pid, signal.SIGTERM)
    argv = state["argv"]
    output = root / argv[argv.index("--output")+1]
    progress = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {}
    manifest = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").exists() else {}
    print(json.dumps(dict(job=job, active=active, complete=manifest.get("complete", False), progress=progress)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--spec")
    run(**vars(parser.parse_args()))

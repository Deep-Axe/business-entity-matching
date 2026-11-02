"""Read-only progress monitoring plus periodic recovery copies of committed cloud shards."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from aws_baseline_transfer import ssh
from aws_baseline_jobs import REMOTE


def run(run, specs, minutes=250, interval=60, backup_interval=900):
    run = Path(run).resolve()
    specifications = json.loads(Path(specs).read_text())
    state = json.loads((run / "state.json").read_text())
    started, backed_up = time.monotonic(), time.monotonic()
    def status(host):
        item = specifications[str(host["node"])]
        command = "cd "+shlex.quote(REMOTE)+" && "+shlex.join([
            ".venv/bin/python", "tools/remote_baseline_job.py", "status", "--job", item["job"]])
        try:
            result = subprocess.run(ssh(run, host)+[command], text=True, capture_output=True, timeout=90)
            if result.returncode:
                return dict(node=host["node"], error="status unavailable")
            return dict(node=host["node"], **json.loads(result.stdout))
        except subprocess.TimeoutExpired:
            return dict(node=host["node"], error="status timed out")
    while time.monotonic()-started < minutes*60:
        with ThreadPoolExecutor(max_workers=8) as pool:
            reports = list(pool.map(status, state["hosts"]))
        summary = dict(utc=datetime.now(timezone.utc).isoformat(), nodes=reports,
            cloud_completed_references=sum(r.get("progress", {}).get("completed_references", 0) for r in reports),
            sum_reported_queries_per_second=sum(r.get("progress", {}).get("generated_queries_per_second", 0) for r in reports))
        temporary = run / "fleet_progress.pending.json"
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        temporary.replace(run / "fleet_progress.json")
        with (run / "fleet_progress.jsonl").open("a") as log:
            log.write(json.dumps(summary)+"\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "nodes"}), flush=True)
        if time.monotonic()-backed_up >= backup_interval or all(r.get("complete") for r in reports):
            # Resolving the venv interpreter symlink selects the system Python,
            # losing the installed project/dependencies. Preserve this interpreter.
            result = subprocess.run([sys.executable, "tools/aws_baseline_jobs.py", "collect",
                "--run", str(run), "--specs", specs], text=True, capture_output=True)
            print(json.dumps(dict(recovery_copy_success=result.returncode == 0)), flush=True)
            copy_report = dict(utc=datetime.now(timezone.utc).isoformat(),
                success=result.returncode == 0, returncode=result.returncode,
                error=result.stderr[-1500:] if result.returncode else None)
            (run / "recovery_copy_status.json").write_text(json.dumps(copy_report, indent=2)+"\n")
            backed_up = time.monotonic()
        if all(r.get("complete") for r in reports):
            print("All cloud feature partitions complete and collection attempted", flush=True)
            return
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--minutes", type=float, default=250)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--backup-interval", type=float, default=900)
    run(**vars(parser.parse_args()))

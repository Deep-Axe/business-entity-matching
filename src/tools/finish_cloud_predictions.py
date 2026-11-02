"""Score completed cloud feature partitions with the frozen model, then collect.

Does not provision resources, modify IAM, or delete anything. Synchronizes only
explicit source/model files after a node's feature job has completed. Re-running
is safe: existing remote prediction jobs are inspected, never launched twice.
If all SSH connections fail, the scoped helper can replace this run's recorded
client /32 with the laptop's current /32; it never opens SSH to a broader range.
"""

import argparse
import importlib
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from aws_baseline_jobs import REMOTE
from aws_baseline_transfer import ssh
from ber.shards import file_hash


def remote(run, host, argv):
    command = "cd " + shlex.quote(REMOTE) + " && " + shlex.join(argv)
    result = subprocess.run(ssh(run, host) + [command], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Node {host['node']} remote operation failed ({result.returncode})")
    return result.stdout


def verify_collected(directory, expected_model, expected_runner):
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if not manifest["complete"] or config["model_sha256"] != expected_model:
        raise ValueError("Incomplete or wrong-model cloud partition")
    if config["runner_code_sha256"] != expected_runner:
        raise ValueError("Cloud runner does not match the frozen runner")
    expected = {i for i in range((config["reference_count"] + config["query_batch_size"] - 1)
                                // config["query_batch_size"])
                if i % config["partitions"] in config["assigned_partitions"]}
    found, count = set(), 0
    for done_file in (directory / "shards").glob("[0-9]*/done.json"):
        done = json.loads(done_file.read_text())
        if done["shard_id"] != int(done_file.parent.name):
            raise ValueError("Shard directory/metadata mismatch")
        found.add(done["shard_id"])
        count += done["references"]
        for name, digest in done["output_sha256"].items():
            if Path(name).name != name or file_hash(done_file.parent / name) != digest:
                raise ValueError("Collected cloud shard checksum mismatch")
    if found != expected or count != manifest["references"]:
        raise ValueError("Collected cloud partition coverage mismatch")
    return count


def run(run, specs, frozen, minutes=250, interval=60):
    run = Path(run).resolve()
    state = json.loads((run / "state.json").read_text())
    if not state.get("hosts_pinned"):
        raise ValueError("Hosts must be authenticated before synchronization")
    features = json.loads(Path(specs).read_text())
    frozen_path = Path(frozen)
    choice = json.loads(frozen_path.read_text())
    model = Path(choice["model"])
    if file_hash(model) != choice["model_sha256"]:
        raise ValueError("Frozen model changed")
    if file_hash("tools/run_direct_pipeline.py") != choice["runner_sha256"]:
        raise ValueError("Frozen prediction runner changed")
    for module, digest in choice["code_sha256"].items():
        if file_hash(importlib.import_module(module).__file__) != digest:
            raise ValueError(f"Frozen source changed: {module}")
    project = Path.cwd().resolve()
    model_relative = model.resolve().relative_to(project)
    predictions = {}
    for host in state["hosts"]:
        node = str(host["node"])
        source = features[node]
        argv = list(source["argv"])
        mode = argv.index("--mode")
        if argv[mode + 1] != "features":
            raise ValueError("Expected feature job specification")
        del argv[mode:mode + 2]
        output = f"artifacts/deadline_test_predictions_cloud_{node}_v1"
        argv[argv.index("--output") + 1] = output
        argv += ["--model", str(model_relative), "--threshold", str(choice["threshold"]),
                 "--from-features", source["output"]]
        predictions[node] = dict(job=f"test_predictions_{node}_v1", output=output, argv=argv)
    destination = run / "prediction_jobs.json"
    if destination.exists() and json.loads(destination.read_text()) != predictions:
        raise ValueError("Existing prediction specification differs")
    destination.write_text(json.dumps(predictions, indent=2) + "\n")
    complete = {}
    started = time.monotonic()

    def handle(host):
        node = str(host["node"])
        if node in complete:
            return complete[node]
        source, prediction = features[node], predictions[node]
        helper = [".venv/bin/python", "tools/remote_baseline_job.py"]
        try:
            feature_status = json.loads(remote(run, host, helper + ["status", "--job", source["job"]]))
            if not feature_status.get("complete"):
                return dict(node=int(node), phase="features", active=feature_status.get("active"),
                            references=feature_status.get("progress", {}).get("completed_references", 0))
            exists = remote(run, host, [".venv/bin/python", "-c",
                "from pathlib import Path; print(Path(" + repr("artifacts/remote_jobs/" + prediction["job"] + ".json") + ").exists())"]).strip()
            if exists != "True":
                # A completed feature job no longer needs the old resume-compatible runner.
                transport = shlex.join(ssh(run, host)[:-1])
                files = ["code/business_entity_resolution", "tools/run_direct_pipeline.py",
                         "tools/remote_baseline_job.py", str(model_relative), str(frozen_path)]
                result = subprocess.run(["rsync", "-aR", "--exclude=__pycache__", "-e", transport,
                    *files, "ubuntu@" + host["ip"] + ":" + REMOTE + "/"], capture_output=True, text=True, timeout=180)
                if result.returncode:
                    raise RuntimeError("Frozen prediction files failed to synchronize")
                hashes = remote(run, host, ["sha256sum", str(model_relative), "tools/run_direct_pipeline.py"]).splitlines()
                if [line.split()[0] for line in hashes] != [choice["model_sha256"], choice["runner_sha256"]]:
                    raise ValueError("Remote frozen model/runner checksum mismatch")
                remote(run, host, helper + ["start", "--job", prediction["job"],
                    "--spec", json.dumps({"argv": prediction["argv"]})])
            status = json.loads(remote(run, host, helper + ["status", "--job", prediction["job"]]))
            if not status.get("complete"):
                return dict(node=int(node), phase="scoring", active=status.get("active"),
                            references=status.get("progress", {}).get("completed_references", 0))
            result = subprocess.run([str(project / ".venv/bin/python"), "tools/aws_baseline_jobs.py", "collect",
                "--run", str(run), "--specs", str(destination), "--node", node], capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError("Completed predictions failed to collect")
            directory = run / "collected" / prediction["job"]
            count = verify_collected(directory, choice["model_sha256"], choice["runner_sha256"])
            return dict(node=int(node), phase="collected_verified", references=count, directory=str(directory))
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            return dict(node=int(node), phase="error", error=str(error))

    while time.monotonic() - started < minutes * 60:
        with ThreadPoolExecutor(max_workers=8) as pool:
            reports = list(pool.map(handle, state["hosts"]))
        for report in reports:
            if report["phase"] == "collected_verified":
                complete[str(report["node"])] = report
        summary = dict(utc=datetime.now(timezone.utc).isoformat(), nodes=reports,
                       all_collected_verified=len(complete) == len(state["hosts"]))
        pending = run / "prediction_progress.pending.json"
        pending.write_text(json.dumps(summary, indent=2) + "\n")
        pending.replace(run / "prediction_progress.json")
        print(json.dumps(summary), flush=True)
        if summary["all_collected_verified"]:
            return
        if all(r.get("phase") == "error" and "(255)" in r.get("error", "") for r in reports):
            result = subprocess.run([str(project / ".venv/bin/python"),
                "tools/refresh_baseline_ssh_source.py", "--run", str(run)],
                capture_output=True, text=True, timeout=120)
            print(json.dumps(dict(scoped_ssh_source_check_success=result.returncode == 0)), flush=True)
        time.sleep(interval)
    raise TimeoutError("Prediction watcher runtime expired; inspect retained jobs and artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "specs", "frozen"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--minutes", type=float, default=250)
    parser.add_argument("--interval", type=float, default=60)
    run(**vars(parser.parse_args()))

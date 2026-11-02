"""Validate a test subset with full target IDs, without changing original data."""

import argparse
import json
import resource
import shutil
import subprocess
import sys
from pathlib import Path

from ber.data import write_json
from ber.shards import file_hash
from ber.validation import validate_exports


def verify(store, references, predictions, resource_dir, output):
    store, references, predictions, resource_dir, output = map(
        Path, (store, references, predictions, resource_dir, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((predictions / "config.json").read_text())
    cache = json.loads((Path(config["features"]) / "config.json").read_text())
    if config["labels_available"] or cache["references_sha256"] != file_hash(references):
        raise ValueError("Expected exact unlabeled test references used for predictions")
    if Path(cache["store"]).resolve() != store.resolve():
        raise ValueError("Prediction and validation stores differ")
    if not json.loads((predictions / "metrics.json").read_text())["complete"]:
        raise ValueError("Predictions incomplete")
    test_dir = resource_dir / "dataset/test"
    validator = resource_dir / "utils/validate_submission.py"
    for required in (validator, test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"):
        if not required.is_file():
            raise FileNotFoundError(required)
    output.mkdir(parents=True)
    subset = output / "test_subset_view"
    subset.mkdir()
    shutil.copy2(references, subset / "test_source1.tsv")
    for name in ("test_source2.tsv", "test_source3.tsv"):
        (subset / name).symlink_to((test_dir / name).resolve())
    matching, candidates = (predictions / name for name in
                            ("matching_results.tsv", "candidate_pairs.tsv"))
    streaming = validate_exports(store, references, matching, candidates, output / "streaming.json")
    command = [sys.executable, str(validator), "--matching", str(matching),
               "--candidate", str(candidates), "--test-dir", str(subset), "--check-ids"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "official_validator.log").write_text(result.stdout)
    report = dict(valid=result.returncode == 0, exit_code=result.returncode,
        reference_rows=streaming["reference_rows"], check_ids=True,
        full_test_submission=False, original_targets_used_via_symlinks=True,
        official_validator_sha256=file_hash(validator),
        validator_peak_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        scope="Selected test S1 subset only; membership checked against all original test S2/S3 IDs. No accuracy evaluation.")
    write_json(output / "report.json", report)
    print(result.stdout, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if result.returncode:
        raise RuntimeError("Official validator failed; inspect preserved log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "references", "predictions", "resource-dir", "output"):
        parser.add_argument(f"--{name}", required=True)
    verify(**vars(parser.parse_args()))

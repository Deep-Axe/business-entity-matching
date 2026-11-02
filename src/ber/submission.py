"""Reproduce a packaged submission from the supplied test TSVs and bundled model."""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from .disk_index import build_index
from .shards import file_hash
from .store import build_store


def run(data_dir, work_dir, output, workers=8, bundle=None, resume=False):
    bundle = Path(bundle) if bundle else Path(__file__).resolve().parents[2]
    data_dir, work_dir, output = map(Path, (data_dir, work_dir, output))
    choice = json.loads((bundle / "model/frozen.json").read_text())
    model, lexicon = bundle / "model/model.cbm", bundle / "model/lexicon"
    if file_hash(model) != choice["model_sha256"] or file_hash(lexicon / "lexicon.json") != choice["lexicon_sha256"]:
        raise ValueError("Bundled model or training-only lexicon changed")
    for name, digest in choice["code_sha256"].items():
        if file_hash(importlib.import_module(name).__file__) != digest:
            raise ValueError(f"Bundled recipe source changed: {name}")
    if file_hash(bundle / "src/run_direct_pipeline.py") != choice["runner_sha256"]:
        raise ValueError("Bundled prediction runner changed")
    work_dir.mkdir(parents=True, exist_ok=True)
    store, index, predictions = work_dir / "test_store", work_dir / "test_index", work_dir / "predictions"
    build_store(data_dir, store, split="test", resume=resume and store.exists(), minimum_free_gib=8)
    build_index(store, index, threads=min(4, workers), heap_mb=256, commit_every=250000,
                resume=resume and index.exists(), minimum_free_gib=8)
    references = data_dir / "test/test_source1.tsv"
    command = [sys.executable, str(bundle / "src/run_direct_pipeline.py"),
        "--store", str(store), "--index", str(index), "--references", str(references),
        "--lexicon", str(lexicon), "--model", str(model), "--output", str(predictions),
        "--gate-terms", str(choice["gate_terms"]), "--threshold", str(choice["threshold"]),
        "--workers", str(workers), "--minimum-free-gib", "8"]
    if resume and predictions.exists():
        command.append("--resume")
    subprocess.run(command, check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        subprocess.run([sys.executable, str(bundle / "src/assemble_direct_submission.py"),
            "--sources", str(predictions), "--references", str(references), "--output", str(output)], check=True)
    elif not resume:
        raise FileExistsError(output)
    else:
        assembled = json.loads((output / "assembly.json").read_text())
        for key in ("model_sha256", "threshold", "gate_terms"):
            if assembled[key] != choice[key]:
                raise ValueError(f"Existing output belongs to another recipe: {key}")
        for name, digest in assembled["output_sha256"].items():
            if Path(name).name != name or file_hash(output / name) != digest:
                raise ValueError("Existing assembled output changed")
    subprocess.run([sys.executable, str(bundle / "src/validate_full_submission.py"),
        "--test-dir", str(data_dir / "test"), "--output", str(output), "--report", str(work_dir / "validation.json")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "work-dir", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bundle")
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))

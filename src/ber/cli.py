import argparse
from pathlib import Path

from .data import prepare_sample
from .experiment import run_experiment
from .store import build_store
from .disk_index import build_index
from .shards import featurize_shards, select_references, finalize_features
from .scalable_model import fit_shards, predict_shards, calibrate_shards
from .validation import validate_exports


def main():
    parser = argparse.ArgumentParser(description="Local entity-resolution baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate", help="Tune a frozen model's threshold on provided tuning labels")
    for name in ("tuning-features", "model", "output"):
        calibrate.add_argument(f"--{name}", type=Path, required=True)
    calibrate.add_argument("--top-k", type=int)
    calibrate.add_argument("--channels", nargs="+")
    calibrate.add_argument("--threads", type=int, default=4)
    finalize = sub.add_parser("finalize-features", help="Verify all parallel feature shards are complete")
    finalize.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Stream-check exported IDs and candidate/match consistency")
    for name in ("store", "references", "matching", "candidates"):
        verify.add_argument(f"--{name}", type=Path, required=True)
    verify.add_argument("--output-report", type=Path)
    fit = sub.add_parser("fit-shards", help="Fit/tune on disjoint cached pairs, optionally evaluate development")
    for name in ("train-features", "tune-features", "output"):
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--dev-features", type=Path)
    fit.add_argument("--top-k", type=int, default=10)
    fit.add_argument("--channels", nargs="+")
    fit.add_argument("--iterations", type=int, default=400)
    fit.add_argument("--depth", type=int, default=6)
    fit.add_argument("--learning-rate", type=float, default=.07)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--seed", type=int, default=2026)
    fit.add_argument("--context-features", action="store_true")
    predict = sub.add_parser("predict-shards", help="Predict and export incrementally with a fixed model/threshold")
    for name in ("features", "model", "output"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--threshold", type=float, required=True)
    predict.add_argument("--top-k", type=int)
    predict.add_argument("--channels", nargs="+")
    predict.add_argument("--threads", type=int, default=4)
    predict.add_argument("--resume", action="store_true")
    predict.add_argument("--stop-after-shards", type=int)
    select = sub.add_parser("select", help="Select disjoint train/tune/dev/final-audit references")
    select.add_argument("--store", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    for name, size in (("train", 12000), ("tune", 3000), ("dev", 3000), ("audit", 3000)):
        select.add_argument(f"--{name}-size", type=int, default=size)
    select.add_argument("--seed", type=int, default=2027)
    select.add_argument("--exclude-sample", type=Path)
    features = sub.add_parser("features", help="Retrieve and featurize in restartable bounded-memory shards")
    for name in ("store", "index", "references", "output"):
        features.add_argument(f"--{name}", type=Path, required=True)
    features.add_argument("--gallery-fold", choices=("train", "tune", "eval", "test"))
    features.add_argument("--top-k", type=int, default=20)
    features.add_argument("--transliteration", action="store_true")
    features.add_argument("--no-country-block", action="store_true")
    features.add_argument("--query-batch-size", type=int, default=200)
    features.add_argument("--resume", action="store_true")
    features.add_argument("--stop-after-shards", type=int)
    features.add_argument("--unlabeled", action="store_true")
    features.add_argument("--workers", type=int, default=1)
    features.add_argument("--worker-id", type=int, default=0)
    features.add_argument("--initialize-only", action="store_true")
    features.add_argument("--minimum-free-gib", type=float, default=10)
    features.add_argument("--max-df-ratio", type=float)
    features.add_argument("--frequency-mode", choices=("drop", "gate"), default="drop")
    features.add_argument("--lexicon", type=Path, help="training lexicon directory containing lexicon.json")
    features.add_argument("--gate-terms", type=int, default=0)
    index = sub.add_parser("build-index", help="Build/resume a bounded-memory lexical index")
    index.add_argument("--store", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--threads", type=int, default=4)
    index.add_argument("--heap-mb", type=int, default=256)
    index.add_argument("--commit-every", type=int, default=100000)
    index.add_argument("--resume", action="store_true")
    index.add_argument("--stop-after-commits", type=int)
    store = sub.add_parser("build-store", help="Stream TSVs into a resumable on-disk store")
    store.add_argument("--data-dir", type=Path, required=True)
    store.add_argument("--output", type=Path, required=True)
    store.add_argument("--split", choices=("train", "test"), default="train")
    store.add_argument("--seed", type=int, default=2026)
    store.add_argument("--batch-size", type=int, default=25000)
    store.add_argument("--resume", action="store_true")
    store.add_argument("--stop-after-batches", type=int)
    prepare = sub.add_parser("prepare", help="Create disjoint folds from original training data")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--train-size", type=int, default=6000)
    prepare.add_argument("--tune-size", type=int, default=1500)
    prepare.add_argument("--eval-size", type=int, default=1500)
    prepare.add_argument("--distractors", type=int, default=50000)
    prepare.add_argument("--seed", type=int, default=2026)
    run = sub.add_parser("run", help="Index, retrieve, fit, tune and evaluate locally")
    run.add_argument("--sample-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--transliteration", action="store_true")
    run.add_argument("--no-country-block", action="store_true")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--iterations", type=int, default=400)
    run.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "calibrate":
        calibrate_shards(**values)
    elif command == "finalize-features":
        finalize_features(**values)
    elif command == "verify":
        validate_exports(**values)
    elif command == "fit-shards":
        fit_shards(**values)
    elif command == "predict-shards":
        predict_shards(**values)
    elif command == "select":
        select_references(**values)
    elif command == "features":
        values["country_block"] = not values.pop("no_country_block")
        featurize_shards(**values)
    elif command == "build-index":
        build_index(**values)
    elif command == "build-store":
        build_store(**values)
    elif command == "prepare":
        prepare_sample(**values)
    else:
        values["country_block"] = not values.pop("no_country_block")
        run_experiment(**values)


if __name__ == "__main__":
    main()

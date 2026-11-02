"""Select a frozen model/budget/threshold on full-gallery tuning refs only."""

import argparse
import json
from pathlib import Path

from ber.data import write_json
from ber.scalable_model import calibrate_shards, predict_shards
from ber.shards import file_hash


def run(models, tune_features, dev_features, output, threads=4, try_k50=True):
    output = Path(output)
    config = dict(models=[str(Path(p).resolve()) for p in models],
        tune_features=str(Path(tune_features).resolve()), dev_features=str(Path(dev_features).resolve()),
        threads=threads, try_k50=try_k50)
    if output.exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Selection inputs changed; use a new output")
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
    entries = []
    for i, directory in enumerate(map(Path, models)):
        model = directory / "model.cbm"
        model_config = json.loads((directory / "config.json").read_text())
        native = model_config["top_k"]
        budgets = [native] + ([50] if try_k50 and native == 20 else [])
        for k in budgets:
            name = f"{i:02d}_{directory.name}_k{k}"
            calibration = output / name
            path = calibration / "threshold.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["model_sha256"] != file_hash(model):
                    raise ValueError("Model changed after calibration")
            else:
                report = calibrate_shards(tune_features, model, calibration,
                    top_k=k, channels=model_config["channels"], threads=threads)
            entry = dict(name=name, model=str(model.resolve()), top_k=k,
                channels=model_config["channels"], threshold=report["threshold"],
                tuning_macro_f05=report["tuning_macro_f05"], calibration=str(calibration.resolve()))
            entries.append(entry)
            print(json.dumps(entry), flush=True)
    entries.sort(key=lambda r: (-r["tuning_macro_f05"], r["top_k"], len(r["channels"]), r["name"]))
    best = entries[0]
    write_json(output / "selection.json", dict(selected=best, comparisons=entries,
        criterion="Full-gallery tuning macro F0.5 only; then smaller candidate budget/channel count.",
        final_audit_used=False))
    for name, features in (("selected_tune", tune_features), ("selected_dev", dev_features)):
        destination = output / name
        result = predict_shards(features, best["model"], best["threshold"], destination,
            top_k=best["top_k"], channels=best["channels"], threads=threads, resume=destination.exists())
        print(json.dumps({"set": name, "selected": best["name"], "matching": result["matching"],
                          "retrieval": result["retrieval"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    for name in ("tune-features", "dev-features", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-k50", dest="try_k50", action="store_false")
    run(**vars(parser.parse_args()))

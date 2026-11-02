"""Resolve v1 targets claimed by several S1 references using name/address evidence.

Only targets that the v1 submission assigns to more than one S1 change; every other
prediction is kept, and the v1 candidate file is reused unchanged (the new matches
are a subset of the old ones). Scores come from text only: v1 saved no probabilities.

  check  -- fit the claimant scorer and choose the keep/abstain rule on training
            labels, using same-core-name rivals from the full training S1 file.
  apply  -- apply the frozen rule to the v1 test output and write a new output folder.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset"
V1 = ROOT / "artifacts/duf-fel_full_output_v1"
SEED = 20260926

LEGAL = {"pvt": "private", "private": "private", "ltd": "limited", "limited": "limited",
         "llc": "llc", "llp": "llp", "inc": "inc", "incorporated": "inc", "corp": "corp",
         "corporation": "corp", "co": "co", "company": "co", "lp": "lp", "pc": "pc",
         "pllc": "pllc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
         "eurl": "eurl", "sa": "sa", "sci": "sci", "ei": "ei"}
HONORIFIC = {"shri", "sri", "smt", "dr", "mr", "mrs", "ms", "the", "m", "s"}
ADDRESS = {"rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
           "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
           "cir": "circle", "pl": "place", "hwy": "highway", "pkwy": "parkway", "trl": "trail",
           "r": "rue", "chem": "chemin", "all": "allee", "psg": "passage", "rte": "route",
           "imp": "impasse", "fbg": "faubourg", "crs": "cours"}
ADDRESS_FILLER = {"no", "n", "nr", "h", "hno", "door", "plot", "unit", "apt", "apartment",
                  "suite", "ste", "flat", "shop", "office", "pmb", "po", "box", "null", "na"}
FEATURES = ["name_set", "name_sort", "name_exact", "legal_agree", "addr_set", "addr_jaccard",
            "target_addr_missing", "house_in_target", "numbers_conflict", "numbers_missing",
            "rel_addr_set", "rel_house", "rel_name_set", "claimants"]


def prep_name(name):
    tokens = re.findall(r"[a-z0-9]+", anyascii(name).lower())
    legal = frozenset(LEGAL[t] for t in tokens if t in LEGAL)
    core = [t for t in tokens if t not in LEGAL and t not in HONORIFIC]
    return " ".join(core), legal


def prep_address(address):
    folded = anyascii(address).lower()
    words = [ADDRESS.get(t, t) for t in re.findall(r"[a-z]+", folded)]
    words = [t for t in words if t not in ADDRESS_FILLER]
    numbers = [t.lstrip("0") or "0" for t in re.findall(r"\d+", folded)]
    return " ".join(words), frozenset(words), numbers


def core_key(name):
    return " ".join(sorted(set(prep_name(name)[0].split())))


def pair_features(s_name, s_addr, t_name, t_addr):
    sc, sl = s_name
    tc, tl = t_name
    sw, sws, sn = s_addr
    tw, tws, tn = t_addr
    both_legal = bool(sl) and bool(tl)
    has_numbers = bool(sn) and bool(tn)
    return [fuzz.token_set_ratio(sc, tc) / 100, fuzz.token_sort_ratio(sc, tc) / 100,
            float(sc == tc and sc != ""),
            (1.0 if sl & tl else -1.0) if both_legal else 0.0,
            fuzz.token_set_ratio(sw, tw) / 100 if sw and tw else 0.0,
            len(sws & tws) / len(sws | tws) if sws and tws else 0.0,
            float(not tw and not tn),
            float(bool(sn) and sn[0] in tn),
            float(has_numbers and not set(sn) & set(tn)),
            float(not has_numbers)]


def case_matrix(rows):
    """rows: per-claimant base features of one target; appends within-target relative columns."""
    base = np.asarray(rows, dtype=np.float32)
    out = []
    for i in range(len(base)):
        others = np.delete(base, i, axis=0)
        out.append(list(base[i]) + [base[i, 4] - others[:, 4].max(), base[i, 7] - others[:, 7].max(),
                                    base[i, 0] - others[:, 0].max(), float(len(base))])
    return np.asarray(out, dtype=np.float32)


def decide(logits, keep_floor, margin):
    """Index of the retained claimant, or None to drop the target from every claimant."""
    order = np.argsort(-logits)
    top, second = logits[order[0]], logits[order[1]]
    if 1 / (1 + np.exp(-top)) >= keep_floor and top - second >= margin:
        return int(order[0])
    return None


def utility(cases, logits_list, keep_floor, margin, gain_fp, loss_tp):
    """Approximate owner-level F0.5 change versus keeping every claimant."""
    total, kept, correct = 0.0, 0, 0
    for (owner_pos, n), logits in zip(cases, logits_list):
        choice = decide(logits, keep_floor, margin)
        removed_wrong = n - (owner_pos is not None)
        if choice is None:
            total += removed_wrong * gain_fp - (loss_tp if owner_pos is not None else 0)
        else:
            kept += 1
            if choice == owner_pos:
                correct += 1
                total += removed_wrong * gain_fp
            else:
                total += (removed_wrong - 1) * gain_fp - (loss_tp if owner_pos is not None else 0)
    return total / len(cases), kept / len(cases), (correct / kept if kept else float("nan"))


def check(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    names, addresses, groups, index = [], [], {}, {}
    with open(DATA / "train/train_source1.tsv") as handle:
        next(handle)
        for line in handle:
            entity, name, address, country = line.rstrip("\n").split("\t")
            i = len(names)
            names.append(name)
            addresses.append(address)
            index[entity] = i
            groups.setdefault((country, core_key(name)), []).append(i)
    store = sqlite3.connect(f"file:{ROOT}/artifacts/train_store_v2/records.sqlite3?mode=ro", uri=True)
    rng = random.Random(SEED)
    max_row = store.execute("select max(row_id) from records").fetchone()[0]
    rows = rng.sample(range(1, max_row + 1), 120000)
    cases = {"fit": [], "tune": [], "eval": []}
    for row in rows:
        entity, name, address, country = store.execute(
            "select entity_id, name, address, country from records where row_id=?", (row,)).fetchone()
        key = (country, core_key(name))
        group = groups.get(key, [])
        owner = store.execute("select reference_id from labels where target_id=?", (entity,)).fetchone()
        owner = index[owner[0]] if owner else None
        t_name, t_addr = prep_name(name), prep_address(address)
        claimants = []
        if owner is not None and (owner in group or
                                  fuzz.token_set_ratio(prep_name(names[owner])[0], t_name[0]) >= 85):
            claimants.append(owner)
        rivals = [i for i in group if i != owner]
        claimants += rng.sample(rivals, min(3, len(rivals)))
        if len(claimants) < 2:
            continue
        rng.shuffle(claimants)
        matrix = case_matrix([pair_features(prep_name(names[i]), prep_address(addresses[i]), t_name, t_addr)
                              for i in claimants])
        owner_pos = claimants.index(owner) if owner in claimants else None
        split = ("fit", "fit", "tune", "eval")[int(hashlib.sha1(repr(key).encode()).hexdigest(), 16) % 4]
        cases[split].append((matrix, owner_pos, country))
    x = np.concatenate([m for m, _, _ in cases["fit"]])
    y = np.concatenate([[float(j == p) for j in range(len(m))] for m, p, _ in cases["fit"]])
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x, y)

    def scored(split):
        return ([(p, len(m)) for m, p, _ in cases[split]],
                [model.decision_function(m) for m, _, _ in cases[split]])

    grid = [(f, m) for f in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
    tune_cases, tune_logits = scored("tune")
    ranked = sorted(grid, key=lambda g: -utility(tune_cases, tune_logits, *g, 0.167, 0.0625)[0])
    keep_floor, margin = ranked[0]
    report = {"seed": SEED, "sampled_targets": len(rows), "features": FEATURES,
              "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
              "selected_on_tune": {"keep_floor": keep_floor, "margin": margin},
              "utility_note": "approximate owner-level gain vs keeping all claimants; asymmetric uses "
                              "+0.167 per removed wrong edge, -0.0625 per removed true edge; symmetric "
                              "uses 0.1 for both", "splits": {}}
    for split in ("tune", "eval"):
        split_cases, logits = scored(split)
        owner_present = np.mean([p is not None for p, _ in split_cases])
        entry = {"cases": len(split_cases), "owner_present_fraction": float(owner_present),
                 "mean_claimants": float(np.mean([n for _, n in split_cases]))}
        for label, policy in (("selected", (keep_floor, margin)), ("argmax_only", (0.0, 0.0)),
                              ("abstain_all", (1.1, 0.0))):
            asym = utility(split_cases, logits, *policy, 0.167, 0.0625)
            sym = utility(split_cases, logits, *policy, 0.1, 0.1)
            entry[label] = {"asymmetric_gain_per_case": asym[0], "symmetric_gain_per_case": sym[0],
                            "kept_fraction": asym[1], "kept_accuracy": asym[2]}
        for country in sorted({c for _, _, c in cases[split]}):
            sub = [(i, c) for i, (_, _, c) in enumerate(cases[split]) if c == country]
            idx = [i for i, _ in sub]
            asym = utility([split_cases[i] for i in idx], [logits[i] for i in idx],
                           keep_floor, margin, 0.167, 0.0625)
            entry[f"selected_{country}"] = {"cases": len(idx), "asymmetric_gain_per_case": asym[0],
                                            "kept_fraction": asym[1], "kept_accuracy": asym[2]}
        report["splits"][split] = entry
    report["seconds"] = time.time() - start
    (output / "check_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def apply(check_report, output):
    output = Path(output)
    (output / "output").mkdir(parents=True, exist_ok=False)
    start = time.time()
    frozen = json.loads(Path(check_report).read_text())
    coef, intercept = np.asarray(frozen["coefficients"]), frozen["intercept"]
    keep_floor, margin = frozen["selected_on_tune"]["keep_floor"], frozen["selected_on_tune"]["margin"]
    rows = []
    claims = {}
    with open(V1 / "matching_results.tsv") as handle:
        header = handle.readline()
        for i, line in enumerate(handle):
            reference, _, matched = line.rstrip("\n").partition("\t")
            targets = [t for t in matched.split(",") if t]
            rows.append((reference, targets))
            for t in targets:
                claims.setdefault(t, []).append(i)
    conflicts = {t: owners for t, owners in claims.items() if len(owners) > 1}
    del claims
    needed = {i for owners in conflicts.values() for i in owners}
    s1 = {}
    with open(DATA / "test/test_source1.tsv") as handle:
        next(handle)
        for i, line in enumerate(handle):
            if i in needed:
                entity, name, address, country = line.rstrip("\n").split("\t")
                if entity != rows[i][0]:
                    raise ValueError("Reference order differs from v1 output")
                s1[i] = (prep_name(name), prep_address(address), country)
    store = sqlite3.connect(f"file:{ROOT}/artifacts/test_store_v2/records.sqlite3?mode=ro", uri=True)
    removed = {}
    stats = {"conflicting_targets": len(conflicts), "conflicting_edges": 0, "kept": 0, "abstained": 0,
             "removed_edges": 0, "by_country": {}}
    for target, owners in conflicts.items():
        name, address = store.execute("select name, address from records where entity_id=?",
                                      (target,)).fetchone()
        t_name, t_addr = prep_name(name), prep_address(address)
        matrix = case_matrix([pair_features(s1[i][0], s1[i][1], t_name, t_addr) for i in owners])
        choice = decide(matrix @ coef + intercept, keep_floor, margin)
        country = s1[owners[0]][2]
        c = stats["by_country"].setdefault(country, {"targets": 0, "edges": 0, "kept": 0, "removed_edges": 0})
        c["targets"] += 1
        c["edges"] += len(owners)
        stats["conflicting_edges"] += len(owners)
        stats["kept" if choice is not None else "abstained"] += 1
        c["kept"] += choice is not None
        for j, i in enumerate(owners):
            if j != choice:
                removed.setdefault(i, set()).add(target)
                stats["removed_edges"] += 1
                c["removed_edges"] += 1
    matching = output / "output/matching_results.tsv"
    with open(matching, "w") as handle:
        handle.write(header)
        for i, (reference, targets) in enumerate(rows):
            drop = removed.get(i, ())
            handle.write(reference + "\t" + ",".join(t for t in targets if t not in drop) + "\n")
    os.link(V1 / "candidate_pairs.tsv", output / "output/candidate_pairs.tsv")
    stats["predicted_edges_before"] = sum(len(t) for _, t in rows)
    stats["predicted_edges_after"] = stats["predicted_edges_before"] - stats["removed_edges"]
    stats["frozen_rule"] = {"check_report": str(check_report), "keep_floor": keep_floor, "margin": margin}
    stats["candidate_pairs"] = "hard link to unchanged v1 candidate_pairs.tsv"
    stats["seconds"] = time.time() - start
    (output / "apply_report.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").add_argument("--output", required=True)
    applied = sub.add_parser("apply")
    applied.add_argument("--check-report", required=True)
    applied.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(args.output)
    else:
        apply(args.check_report, args.output)

"""Training-only token correspondences for a retrieval-only transliteration probe.

This is not a new achieved matching score. It measures whether supervised query
expansion adds missed positives to an existing lexical candidate union.
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tantivy
from rapidfuzz.fuzz import ratio

from ber.data import read_tsv, write_json, write_tsv
from ber.disk_index import DiskRetriever
from ber.metrics import retrieval_metrics
from ber.shards import ablate_arrays, gold_for, load_shards
from ber.store import connect, lookup, record_row
from ber.text import Record


def align(left, right):
    """Order-preserving token alignment with explicit insertion/deletion costs."""
    n, m = len(left), len(right)
    cost = np.zeros((n + 1, m + 1))
    cost[:, 0], cost[0, :] = np.arange(n + 1) * .6, np.arange(m + 1) * .6
    back = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [cost[i - 1, j - 1] + 1 - ratio(left[i - 1], right[j - 1]) / 100,
                       cost[i - 1, j] + .6, cost[i, j - 1] + .6]
            choice = int(np.argmin(options))
            cost[i, j], back[i, j] = options[choice], choice
    pairs = []
    while n and m:
        choice = back[n, m]
        if choice == 0:
            a, b = left[n - 1], right[m - 1]
            if min(len(a), len(b)) >= 3 and ratio(a, b) >= 30:
                pairs.append((a, b))
            n, m = n - 1, m - 1
        elif choice == 1:
            n -= 1
        else:
            m -= 1
    return pairs


def learn(store, references, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = list(read_tsv(references))
    ids = [r["entity_id"] for r in rows]
    db = connect(store)
    anchors = lookup(db, "anchors", "entity_id", ids)
    if len(anchors) != len(ids) or any(r["fold"] != "train" for r in anchors.values()):
        raise ValueError("Learn only from unique provided training-fold references")
    if any(record_row(anchors[r["entity_id"]]) != r for r in rows):
        raise ValueError("Reference contents differ from training store")
    gold = gold_for(db, ids)
    targets = lookup(db, "records", "entity_id", [t for values in gold.values() for t in values])
    evidence, aligned_refs, pairs_used = Counter(), 0, 0
    for row in rows:
        reference = Record.from_row(row)
        if reference.country != "India" or not reference.original_name.isascii():
            continue
        current = set()
        for target_id in gold[reference.entity_id]:
            target_row = targets[target_id]
            if target_row["fold"] != "train":
                raise ValueError("Target owner split leakage")
            target = Record.from_row(record_row(target_row))
            if target.original_name.isascii():
                continue
            current.update(align(reference.ascii_name.split(), target.ascii_name.split()))
            pairs_used += 1
        evidence.update(current)  # Each reference contributes at most one vote per mapping.
        aligned_refs += bool(current)
    candidates = defaultdict(list)
    for (left, right), count in evidence.items():
        if left != right and count >= 2:
            candidates[left].append((right, count))
    lexicon = {word: sorted(values, key=lambda p: (-p[1], p[0]))[:6]
               for word, values in sorted(candidates.items())}
    output.mkdir(parents=True)
    write_tsv(output / "training_reference_ids.tsv", ({"entity_id": eid} for eid in ids), ["entity_id"])
    write_json(output / "lexicon.json", lexicon)
    report = dict(training_references=len(ids), aligned_references=aligned_refs,
        positive_cross_script_pairs_used=pairs_used, expanded_words=len(lexicon),
        variant_mappings=sum(map(len, lexicon.values())), min_distinct_training_references=2,
        max_variants_per_word=6, external_data_used=False,
        warning="Heuristic learned token alignments; not a verified linguistic dictionary.")
    write_json(output / "manifest.json", report)
    db.close()
    print(json.dumps(report, indent=2), flush=True)


def evaluate(baseline_features, lexicon, output, baseline_k=20, expansion_k=20, scoring="flat"):
    baseline_features, lexicon, output = map(Path, (baseline_features, lexicon, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((baseline_features / "config.json").read_text())
    if not json.loads((baseline_features / "manifest.json").read_text())["complete"] or not config["labels_available"]:
        raise ValueError("Need a complete labeled feature cache")
    if baseline_k > config["top_k_per_channel"]:
        raise ValueError("Baseline k exceeds cached ranks")
    a = load_shards(baseline_features, top_k=baseline_k)
    training_ids = {r["entity_id"] for r in read_tsv(lexicon / "training_reference_ids.tsv")}
    if training_ids & set(a["reference_ids"]):
        raise ValueError("Expansion evaluation references overlap lexicon fitting")
    mappings = json.loads((lexicon / "lexicon.json").read_text())
    db = connect(config["store"])
    anchors = lookup(db, "anchors", "entity_id", list(a["reference_ids"]))
    retriever = DiskRetriever(config["index"], channels=["ascii_words"],
        country_block=config["country_block"], fold=config["gallery_fold"],
        max_df_ratio=config["max_df_ratio"], frequency_mode=config["frequency_mode"])
    original = defaultdict(set)
    for q, target in zip(a["query_ids"], a["target_ids"]):
        original[int(q)].add(str(target))
    extra_queries, extra_labels, rows = [], [], []
    started, queried, newly_covered = time.perf_counter(), 0, 0
    for q, eid in enumerate(a["reference_ids"]):
        record = Record.from_row(record_row(anchors[eid]))
        expanded = set(record.ascii_name.split())
        if record.country == "India":
            for word in tuple(expanded):
                expanded.update(variant for variant, count in mappings.get(word, []))
        extra = []
        if expanded != set(record.ascii_name.split()):
            queried += 1
            full, gate = retriever._field_parts("ascii_words", expanded)
            if scoring in ("grouped", "grouped_combined"):
                families = []
                # A rare spelling of a common word must not receive the same
                # IDF advantage as a genuinely rare business-name word.
                words = sorted(set(record.ascii_name.split()),
                    key=lambda word: (retriever.frequency("ascii_words", word), word))[:12]
                for word in words:
                    variants = {word, *(variant for variant, count in mappings.get(word, []))}
                    terms = [(tantivy.Occur.Should, tantivy.Query.term_query(
                        retriever.schema, "ascii_words", variant, "freq")) for variant in sorted(variants)
                        if retriever.frequency("ascii_words", variant)]
                    if terms:
                        group = tantivy.Query.boolean_query(terms, minimum_number_should_match=1)
                        weight = math.log1p(retriever.searcher.num_docs /
                                           (retriever.frequency("ascii_words", word) + 1))
                        families.append((tantivy.Occur.Should, tantivy.Query.const_score_query(group, weight)))
                full = tantivy.Query.boolean_query(families, minimum_number_should_match=1) if families else None
                if scoring == "grouped_combined" and full is not None:
                    address, address_gate = retriever._field_parts("address_words", record.address_tokens)
                    if address is not None:
                        full = tantivy.Query.boolean_query([
                            (tantivy.Occur.Should, tantivy.Query.boost_query(full, 2.)),
                            (tantivy.Occur.Should, address)], minimum_number_should_match=1)
                        if gate is not None and address_gate is not None:
                            gate = tantivy.Query.boolean_query([(tantivy.Occur.Should, gate),
                                (tantivy.Occur.Should, address_gate)], minimum_number_should_match=1)
            clauses = [(tantivy.Occur.Must, full)] if full is not None else []
            if gate is not None:
                clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(gate, 0.)))
            for field, value in (("country", record.country if config["country_block"] else None), ("fold", config["gallery_fold"])):
                if value is not None:
                    clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(
                        tantivy.Query.term_query(retriever.schema, field, value, "basic"), 0.)))
            if full is not None:
                hits = retriever.searcher.search(tantivy.Query.boolean_query(clauses), limit=expansion_k, count=False).hits
                row_ids = retriever.searcher.fast_field_values("row_id", [address for _, address in hits])
                targets = lookup(db, "records", "row_id", row_ids)
                for row_id in row_ids:
                    target_id = targets[row_id]["entity_id"]
                    if target_id in original[q]:
                        continue
                    extra.append(target_id)
                    extra_queries.append(q)
                    positive = int(target_id in a["gold"][str(eid)])
                    extra_labels.append(positive)
                    newly_covered += positive
        rows.append({"source1_entity_id": str(eid), "extra_candidate_ids": ",".join(extra)})
    base_report = retrieval_metrics(a["query_ids"], a["y"], a["true_counts"])
    combined = retrieval_metrics(np.concatenate((a["query_ids"], np.array(extra_queries, dtype=int))),
        np.concatenate((a["y"], np.array(extra_labels, dtype=np.int8))), a["true_counts"])
    report = dict(baseline_features=str(baseline_features.resolve()), baseline_k=baseline_k,
        expansion_k=expansion_k, scoring=scoring, references=len(a["reference_ids"]), expanded_queries=queried,
        extra_candidates=len(extra_labels), newly_retrieved_positives=newly_covered,
        baseline=base_report, augmented=combined, seconds=time.perf_counter() - started,
        external_data_used=False, warning="Retrieval-only probe. Added candidates have not passed a matching classifier; oracle metrics are upper bounds, not achieved matching accuracy.")
    output.mkdir(parents=True)
    write_tsv(output / "extra_candidates.tsv", rows, ["source1_entity_id", "extra_candidate_ids"])
    write_json(output / "metrics.json", report)
    db.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("learn")
    for name in ("store", "references", "output"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("evaluate")
    for name in ("baseline-features", "lexicon", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--baseline-k", type=int, default=20)
    p.add_argument("--expansion-k", type=int, default=20)
    p.add_argument("--scoring", choices=("flat", "grouped", "grouped_combined"), default="flat")
    args = vars(parser.parse_args())
    command = args.pop("command")
    (learn if command == "learn" else evaluate)(**args)

"""Compare the completed pilot with local runs and estimate, never promise, cost."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ber.data import write_json

TEST_POPULATION = {"US": 663106, "India": 809986, "France": 259452}


def summarize(run):
    run = Path(run)
    cloud = run / "cloud_results"
    complete = json.loads((cloud / "complete.json").read_text())
    if not complete["complete"]:
        raise ValueError("Cloud workload is not complete")
    locations = {"local_1": run / "local_one_worker", "local_2": run / "local_two_workers",
        "local_4": run / "local_four_workers", "aws_1": cloud / "one_worker", "aws_2": cloud / "two_workers"}
    if (run / "local_eight_workers/summary.json").is_file():
        locations["local_8"] = run / "local_eight_workers"
    if (run / "local_eight_sustained/summary.json").is_file():
        locations["local_8_sustained"] = run / "local_eight_sustained"
    reports, checksums, signatures = {}, set(), set()
    for name, location in locations.items():
        result = json.loads((location / "summary.json").read_text())
        config = result["config"]
        signatures.add(tuple(config[k] for k in ("index_records", "references_sha256",
            "lexicon_sha256", "benchmark_code_sha256", "seed", "references_per_round")))
        rounds = result["rounds"]
        hours = []
        for r in rounds:
            checksums.add(r["candidate_ids_sha256"])
            sample_work = sum(c["queries"] * c["mean_query_seconds"] for c in r["countries"].values())
            projected_work = sum(TEST_POPULATION[country] * c["mean_query_seconds"]
                                 for country, c in r["countries"].items())
            hours.append(r["wall_seconds"] * projected_work / sample_work / 3600)
        reports[name] = dict(workers=config["workers"], rounds=len(rounds),
            measured_queries=sum(r["references"] for r in rounds),
            measured_search_wall_seconds=sum(r["wall_seconds"] for r in rounds),
            aggregate_queries_per_second=sum(r["references"] for r in rounds) / sum(r["wall_seconds"] for r in rounds),
            round_queries_per_second=[r["references_per_second"] for r in rounds],
            maximum_worker_peak_rss_MiB=max(r["maximum_worker_peak_rss_MiB"] for r in rounds),
            country_mix_adjusted_full_search_hours_median=float(np.median(hours)),
            country_mix_adjusted_full_search_hours_range=[min(hours), max(hours)])
    if len(signatures) != 1:
        raise ValueError("Benchmarks used different inputs or code")
    training = json.loads((cloud / "training/metrics.json").read_text())
    prices = json.loads((run / "prices.json").read_text())
    hourly = float(prices["prices"]["instance"]["PricePerUnit"]) + prices["ipv4_hourly_usd"]
    # Larger 48 GiB disk for possible full inference: not provisioned in this pilot.
    hourly += 48 * float(prices["prices"]["gp3"]["PricePerUnit"]) / 720
    cloud_hours = reports["aws_2"]["country_mix_adjusted_full_search_hours_median"]
    local_hours = reports["local_4"]["country_mix_adjusted_full_search_hours_median"]
    estimates = dict(one_cloud_search_hours=cloud_hours,
        one_cloud_48_gib_list_price_search_usd=cloud_hours * hourly,
        two_equal_cloud_hosts_search_hours=cloud_hours / 2,
        local_plus_two_cloud_hosts_search_hours=1 / (1 / local_hours + 2 / cloud_hours))
    if "local_8" in reports:
        key = "local_8_sustained" if "local_8_sustained" in reports else "local_8"
        local_eight = reports[key]["country_mix_adjusted_full_search_hours_median"]
        estimates["local_eight_workers_search_hours"] = local_eight
        estimates["local_eight_plus_two_cloud_hosts_search_hours"] = 1 / (1 / local_eight + 2 / cloud_hours)
    report = dict(measurements=reports, identical_candidate_ids_all_hosts_rounds=len(checksums) == 1,
        candidate_ids_sha256=sorted(checksums),
        training=dict(references=12000, cpu_threads=2, seconds=training["training_seconds"],
            total_seconds=training["elapsed_seconds"], peak_rss_MiB=training["peak_rss_MiB"],
            model_adopted_for_submission=False),
        search_only_linear_planning_estimates=estimates,
        planning_caveat="Not a commitment or full inference estimate: fixed repeated sample, warm caches, Flex availability, country mix, parallel scaling, record lookup, features, scoring, transfer, assembly, and retries can change runtime/cost. Future resources are not authorized or provisioned.")
    state = json.loads((run / "state.json").read_text())
    report["cleanup_status"] = state["status"]
    if state["status"] == "cleaned_up":
        duration = (datetime.fromisoformat(state["cleanup_utc"]) - datetime.fromisoformat(state["launch_time"])).total_seconds()
        pilot_hourly = float(prices["prices"]["instance"]["PricePerUnit"]) + prices["ipv4_hourly_usd"]
        pilot_hourly += state["root_volume_gib"] * float(prices["prices"]["gp3"]["PricePerUnit"]) / 720
        report["pilot_cost_estimate"] = dict(elapsed_to_cleanup_seconds=duration,
            compute_disk_ipv4_usd=pilot_hourly * duration / 3600, approved_ceiling_usd=5,
            posted_charge_known=False,
            note="Approximate upper runtime through cleanup; excludes tax and small transfer costs. Billing can lag.")
    write_json(run / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    summarize(**vars(parser.parse_args()))

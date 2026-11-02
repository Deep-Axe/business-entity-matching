"""Replace this run's exact SSH /32 when the authorized laptop's public IP changes.

Never opens SSH to a network range, changes IAM, or touches an unrelated group.
Only the uniquely tagged run's recorded TCP/22 ingress rule can be modified.
"""

import argparse
import ipaddress
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from aws_baseline_fleet import TAG, save
from aws_benchmark_ctl import aws


def run(run):
    run = Path(run).resolve()
    state = json.loads((run / "state.json").read_text())
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=15) as response:
        address = ipaddress.IPv4Address(response.read().decode().strip())
    if not address.is_global:
        raise ValueError("Expected the laptop's global public IPv4 address")
    current = str(address) + "/32"
    previous = state["ssh_source_cidr"]
    if ipaddress.IPv4Network(previous).prefixlen != 32:
        raise ValueError("Recorded rule is not a single-client /32")
    if current == previous:
        print("Client IP unchanged; no rule modified", flush=True)
        return
    group = aws("ec2", "describe-security-groups", "--group-ids", state["security_group_id"])["SecurityGroups"][0]
    if {t["Key"]: t["Value"] for t in group.get("Tags", [])}.get(TAG) != state["run_id"]:
        raise ValueError("Security group is not owned by this baseline run")
    rules = aws("ec2", "describe-security-group-rules", "--filters",
                "Name=group-id,Values=" + state["security_group_id"])["SecurityGroupRules"]
    inbound = [r for r in rules if not r["IsEgress"]]
    if (len(inbound) != 1 or inbound[0].get("IpProtocol") != "tcp"
            or inbound[0].get("FromPort") != 22 or inbound[0].get("ToPort") != 22
            or inbound[0].get("CidrIpv4") != previous):
        raise ValueError("Unexpected ingress rules; refusing to modify them")
    aws("ec2", "modify-security-group-rules", "--group-id", state["security_group_id"],
        "--security-group-rules", json.dumps([{
            "SecurityGroupRuleId": inbound[0]["SecurityGroupRuleId"],
            "SecurityGroupRule": {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                                  "CidrIpv4": current, "Description": state["run_id"]}}]))
    state.setdefault("ssh_source_history", []).append(dict(previous=previous, current=current,
        utc=datetime.now(timezone.utc).isoformat()))
    state["ssh_source_cidr"] = current
    save(run, state)
    print("Updated only this run's SSH rule to the laptop's current /32; previous address replaced", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    run(**vars(parser.parse_args()))

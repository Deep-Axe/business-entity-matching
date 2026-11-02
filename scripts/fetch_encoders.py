"""Download pinned encoder checkpoints from Hugging Face into BER_WEIGHTS.

Usage (from the repository root, after `source scripts/env.sh`):
  python scripts/fetch_encoders.py              # all eight encoders
  python scripts/fetch_encoders.py --tag kA     # one member
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = os.environ.get("BER_HF_REPO", "DeeAxe/business-entity-matching-encoders")
# Immutable snapshot. Override with BER_HF_REVISION only for debugging.
REVISION = os.environ.get("BER_HF_REVISION", "9b04e5a37a28c9d73cd3dfa9953a399b8e18a81e")

ASSETS = {
    "e5_cont1300k": "e5_cont1300k/model.pt",
    "kA": "kA/model.pt",
    "kB": "kB/model.pt",
    "kC": "kC/model.pt",
    "kD": "kD/model.pt",
    "kE": "kE/model.pt",
    "kF": "kF/model.pt",
    "kG": "kG/model.pt",
}


def weights_dir() -> Path:
    pkg = Path(__file__).resolve().parents[1]
    return Path(os.environ.get("BER_WEIGHTS", pkg / "model" / "encoders"))


def expected_hashes(sums: Path) -> dict[str, str]:
    out = {}
    for line in sums.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        out[name.strip().lstrip("*")] = digest
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_asset(tag: str, root: Path | None = None, expected: dict[str, str] | None = None) -> Path:
    if tag not in ASSETS:
        raise SystemExit(f"unknown asset {tag}")
    rel = ASSETS[tag]
    root = Path(root) if root is not None else weights_dir()
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected is None:
        sums = root / "SHA256SUMS"
        expected = expected_hashes(sums) if sums.exists() else None
    if dest.exists() and expected and expected.get(rel) == sha256(dest):
        print(f"ok {tag} (cached {REVISION[:12]})", flush=True)
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface-hub  (see requirements.gpu.txt)") from e
    path = Path(hf_hub_download(REPO, rel, revision=REVISION, local_dir=str(root)))
    if expected and rel in expected:
        got = sha256(path)
        if got != expected[rel]:
            path.unlink(missing_ok=True)
            raise SystemExit(f"SHA256 mismatch for {rel}@{REVISION}: {got}")
    print(f"ok {tag} rev {REVISION[:12]} -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", choices=sorted(ASSETS), help="one encoder (default: all eight)")
    ap.add_argument("--weights", type=Path, help="destination parent (default: BER_WEIGHTS)")
    args = ap.parse_args()
    root = args.weights or weights_dir()
    sums = root / "SHA256SUMS"
    expected = expected_hashes(sums) if sums.exists() else None
    tags = (args.tag,) if args.tag else tuple(ASSETS)
    for tag in tags:
        fetch_asset(tag, root, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())

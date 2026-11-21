"""Copy evidence out of the private lab checkout, byte for byte.

The lab notebook (``portfolio/``, ``acceptance_runs/``) is where the numbers are
produced and audited. This public repo may only carry copies, and a copy that
drifts from its source is worse than no copy at all -- readers would trust a
number the lab no longer stands behind.

So the rule is: copy bytes, then read the destination back and compare sha256
with the source. If they differ, fail loudly instead of leaving a half-synced
file behind. ``--check`` compares without writing, which is what CI and
``scripts/check_public.py`` want.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # publish/quantscope
LAB_ROOT = ROOT.parents[1]  # the private project root

# destination (inside this repo) -> source (inside the lab)
COPIES = {
    "docs/evidence/ff_controlled_evidence.json": "portfolio/ff_controlled_evidence.json",
    "docs/evidence/ff_evidence.json": (
        "acceptance_runs/20261121T1810Z-public-ff-evidence/ff_evidence.json"
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare only, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite a destination that differs")
    args = parser.parse_args(argv)

    problems: list[str] = []
    for dest_rel, src_rel in sorted(COPIES.items()):
        src = LAB_ROOT / src_rel
        dest = ROOT / dest_rel
        if not src.is_file():
            problems.append(f"missing source: {src_rel}")
            continue
        src_hash = sha256_path(src)
        if dest.is_file():
            dest_hash = sha256_path(dest)
            if dest_hash == src_hash:
                print(f"unchanged  {dest_rel}  sha256={src_hash[:16]}..")
                continue
            if args.check or not args.force:
                problems.append(
                    f"drift: {dest_rel} (copy {dest_hash[:12]}.. != source {src_hash[:12]}..)"
                )
                continue
        if args.check:
            problems.append(f"missing copy: {dest_rel}")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        written = sha256_path(dest)
        if written != src_hash:
            problems.append(f"copy verification failed: {dest_rel} {written[:12]}.. != {src_hash[:12]}..")
            continue
        print(f"copied     {dest_rel}  sha256={src_hash[:16]}..  bytes={dest.stat().st_size}")

    for line in problems:
        print(f"ERROR {line}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare two FF+ternary manifests and exit non-zero if the direction is lost.

This is the guard the CI smoke job runs. It deliberately checks *less* than the
published table claims, because a 4,000-image, one-minute run cannot support the
same sentence as a full-MNIST, three-seed experiment:

* the two manifests must describe the same settings except for the grouping;
* both runs must be clearly better than chance;
* ``aligned`` must not be worse than ``row`` (tolerance below).

What it does **not** check: the size of the gap, or that either number matches the
published table. Those need the full training set and several seeds; see
``docs/forward-forward-ternary.md``.

    python assert_direction.py --aligned runs/aligned-s0.json --row runs/row-s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "quantscope-ff-mnist-v1"
SHARED_SETTINGS = (
    "pixels",
    "labels",
    "in_dim",
    "n_layers",
    "hidden",
    "batch_size",
    "lr",
    "theta",
    "train",
    "test",
    "epochs_per_layer",
)


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema", "group", "seed", "settings", "result", "quantisation"):
        if key not in payload:
            raise SystemExit(f"{path}: missing key {key!r} (is this a run manifest?)")
    if payload["schema"] != SCHEMA:
        raise SystemExit(f"{path}: schema {payload['schema']!r}, expected {SCHEMA!r}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aligned", required=True, help="manifest from --group aligned")
    parser.add_argument("--row", required=True, help="manifest from --group row")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="how much worse 'aligned' may be and still count as a pass")
    parser.add_argument("--min-accuracy", type=float, default=0.45,
                        help="both runs must beat this, otherwise the comparison is noise")
    args = parser.parse_args(argv)

    aligned = load(Path(args.aligned))
    row = load(Path(args.row))

    problems: list[str] = []

    if aligned["group"] != "aligned":
        problems.append(f"first manifest is group={aligned['group']!r}, expected 'aligned'")
    if row["group"] != "row":
        problems.append(f"second manifest is group={row['group']!r}, expected 'row'")
    if aligned["seed"] != row["seed"]:
        problems.append(f"different seeds: aligned={aligned['seed']} row={row['seed']}")

    for key in SHARED_SETTINGS:
        left = aligned["settings"].get(key)
        right = row["settings"].get(key)
        if left != right:
            problems.append(f"settings differ on {key!r}: aligned={left!r} row={right!r}")

    if aligned["settings"].get("scale_groups_layer0") == row["settings"].get("scale_groups_layer0"):
        problems.append("both runs used the same number of scale groups — the grouping did not change")

    for name, payload in (("aligned", aligned), ("row", row)):
        count = payload["result"]["count"]
        correct = payload["result"]["correct"]
        if count <= 0:
            problems.append(f"{name}: empty test set")
        elif not 0 <= correct <= count:
            problems.append(f"{name}: correct={correct} outside [0, {count}]")

    aligned_acc = aligned["result"]["accuracy"]
    row_acc = row["result"]["accuracy"]

    for name, accuracy in (("aligned", aligned_acc), ("row", row_acc)):
        if accuracy < args.min_accuracy:
            problems.append(
                f"{name}: accuracy {accuracy:.4f} < {args.min_accuracy:.2f} — too weak to compare"
            )

    delta_pp = (aligned_acc - row_acc) * 100
    if delta_pp < -args.tolerance * 100:
        problems.append(
            f"aligned is {abs(delta_pp):.2f} pp *worse* than row "
            f"(tolerance {args.tolerance * 100:.2f} pp) — the mechanism did not reproduce"
        )

    print(
        f"aligned {aligned['result']['correct']}/{aligned['result']['count']} = {aligned_acc:.4f}"
        f"  row {row['result']['correct']}/{row['result']['count']} = {row_acc:.4f}"
        f"  delta {delta_pp:+.2f} pp"
    )
    print(
        f"groups layer0: aligned={aligned['settings'].get('scale_groups_layer0')} "
        f"row={row['settings'].get('scale_groups_layer0')} · "
        f"zero-fraction layer0: aligned={aligned['quantisation']['zero_fraction_layer0']:.3f} "
        f"row={row['quantisation']['zero_fraction_layer0']:.3f}"
    )

    if problems:
        print("\nFAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

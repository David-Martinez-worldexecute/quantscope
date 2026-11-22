"""Run one Forward-Forward + ternary-weight configuration and write a manifest.

The manifest is the point of this script: it records what was trained, on which
bytes, with which grouping, and what came out — so a later reader can compare two
runs without trusting the prose around them.

    python run.py --group aligned --train 2000 --test 500 --epochs 1 --data runs/mnist
    python run.py --group row     --train 2000 --test 500 --epochs 1 --data runs/mnist

Both commands above are what the CI smoke job runs; only ``--group`` differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import data as data_mod
import ff as ff_mod
import quant as quant_mod

SCHEMA = "quantscope-ff-mnist-v1"
GROUPS = ("row", "aligned")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a two-layer Forward-Forward net on MNIST with ternary weights."
    )
    parser.add_argument("--group", choices=GROUPS, required=True,
                        help="row = one scale per output row; aligned = one per input group")
    parser.add_argument("--data", default="runs/mnist", help="directory with the four idx files")
    parser.add_argument("--download", action="store_true", help="fetch the idx files into --data")
    parser.add_argument("--train", type=int, default=2000, help="training images to sample")
    parser.add_argument("--test", type=int, default=500, help="test images to sample")
    parser.add_argument("--epochs", type=int, default=1, help="epochs per layer")
    parser.add_argument("--hidden", type=int, default=500, help="width of each layer")
    parser.add_argument("--layers", type=int, default=2, help="number of layers")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; 0 leaves the default")
    parser.add_argument("--out", default=None, help="manifest path; default runs/<group>-<seed>.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.data)
    if args.download:
        base = data_mod.download(base)

    payload = data_mod.load(base, args.train, args.test, seed=args.seed)
    x_train = ff_mod.normalise(torch.from_numpy(payload["x_train"]))
    y_train = torch.from_numpy(payload["y_train"]).long()
    x_test = ff_mod.normalise(torch.from_numpy(payload["x_test"]))
    y_test = torch.from_numpy(payload["y_test"]).long()

    spans = quant_mod.spans_for(args.group)
    model = ff_mod.FFNet(hidden=args.hidden, n_layers=args.layers, spans=spans)

    print(
        f"[run] group={args.group} train={len(x_train)} test={len(x_test)} "
        f"epochs={args.epochs} hidden={args.hidden} layers={args.layers} seed={args.seed}"
    )
    losses = ff_mod.train_ff(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        theta=args.theta,
        batch_size=args.batch,
        seed=args.seed,
    )

    correct, per_class = ff_mod.evaluate(model, x_test, y_test)
    count = len(x_test)
    accuracy = correct / count

    layer0 = model.layers[0]
    spans_layer0 = layer0.spans
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.time() - started, 2),
        "group": args.group,
        "seed": args.seed,
        "settings": {
            "train": int(len(x_train)),
            "test": int(count),
            "epochs_per_layer": int(args.epochs),
            "hidden": int(args.hidden),
            "n_layers": int(args.layers),
            "in_dim": ff_mod.IN_DIM,
            "pixels": ff_mod.PIXELS,
            "labels": ff_mod.LABELS,
            "lr": float(args.lr),
            "theta": float(args.theta),
            "batch_size": int(args.batch),
            "scale_groups_layer0": 1 if spans_layer0 is None else len(spans_layer0),
            "scale_spans_layer0": None if spans_layer0 is None else [list(s) for s in spans_layer0],
        },
        "result": {
            "correct": correct,
            "count": count,
            "accuracy": accuracy,
            "per_class_correct": per_class,
            "mean_loss_per_layer": [[round(v, 6) for v in layer] for layer in losses],
        },
        "quantisation": {
            "zero_fraction_layer0": quant_mod.zero_fraction(layer0.weight, spans_layer0),
            "zero_fraction_layer1": quant_mod.zero_fraction(model.layers[1].weight,
                                                           model.layers[1].spans),
        },
        "data": {
            "source": payload["source"],
            "train_labels_sha256": hashlib.sha256(payload["y_train"].tobytes()).hexdigest(),
            "test_labels_sha256": hashlib.sha256(payload["y_test"].tobytes()).hexdigest(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    out = Path(args.out) if args.out else Path("runs") / f"{args.group}-s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print(
        f"[run] {args.group}: {correct}/{count} = {accuracy:.4f} "
        f"zero_fraction_layer0={manifest['quantisation']['zero_fraction_layer0']:.3f} "
        f"groups={manifest['settings']['scale_groups_layer0']} "
        f"{manifest['seconds']}s -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

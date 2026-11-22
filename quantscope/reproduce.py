# -*- coding: utf-8 -*-
"""Recompute every published headline number from the numbers stored beside it.

Why this exists: the docs quote things like "the aligned grouping recovers 11.86 of
the 11.97 points". Those are arithmetic on block values that already live in
`docs/evidence/ff_evidence.json`. If a headline and its blocks ever drift apart,
the prose keeps sounding confident while the claim is no longer true. This module
turns that drift into a failure.

What it checks
--------------
1. `precision_matrix.summary[mode].mean_accuracy` equals the mean of that mode's
   per-seed cells (`correct / count`), and each `cell.accuracy` equals
   `cell.correct / cell.count`.
2. Every block that stores `correct` also stores an `accuracy_mean` equal to the
   mean of `correct / count` (CIFAR, mechanism, group-scale, layer-1 runs).
3. All 21 `headline_pp` values equal the arithmetic stated in `FORMULAS`.

What it does not check
----------------------
It cannot know whether the experiment was worth running, whether the seeds are
enough, or whether the numbers were produced by the code in this repository.
It only proves that the published headline is the same number as the published
blocks, to within the rounding used when those files were written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "quantscope-reproduce-v1"

# Headline values are written rounded to six decimals, so a recomputation may
# legitimately differ in the last digit. 5e-6 accepts that and nothing larger.
TOL_PP = 5e-6
TOL_ACC = 1e-9

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PACKAGE_ROOT / "docs" / "evidence" / "ff_evidence.json"
EVIDENCE_DIR = PACKAGE_ROOT / "docs" / "evidence"

# Two bundle shapes live in docs/evidence/. The file name says which model family the
# numbers came from; the schema says how to recompute them. Adding a third bundle means
# adding one branch here, not teaching the CLI a new flag.
FF_SCHEMA = "ff-evidence-v2"
REAL_MODEL_SCHEMA = "quantscope-real-model-v1"
REAL_MODEL_MODES = ("baseline", "int8", "int4", "ternary")


class ReproduceError(ValueError):
    """Raised when a published number no longer follows from the blocks beside it."""


def _mean(values):
    values = list(values)
    if not values:
        raise ReproduceError("cannot average an empty list")
    return sum(values) / len(values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(checks, check_id, stored, recomputed, tolerance, note=""):
    delta = None if stored is None else recomputed - stored
    ok = delta is not None and abs(delta) <= tolerance
    checks.append({
        "id": check_id,
        "stored": stored,
        "recomputed": recomputed,
        "delta": delta,
        "tolerance": tolerance,
        "ok": ok,
        "note": note,
    })


def _summary_means(precision_matrix, checks) -> dict:
    """Mean accuracy per mode, recomputed from the per-seed cells.

    A stored cell accuracy that disagrees with its own ``correct/count`` becomes a
    failing check rather than an exception: the caller should see every problem in one
    pass, not the first one only. Missing or empty counts are still structural and raise.
    """
    by_mode = {}
    for cell in precision_matrix["cells"]:
        if cell["count"] <= 0:
            raise ReproduceError(f"{cell['mode']} seed {cell['seed']} has count={cell['count']}")
        expected_accuracy = cell["correct"] / cell["count"]
        by_mode.setdefault(cell["mode"], []).append(expected_accuracy)
        _check(checks, f"cell.{cell['mode']}.seed{cell['seed']}.accuracy",
               cell["accuracy"], expected_accuracy, TOL_ACC,
               note="stored accuracy must equal correct/count")
    return {mode: _mean(values) for mode, values in by_mode.items()}


def _block_mean(block, count: int) -> float:
    corrects = block["correct"]
    if not corrects:
        raise ReproduceError("block has no correct counts")
    return _mean(c / count for c in corrects)


def _check_ff(payload: dict, checks: list[dict]) -> None:
    """Every headline in `ff_evidence.json`, recomputed from the blocks stored beside it."""
    for required in ("headline_pp", "precision_matrix", "mechanism", "group_scale",
                     "cifar", "cifar_full", "group_two_layers", "depth"):
        if required not in payload:
            raise ReproduceError(f"evidence bundle is missing '{required}'")

    pm = payload["precision_matrix"]
    summary = pm["summary"]
    test_count = pm["settings"]["test_count"]

    # 1. per-mode means from per-seed cells
    cell_means = _summary_means(pm, checks)
    for mode, block in summary.items():
        if mode not in cell_means:
            raise ReproduceError(f"summary lists '{mode}' but no cells have that mode")
        _check(checks, f"summary.{mode}.mean_accuracy", block["mean_accuracy"],
               cell_means[mode], TOL_ACC)

    fp32 = summary["fp32"]["mean_accuracy"]
    ternary = summary["ternary"]["mean_accuracy"]
    first_fp32 = summary["first_fp32"]["mean_accuracy"]

    # 2. block-level accuracy_mean vs stored correct counts
    for block_name, block in payload["cifar"]["variants"].items():
        _check(checks, f"cifar.{block_name}.accuracy_mean", block["accuracy_mean"],
               _block_mean(block, payload["cifar"]["settings"]["test_size"]), TOL_ACC)
    for block_name, block in payload["cifar_full"]["variants"].items():
        _check(checks, f"cifar_full.{block_name}.accuracy_mean", block["accuracy_mean"],
               _block_mean(block, payload["cifar_full"]["settings"]["test_size"]), TOL_ACC)
    for block_name in ("two_bit_layer0", "two_bit_both", "label_at_second_layer",
                       "blockwise_patch", "blockwise_in_training"):
        block = payload["mechanism"][block_name]
        _check(checks, f"mechanism.{block_name}.accuracy_mean", block["accuracy_mean"],
               _block_mean(block, test_count), TOL_ACC)
    for block_name, block in payload["group_scale"]["variants"].items():
        _check(checks, f"group_scale.{block_name}.accuracy_mean", block["accuracy_mean"],
               _block_mean(block, test_count), TOL_ACC)
    for run_id, run in payload["group_two_layers"]["runs"].items():
        for block_name, block in run["variants"].items():
            _check(checks, f"group_two_layers.{run_id}.{block_name}.accuracy_mean",
                   block["accuracy_mean"], _block_mean(block, test_count), TOL_ACC)

    # 3. every headline number, recomputed from the blocks
    mech = payload["mechanism"]
    gs = payload["group_scale"]["variants"]
    cf = payload["cifar"]["variants"]
    cff = payload["cifar_full"]["variants"]
    runs = payload["group_two_layers"]["runs"]
    finished = [run for run in runs.values() if run["status"] == "complete"]
    if not finished:
        raise ReproduceError("no completed layer-1 run to recompute from")
    tl = finished[0]["variants"]

    blockwise_recovery = (mech["blockwise_patch"]["accuracy_mean"] - ternary) * 100.0
    fp32_to_ternary = (ternary - fp32) * 100.0
    cifar_gap = (cf["ternary_row"]["accuracy_mean"] - cf["fp32"]["accuracy_mean"]) * 100.0
    cifar_recovery = (cf["aligned_g11"]["accuracy_mean"] - cf["ternary_row"]["accuracy_mean"]) * 100.0
    full_gap = (cff["ternary_row"]["accuracy_mean"] - cff["fp32"]["accuracy_mean"]) * 100.0
    full_recovery = (cff["aligned_g11"]["accuracy_mean"] - cff["ternary_row"]["accuracy_mean"]) * 100.0

    recomputed = {
        "fp32_to_ternary": fp32_to_ternary,
        "gap_left_vs_first_fp32":
            (mech["blockwise_in_training"]["accuracy_mean"] - first_fp32) * 100.0,
        "blockwise_recovery": blockwise_recovery,
        "blockwise_share_of_gap": blockwise_recovery / abs(fp32_to_ternary),
        "two_bit_layer0_recovery": (mech["two_bit_layer0"]["accuracy_mean"] - ternary) * 100.0,
        "two_bit_both_recovery": (mech["two_bit_both"]["accuracy_mean"] - ternary) * 100.0,
        "label_at_second_vs_ternary":
            (mech["label_at_second_layer"]["accuracy_mean"] - ternary) * 100.0,
        "group_scale_aligned_vs_ternary":
            (gs["aligned_g11"]["accuracy_mean"] - ternary) * 100.0,
        "group_scale_aligned_vs_unaligned":
            (gs["aligned_g11"]["accuracy_mean"] - gs["unaligned_g11"]["accuracy_mean"]) * 100.0,
        "group_scale_aligned_g14_vs_g11":
            (gs["aligned_g14"]["accuracy_mean"] - gs["aligned_g11"]["accuracy_mean"]) * 100.0,
        "cifar_fp32_to_ternary": cifar_gap,
        "cifar_aligned_recovery": cifar_recovery,
        "cifar_aligned_share_of_gap": cifar_recovery / abs(cifar_gap),
        "cifar_unaligned_recovery":
            (cf["unaligned_g11"]["accuracy_mean"] - cf["ternary_row"]["accuracy_mean"]) * 100.0,
        "cifar_aligned_vs_unaligned":
            (cf["aligned_g11"]["accuracy_mean"] - cf["unaligned_g11"]["accuracy_mean"]) * 100.0,
        "cifar_full_fp32_to_ternary": full_gap,
        "cifar_full_aligned_recovery": full_recovery,
        "cifar_full_aligned_share_of_gap": full_recovery / abs(full_gap),
        "cifar_full_unaligned_recovery":
            (cff["unaligned_g11"]["accuracy_mean"] - cff["ternary_row"]["accuracy_mean"]) * 100.0,
        "layer1_g10_vs_layer0_aligned":
            (tl["l1_g10"]["accuracy_mean"] - gs["aligned_g11"]["accuracy_mean"]) * 100.0,
        "layer1_only_vs_ternary": (tl["l1_g50_only"]["accuracy_mean"] - ternary) * 100.0,
    }

    stored_headlines = payload["headline_pp"]
    missing = sorted(set(stored_headlines) - set(recomputed))
    extra = sorted(set(recomputed) - set(stored_headlines))
    if missing or extra:
        raise ReproduceError(
            f"headline_pp and the recomputable set disagree: missing={missing} extra={extra}")
    for key in sorted(stored_headlines):
        _check(checks, f"headline_pp.{key}", stored_headlines[key], recomputed[key], TOL_PP)

    # 4. the depth table's deltas, as a cross-check on the same identity
    for depth, block in payload["depth"]["summary"].items():
        _check(checks, f"depth.{depth}.delta_pp", block["delta_pp"],
               (block["ternary_mean_accuracy"] - block["fp32_mean_accuracy"]) * 100.0, 1e-5)

def _check_real_model(payload: dict, checks: list[dict]) -> None:
    """The 171-question real-model run: four modes, each checked against its own counts.

    Everything here is a ratio of two integers stored in the same file, so the checks
    are exact. The strongest one is `accuracy_from_baseline`: baseline's correct count,
    minus the questions this mode got wrong that baseline got right, plus the ones it
    gained, has to land on the accuracy the page prints.
    """
    for required in ("model", "dataset", "run", "modes", "protocol", "limits"):
        if required not in payload:
            raise ReproduceError(f"real-model bundle is missing '{required}'")
    count = payload["dataset"]["count"]
    if count <= 0:
        raise ReproduceError(f"dataset count is {count}")
    modes = payload["modes"]
    if sorted(modes) != sorted(REAL_MODEL_MODES):
        raise ReproduceError(f"expected modes {sorted(REAL_MODEL_MODES)}, got {sorted(modes)}")
    if payload["model"]["file_count"] != len(payload["model"]["file_sha256"]):
        raise ReproduceError("model.file_count disagrees with the number of stored hashes")

    for mode in REAL_MODEL_MODES:
        block = modes[mode]
        if block["count"] != count:
            raise ReproduceError(f"{mode}: {block['count']} questions, dataset says {count}")
        if not 0 <= block["correct"] <= count:
            raise ReproduceError(f"{mode}: correct={block['correct']} outside 0..{count}")
        _check(checks, f"real_model.{mode}.correct", block["correct"],
               int(round(block["accuracy"] * count)), 0,
               note="the stored correct count must be the one the stored accuracy came from")
        _check(checks, f"real_model.{mode}.accuracy", block["accuracy"],
               block["correct"] / count, TOL_ACC,
               note="stored accuracy must equal correct/count")

    baseline = modes["baseline"]
    for mode in REAL_MODEL_MODES:
        pairs = modes[mode]["pairs_vs_baseline"]
        if mode == "baseline":
            if pairs is not None:
                raise ReproduceError("baseline must not carry pairs_vs_baseline")
            continue
        if not pairs:
            raise ReproduceError(f"{mode}: missing pairs_vs_baseline")
        total = pairs["n00"] + pairs["n01"] + pairs["n10"] + pairs["n11"]
        if total != count:
            raise ReproduceError(f"{mode}: pairs sum to {total}, expected {count}")
        _check(checks, f"real_model.{mode}.delta_pp", pairs["delta_pp"],
               100.0 * (pairs["n01"] - pairs["n10"]) / count, TOL_PP,
               note="delta in points = 100*(gained - lost)/count")
        _check(checks, f"real_model.{mode}.accuracy_from_baseline",
               modes[mode]["accuracy"],
               (baseline["correct"] - pairs["n10"] + pairs["n01"]) / count, TOL_ACC,
               note="this mode's correct = baseline's correct - lost + gained")


def _bundle_paths(evidence_path=None) -> list[Path]:
    """Which bundles to check: the one asked for, or the whole evidence directory."""
    if evidence_path:
        return [Path(evidence_path)]
    paths = [DEFAULT_EVIDENCE]
    paths += sorted(EVIDENCE_DIR.glob("real_model_*.json"))
    return paths


def _check_bundle(payload: dict, checks: list[dict], name: str) -> None:
    schema = payload.get("schema")
    if schema == FF_SCHEMA:
        _check_ff(payload, checks)
    elif schema == REAL_MODEL_SCHEMA:
        _check_real_model(payload, checks)
    else:
        raise ReproduceError(f"{name}: unknown evidence schema {schema!r}")


def reproduce(evidence_path: Path | None = None) -> dict:
    """Recompute every published number from the bundle(s) that store it.

    With no argument this walks `docs/evidence/`: the FF bundle plus every
    `real_model_*.json`. A new evidence file is covered the moment it is added, because
    the schema — not a hard-coded file list — decides how to check it.
    """
    paths = _bundle_paths(evidence_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"evidence bundle not found: {path}")

    checks: list[dict] = []
    files: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = len(checks)
        _check_bundle(payload, checks, path.name)
        files.append({
            "name": path.name,
            "schema": payload.get("schema"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "checks": len(checks) - before,
        })

    failed = [check for check in checks if not check["ok"]]
    first = files[0]
    return {
        "schema": SCHEMA,
        "evidence": first["name"],
        "evidence_bytes": first["bytes"],
        "evidence_sha256": first["sha256"],
        "files": files,
        "checks": checks,
        "counts": {"checked": len(checks), "failed": len(failed)},
        "status": "pass" if not failed else "fail",
    }


def raise_for_failures(report: dict) -> None:
    """Turn a failing report into the exception the CLI already knows how to print."""
    failed = [c for c in report["checks"] if not c["ok"]]
    if failed:
        detail = "; ".join(
            f"{c['id']} stored={c['stored']!r} recomputed={c['recomputed']!r}" for c in failed[:5])
        more = "" if len(failed) <= 5 else f" (+{len(failed) - 5} more)"
        raise ReproduceError(
            f"{len(failed)} of {len(report['checks'])} published numbers no longer follow "
            f"from the evidence bundle: {detail}{more}")

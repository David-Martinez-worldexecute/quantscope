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
import math
import random
from pathlib import Path

SCHEMA = "quantscope-reproduce-v1"

# Headline values are written rounded to six decimals, so a recomputation may
# legitimately differ in the last digit. 5e-6 accepts that and nothing larger.
TOL_PP = 5e-6
TOL_ACC = 1e-9
TOL_RATE = 1e-3

# The interval beside each accuracy is a percentile bootstrap over the per-question
# outcomes. It is recomputed here with the same seed, the same resample count and the
# same random stream, because an interval nobody can recompute is decoration.
BOOTSTRAP_SEED = 12345
BOOTSTRAP_RESAMPLES = 4000


def bootstrap_interval(values, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """Percentile 95% interval of the mean of `values`, or (None, None) if empty.

    The stream is `random.Random(seed).random()` consumed in a fixed order, one draw
    per resampled element, never `random.choice`: the docs publish the exact interval
    this produces, so the draw order is part of the published number.
    """
    count = len(values)
    if count == 0:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += values[int(rng.random() * count)]
        means.append(total / count)
    means.sort()
    edge = resamples // 40
    return means[edge], means[-edge - 1]


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PACKAGE_ROOT / "docs" / "evidence" / "ff_evidence.json"
EVIDENCE_DIR = PACKAGE_ROOT / "docs" / "evidence"

# Two bundle shapes live in docs/evidence/. The file name says which model family the
# numbers came from; the schema says how to recompute them. Adding a third bundle means
# adding one branch here, not teaching the CLI a new flag.
FF_SCHEMA = "ff-evidence-v2"
REAL_MODEL_SCHEMA = "quantscope-real-model-v1"
REAL_MODEL_MODES = ("baseline", "int8", "int4", "ternary")
BENCH_SCHEMA = "quantscope-bench-v1"
PACKED_SCHEMA = "quantscope-packed-v1"
PACKED_MODES = ("baseline", "int4", "ternary")
FOOTPRINT_SCHEMA = "quantscope-packed-footprint-v1"
EQUIVALENCE_SCHEMA = "quantscope-packing-equivalence-v1"
EQUIVALENCE_MODES = ("int4", "ternary")
ANSWER_SCHEMA = "quantscope-answer-distribution-v1"
ANSWER_MODES = ("baseline", "int8", "int4", "ternary")
ANSWER_LETTERS = ("A", "B", "C", "D")
# A format that picks one letter for almost every question is not being measured any
# more; the share is the threshold the published sentence uses to say so.
COLLAPSE_SHARE = 0.90
# The layer sweep and the real 4-bit container answer two different questions about the
# same model: one asks where the ternary collapse starts, the other what a real format
# costs in bytes and speed. Both are recomputed from the blocks stored beside them.
PREFIX_SCHEMA = "quantscope-prefix-scan-v1"
LINEAR_PER_LAYER = 7
NF4_SCHEMA = "quantscope-nf4-bench-v1"
LAYER_CLIFF_SCHEMA = "quantscope-layer-cliff-v1"
SCALING_SCHEMA = "quantscope-scaling-check-v1"
AMP_SCHEMA = "quantscope-amplification-v1"
GRANULARITY_SCHEMA = "quantscope-granularity-v1"
FAIR_SCHEMA = "quantscope-fair-remeasure-v1"
CLIFF_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BENCH_MODES = ("baseline", "int8", "int4", "ternary")


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
        outcomes = block.get("outcomes")
        if outcomes is None:
            raise ReproduceError(
                f"{mode}: missing outcomes; the published interval cannot be re-derived "
                "without the per-question 0/1 sequence"
            )
        if len(outcomes) != count:
            raise ReproduceError(f"{mode}: {len(outcomes)} outcomes for {count} questions")
        if any(value not in (0, 1) for value in outcomes):
            raise ReproduceError(f"{mode}: outcomes must be 0 or 1")
        _check(checks, f"real_model.{mode}.outcomes_sum", sum(outcomes), block["correct"], 0,
               note="the stored outcomes have to add up to the stored correct count")
        interval = block.get("ci95") or {}
        low, high = bootstrap_interval(outcomes)
        _check(checks, f"real_model.{mode}.ci95_low", interval.get("low"), low, TOL_ACC,
               note="2.5th percentile of the bootstrap means, same seed and resample count")
        _check(checks, f"real_model.{mode}.ci95_high", interval.get("high"), high, TOL_ACC,
               note="97.5th percentile of the bootstrap means, same seed and resample count")

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
        # Both outcome lists are stored in ascending question-id order, so index i is the
        # same question in both. The paired interval is the bootstrap of the per-question
        # difference, which is tighter than the difference of two separate intervals.
        base_outcomes = baseline["outcomes"]
        deltas = [
            100.0 * (value - base)
            for value, base in zip(modes[mode]["outcomes"], base_outcomes, strict=True)
        ]
        paired = modes[mode].get("paired_delta_pp_ci95") or {}
        low, high = bootstrap_interval(deltas)
        _check(checks, f"real_model.{mode}.paired_ci95_low", paired.get("low"), low, TOL_ACC,
               note="2.5th percentile of the bootstrap of per-question paired differences")
        _check(checks, f"real_model.{mode}.paired_ci95_high", paired.get("high"), high, TOL_ACC,
               note="97.5th percentile of the bootstrap of per-question paired differences")


def _check_bench(payload: dict, checks: list[dict]) -> None:
    """The cost run: four formats, one prompt, one process on one CPU.

    Timings cannot be recomputed on a different machine, so the checks are the
    identities that must hold inside the file (stored seconds against stored
    rates, a median between its own min and max) plus the one ordering that is
    true of the formats themselves: fewer bits, fewer bytes.
    """
    for required in ("model", "protocol", "modes", "limits"):
        if required not in payload:
            raise ReproduceError(f"bench bundle is missing '{required}'")
    modes = payload["modes"]
    if sorted(modes) != sorted(BENCH_MODES):
        raise ReproduceError(f"expected modes {sorted(BENCH_MODES)}, got {sorted(modes)}")
    protocol = payload["protocol"]
    if protocol["prompt_tokens"] <= 0 or protocol["max_new_tokens"] <= 0:
        raise ReproduceError(f"protocol counts are not positive: {protocol}")

    sizes = {}
    for mode in BENCH_MODES:
        block = modes[mode]
        if block["max_new_tokens"] <= 0:
            raise ReproduceError(f"{mode}: max_new_tokens is {block['max_new_tokens']}")
        if block["decode_seconds"] <= 0:
            raise ReproduceError(f"{mode}: decode_seconds is {block['decode_seconds']}")
        _check(checks, f"bench.{mode}.decode_tokens_per_second",
               block["decode_tokens_per_second"],
               block["max_new_tokens"] / block["decode_seconds"], TOL_RATE,
               note="the stored rate must equal new tokens / stored decode seconds")
        median = block["prefill_ms_median"]
        low, high = block["prefill_ms_min"], block["prefill_ms_max"]
        _check(checks, f"bench.{mode}.prefill_ms_median", median,
               min(max(median, low), high), 0,
               note="the median must sit between the stored min and max")
        sizes[mode] = block["quantized_bytes"]

    if not (sizes["ternary"] < sizes["int4"] < sizes["int8"] < sizes["baseline"]):
        raise ReproduceError(
            "payload sizes are not ordered by width: " +
            ", ".join(f"{mode} {sizes[mode]}" for mode in BENCH_MODES))


def _check_packed(payload: dict, checks: list[dict]) -> None:
    """The packed run: the same model with INT4 and ternary weights really stored small.

    The point of this bundle is a memory claim, so the checks tie the stored timing
    identities together and refuse to accept a payload order that would make the
    table lie. Peak RSS cannot be recomputed from anything, so it is only required
    to be present and positive.
    """
    for required in ("model", "protocol", "modes", "limits"):
        if required not in payload:
            raise ReproduceError(f"packed bundle is missing '{required}'")
    modes = payload["modes"]
    if sorted(modes) != sorted(PACKED_MODES):
        raise ReproduceError(f"expected modes {sorted(PACKED_MODES)}, got {sorted(modes)}")
    protocol = payload["protocol"]
    if protocol.get("isolation") != "one child process per mode":
        raise ReproduceError(
            "packed bundle does not record per-mode process isolation: "
            f"{protocol.get('isolation')!r}")
    max_new_tokens = protocol["max_new_tokens"]
    if max_new_tokens <= 0:
        raise ReproduceError(f"max_new_tokens is {max_new_tokens}")

    sizes = {}
    for mode in PACKED_MODES:
        block = modes[mode]
        if block["decode_seconds"] <= 0:
            raise ReproduceError(f"{mode}: decode_seconds is {block['decode_seconds']}")
        rss = block.get("peak_rss_mib")
        if rss is None or rss <= 0:
            raise ReproduceError(f"{mode}: peak_rss_mib is {rss!r}")
        _check(checks, f"packed.{mode}.decode_tokens_per_second",
               block["decode_tokens_per_second"],
               max_new_tokens / block["decode_seconds"], TOL_RATE,
               note="the stored rate must equal new tokens / stored decode seconds")
        median = block["prefill_ms_median"]
        low, high = block["prefill_ms_min"], block["prefill_ms_max"]
        _check(checks, f"packed.{mode}.prefill_ms_median", median,
               min(max(median, low), high), 0,
               note="the median must sit between the stored min and max")
        sizes[mode] = block["payload_bytes"]

    if not (sizes["ternary"] < sizes["int4"] < sizes["baseline"]):
        raise ReproduceError(
            "packed payload sizes are not ordered by width: " +
            ", ".join(f"{mode} {sizes[mode]}" for mode in PACKED_MODES))


def _check_footprint(payload: dict, checks: list[dict]) -> None:
    """What the weights occupy when the checkpoint is streamed instead of loaded.

    This is the bundle behind the sentence "the low-bit formats really are smaller": it
    never builds the model, so the number it reports is the stored weights rather than
    the float32 copy plus the packed one. Two identities are recomputable from the
    stored bytes (held_mib and the tensor count), and the ordering is the claim itself,
    so a payload that reverses it would make the page lie.
    """
    for required in ("model", "protocol", "modes", "limits"):
        if required not in payload:
            raise ReproduceError(f"footprint bundle is missing '{required}'")
    modes = payload["modes"]
    if sorted(modes) != sorted(PACKED_MODES):
        raise ReproduceError(f"expected modes {sorted(PACKED_MODES)}, got {sorted(modes)}")

    held = {}
    for mode in PACKED_MODES:
        block = modes[mode]
        held[mode] = block["held_bytes"]
        _check(checks, f"footprint.{mode}.held_mib", block["held_mib"],
               block["held_bytes"] / (1024 * 1024), 0.001,
               note="MiB must be the stored bytes divided by 1024*1024")
        expected_tensors = block["matrices"] if mode == "baseline" else block["matrices"] * 2
        if block["held_tensors"] != expected_tensors:
            raise ReproduceError(
                f"{mode}: {block['held_tensors']} stored tensors for {block['matrices']} "
                "matrices; each packed matrix stores codes plus scales")
        rss = block.get("rss_settled_mib")
        if rss is None or rss <= 0:
            raise ReproduceError(f"{mode}: rss_settled_mib is {rss!r}")
        if block["load_seconds"] <= 0:
            raise ReproduceError(f"{mode}: load_seconds is {block['load_seconds']}")

    if not (held["ternary"] < held["int4"] < held["baseline"]):
        raise ReproduceError(
            "held weight bytes are not ordered by width: " +
            ", ".join(f"{mode} {held[mode]}" for mode in PACKED_MODES))
    settled = {mode: modes[mode]["rss_settled_mib"] for mode in PACKED_MODES}
    if not (settled["ternary"] < settled["int4"] < settled["baseline"]):
        raise ReproduceError(f"settled RSS does not fall with the held bytes: {settled}")


def _check_equivalence(payload: dict, checks: list[dict]) -> None:
    """Packing must reproduce simulated quantisation bit for bit, not approximately.

    The bundle compares the packed-then-unpacked weights against the repository's own
    quantiser on the same checkpoint, and then compares the answers both paths give on
    the same questions. Both halves are recomputable: the tensor difference must be
    exactly zero, and the answer lists must agree element by element.
    """
    for required in ("model", "protocol", "modes", "limits", "discrepancies"):
        if required not in payload:
            raise ReproduceError(f"equivalence bundle is missing '{required}'")
    modes = payload["modes"]
    if sorted(modes) != sorted(EQUIVALENCE_MODES):
        raise ReproduceError(
            f"expected modes {sorted(EQUIVALENCE_MODES)}, got {sorted(modes)}")
    if payload["matrices_compared"] <= 0:
        raise ReproduceError("equivalence bundle compared no matrices")
    if payload["discrepancies"]:
        raise ReproduceError(
            f"{len(payload['discrepancies'])} matrices differ between packed and "
            "simulated weights; the receipt does not support 'same numbers'")
    _check(checks, "equivalence.worst_abs_diff", payload["worst_abs_diff"], 0.0, 0.0,
           note="packed weights must equal the simulated weights element for element")

    questions = payload["protocol"]["questions"]
    for mode in EQUIVALENCE_MODES:
        block = modes[mode]
        simulated = block["simulated_answers"]
        packed = block["packed_answers"]
        if not (len(simulated) == len(packed) == block["count"] == questions):
            raise ReproduceError(
                f"{mode}: {len(simulated)}/{len(packed)} answers, count "
                f"{block['count']}, protocol says {questions}")
        agree = sum(1 for want, got in zip(simulated, packed) if want == got)
        _check(checks, f"equivalence.{mode}.answers_identical_to_simulated",
               block["answers_identical_to_simulated"], agree, 0)
        _check(checks, f"equivalence.{mode}.max_abs_logit_diff",
               block["max_abs_logit_diff"], 0.0, 0.0,
               note="identical weights read through the same prompt must agree exactly")


def _check_answer_distribution(payload: dict, checks: list[dict]) -> None:
    """What letter each format chose, and whether that is still a model at all.

    A format that answers the same letter for 99.8% of the questions is not degraded by
    three points; it has stopped depending on the question. Every share here is
    recomputed from the stored pick counts, and the constant-output accuracy is
    recomputed from the stored gold counts, so the collapse claim cannot drift.
    """
    for required in ("modes", "questions", "limits", "source_run"):
        if required not in payload:
            raise ReproduceError(f"answer bundle is missing '{required}'")
    modes = payload["modes"]
    if sorted(modes) != sorted(ANSWER_MODES):
        raise ReproduceError(
            f"expected modes {sorted(ANSWER_MODES)}, got {sorted(modes)}")
    total = payload["questions"]
    gold_seen = None

    for mode in ANSWER_MODES:
        block = modes[mode]
        if block["count"] != total:
            raise ReproduceError(f"{mode}: count {block['count']} but {total} questions")
        picks = block["picks"]
        share = block["share"]
        if sorted(picks) != sorted(ANSWER_LETTERS):
            raise ReproduceError(f"{mode}: picks cover {sorted(picks)}")
        if sum(picks.values()) != total:
            raise ReproduceError(
                f"{mode}: picks sum to {sum(picks.values())}, not {total}")
        for letter in ANSWER_LETTERS:
            _check(checks, f"answers.{mode}.share.{letter}", share[letter],
                   picks[letter] / total, 1e-5,
                   note="each share must be its pick count over the total")
        expected_pick = max(picks, key=lambda letter: picks[letter])
        if block["most_common_pick"] != expected_pick:
            raise ReproduceError(
                f"{mode}: most_common_pick is {block['most_common_pick']!r}, "
                f"the counts say {expected_pick!r}")
        _check(checks, f"answers.{mode}.most_common_share",
               block["most_common_share"], picks[expected_pick] / total, 1e-5)

        gold = block["gold"]
        if sorted(gold) != sorted(ANSWER_LETTERS):
            raise ReproduceError(f"{mode}: gold covers {sorted(gold)}")
        if sum(gold.values()) != total:
            raise ReproduceError(f"{mode}: gold sums to {sum(gold.values())}, not {total}")
        if gold_seen is None:
            gold_seen = gold
        elif gold != gold_seen:
            raise ReproduceError(
                f"{mode}: the gold labels changed between modes; every format "
                "must be scored on the same questions")
        modal = max(gold, key=lambda letter: gold[letter])
        if block["modal_gold"] != modal:
            raise ReproduceError(
                f"{mode}: modal_gold is {block['modal_gold']!r}, the counts say {modal!r}")
        _check(checks, f"answers.{mode}.constant_output_accuracy",
               block["constant_output_accuracy"], gold[modal] / total, 1e-5,
               note="answering the most common gold letter every time is the floor")

    collapsed = sorted(mode for mode in ANSWER_MODES
                       if modes[mode]["most_common_share"] >= COLLAPSE_SHARE)
    if collapsed != ["ternary"]:
        raise ReproduceError(
            f"the collapsed set is {collapsed}, not ['ternary']; the page says one "
            "format stopped depending on the question and the counts must agree")


def _check_prefix_scan(payload: dict, checks: list[dict]) -> None:
    """The one-layer-at-a-time sweep: every count, share and tensor total must agree."""
    for key in ("model", "protocol", "conditions", "limits", "versions"):
        if key not in payload:
            raise ReproduceError(f"prefix scan: missing {key!r}")
    protocol = payload["protocol"]
    conditions = payload["conditions"]
    if not conditions:
        raise ReproduceError("prefix scan: no conditions")
    collapse_share = protocol.get("collapse_share", COLLAPSE_SHARE)
    if not 0 < collapse_share <= 1:
        raise ReproduceError(f"prefix scan: collapse_share {collapse_share!r} is not a share")
    total = max(block["layers_quantised"] for block in conditions)
    seen: list[int] = []
    for block in conditions:
        layers = block["layers_quantised"]
        if layers in seen:
            raise ReproduceError(f"prefix scan: {layers} quantised layers appear twice")
        seen.append(layers)
        if block["float32_layers"] != total - layers:
            raise ReproduceError(
                f"prefix scan: {layers} quantised layers are listed with "
                f"{block['float32_layers']} float32 ones, but the sweep has {total} layers")
        count = block["count"]
        if count <= 0:
            raise ReproduceError(f"prefix scan: {layers} layers has count={count}")
        histogram = block["histogram"]
        if sorted(histogram) != sorted(ANSWER_LETTERS):
            raise ReproduceError(f"prefix scan: {layers} layers counts {sorted(histogram)}")
        if sum(histogram.values()) != count:
            raise ReproduceError(
                f"prefix scan: {layers} layers: the counts sum to {sum(histogram.values())}, "
                f"not {count}")
        if block["correct"] > count:
            raise ReproduceError(
                f"prefix scan: {layers} layers answers {block['correct']} of {count}")
        modal = max(histogram, key=lambda letter: histogram[letter])
        if block["most_common"] != modal:
            raise ReproduceError(
                f"prefix scan: {layers} layers: most_common is {block['most_common']!r}, "
                f"the counts say {modal!r}")
        _check(checks, f"prefix_scan.{layers:02d}.most_common_share",
               block["most_common_share"], histogram[modal] / count, 1e-5)
        _check(checks, f"prefix_scan.{layers:02d}.linear_layers",
               block["linear_layers"], LINEAR_PER_LAYER * layers, 0,
               note="each quantised layer holds seven Linear weights")

    shares = {block["layers_quantised"]: block["most_common_share"] for block in conditions}
    collapsed = sorted(layers for layers, share in shares.items() if share >= collapse_share)
    if not collapsed:
        raise ReproduceError(
            "prefix scan: no prefix collapsed, so the page's mechanism claim has no support")
    _check(checks, "prefix_scan.first_collapse", payload["first_collapse"], collapsed[0], 0)
    if payload["collapse_lengths"] != collapsed:
        raise ReproduceError(
            f"prefix scan: collapse_lengths is {payload['collapse_lengths']}, "
            f"the shares say {collapsed}")
    letters = [next(block["most_common"] for block in conditions
                    if block["layers_quantised"] == layers) for layers in collapsed]
    if payload["collapse_letters"] != letters:
        raise ReproduceError(
            f"prefix scan: collapse_letters is {payload['collapse_letters']}, "
            f"the counts say {letters}")
    contiguous = collapsed == list(range(collapsed[0], collapsed[-1] + 1))
    if bool(payload["collapse_contiguous"]) is not contiguous:
        raise ReproduceError(
            f"prefix scan: collapse_contiguous is {payload['collapse_contiguous']}, "
            f"but the collapsed prefixes are {collapsed}")


def _check_nf4(payload: dict, checks: list[dict]) -> None:
    """A real 4-bit container on the GPU: bytes, answers and the memory they landed in."""
    for key in ("float32", "nf4", "protocol", "limits"):
        if key not in payload:
            raise ReproduceError(f"nf4 bench: missing {key!r}")
    stage = payload["float32"]
    packed = payload["nf4"]
    for name, block in (("float32", stage), ("nf4", packed)):
        count = block["count"]
        histogram = block["histogram"]
        if sorted(histogram) != sorted(ANSWER_LETTERS):
            raise ReproduceError(f"nf4 bench: {name}: counts {sorted(histogram)}")
        if sum(histogram.values()) != count:
            raise ReproduceError(
                f"nf4 bench: {name}: the counts sum to {sum(histogram.values())}, not {count}")
        if block["correct"] > count:
            raise ReproduceError(f"nf4 bench: {name}: answers {block['correct']} of {count}")
        _check(checks, f"nf4_bench.{name}.stored_bytes",
               block["stored_bytes"], sum(block["stored_by_dtype"].values()), 0,
               note="every stored tensor is counted once, by dtype")
        if block["gpu_memory_after_load"] < block["stored_bytes"]:
            raise ReproduceError(
                f"nf4 bench: {name}: loading reported less memory than the weights themselves")
    want = stage["count"]
    if len(packed["answers"]) != want or len(packed["float32_answers"]) != want:
        raise ReproduceError(
            "nf4 bench: the answer lists do not cover the questions that were answered")
    agree = sum(1 for before, after in zip(packed["float32_answers"], packed["answers"])
                if before == after)
    _check(checks, "nf4_bench.nf4.answers_identical_to_float32",
           packed["answers_identical_to_float32"], float(agree), 0)
    if not packed["stored_bytes"] < stage["stored_bytes"]:
        raise ReproduceError(
            "nf4 bench: the packed model stores "
            f"{packed['stored_bytes']} bytes, the float32 one {stage['stored_bytes']}; "
            "the page says the real 4-bit container is smaller")


def _check_layer_cliff(payload: dict, checks: list[dict]) -> None:
    """The 18-layer cliff: which weight class, put back to float32, breaks the collapse."""
    for field in ("model", "protocol", "conditions", "matrix_stats", "limits", "anchor"):
        if field not in payload:
            raise ReproduceError(f"layer cliff: missing {field!r}")
    protocol = payload["protocol"]
    depth = protocol.get("quantised_layers")
    collapse = protocol.get("collapse_share")
    if not isinstance(depth, int) or depth <= 0:
        raise ReproduceError(f"layer cliff: quantised_layers must be a positive integer, got {depth!r}")
    if not isinstance(collapse, (int, float)) or not 0 < collapse <= 1:
        raise ReproduceError(f"layer cliff: collapse_share must be in (0, 1], got {collapse!r}")
    questions = protocol.get("questions")
    if not isinstance(questions, int) or questions <= 0:
        raise ReproduceError(f"layer cliff: questions must be a positive integer, got {questions!r}")

    conditions = payload["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ReproduceError("layer cliff: conditions must be a non-empty list")
    names = [item.get("condition") for item in conditions]
    if len(set(names)) != len(names):
        raise ReproduceError(f"layer cliff: duplicate condition names: {names}")
    for required in ("all_quantised",):
        if required not in names:
            raise ReproduceError(f"layer cliff: the {required} condition is missing")

    total_matrices = depth * LINEAR_PER_LAYER
    for item in conditions:
        name = item["condition"]
        quantised = item.get("quantised_matrices")
        kept = item.get("float32_matrices")
        if not isinstance(quantised, int) or not isinstance(kept, int):
            raise ReproduceError(f"layer cliff: {name}: matrix counts must be integers")
        if quantised + kept != total_matrices:
            raise ReproduceError(
                f"layer cliff: {name} accounts for {quantised} + {kept} matrices, "
                f"but {depth} layers hold {total_matrices}")
        count = item.get("count")
        histogram = item.get("histogram")
        if not isinstance(histogram, dict) or not histogram or sum(histogram.values()) != count:
            raise ReproduceError(f"layer cliff: {name}: the histogram does not sum to {count}")
        if not isinstance(item.get("correct"), int) or item["correct"] > count:
            raise ReproduceError(f"layer cliff: {name}: {item.get('correct')!r} correct of {count}")
        most = max(histogram, key=lambda letter: histogram[letter])
        if item.get("most_common") != most:
            raise ReproduceError(
                f"layer cliff: {name}: most_common is {item.get('most_common')!r}, "
                f"the histogram says {most!r}")
        _check(checks, f"cliff.{name}.most_common_share", item["most_common_share"],
               histogram[most] / count, 1e-5)

    collapsed_row = next(item for item in conditions if item["condition"] == "all_quantised")
    if collapsed_row["most_common_share"] < collapse:
        raise ReproduceError(
            "layer cliff: the all_quantised condition is not collapsed "
            f"(share {collapsed_row['most_common_share']}), so the page's question has no subject")

    anchor = payload["anchor"]
    if anchor.get("layers_quantised") != depth:
        raise ReproduceError(
            f"layer cliff: the anchor covers {anchor.get('layers_quantised')} layers, "
            f"the protocol says {depth}")
    if (anchor["correct"], anchor["most_common"], anchor["histogram"]) != (
        collapsed_row["correct"],
        collapsed_row["most_common"],
        collapsed_row["histogram"],
    ):
        raise ReproduceError("layer cliff: the anchor row disagrees with the all_quantised condition")

    survivors = sorted(item["condition"] for item in conditions
                       if item["most_common_share"] < collapse)
    recorded = sorted(payload.get("survivors") or [])
    if recorded != survivors:
        raise ReproduceError(
            f"layer cliff: survivors {recorded} do not match the shares {survivors}")
    kept = sorted(item["condition"] for item in conditions[1:]
                  if item["most_common_share"] >= collapse)
    stated = sorted(payload.get("collapsed_keeps") or [])
    if stated != kept:
        raise ReproduceError(f"layer cliff: collapsed_keeps {stated} do not match the shares {kept}")

    stats = payload["matrix_stats"]
    if stats.get("matrices") != collapsed_row["quantised_matrices"]:
        raise ReproduceError(
            f"layer cliff: matrix_stats counts {stats.get('matrices')} matrices, "
            f"the all_quantised condition quantised {collapsed_row['quantised_matrices']}")
    by_kind = stats.get("by_kind") or {}
    if sorted(by_kind) != sorted(CLIFF_KINDS):
        raise ReproduceError(f"layer cliff: by_kind covers {sorted(by_kind)}, not the seven projections")
    for kind, bucket in by_kind.items():
        if bucket.get("matrices") != depth:
            raise ReproduceError(
                f"layer cliff: {kind}: {bucket.get('matrices')} matrices for {depth} layers")
        _check(checks, f"cliff.by_kind.{kind}.mean_zero_fraction", bucket["mean_zero_fraction"],
               bucket["zero_fraction_sum"] / bucket["matrices"], 2e-6)
        _check(checks, f"cliff.by_kind.{kind}.mean_relative_error", bucket["mean_relative_error"],
               bucket["relative_error_sum"] / bucket["matrices"], 2e-6)


def _check_scaling(payload: dict, checks: list[dict]) -> None:
    """The same sweep at sixty and at six hundred questions: does the collapse survive?

    Two things are recomputed here and nowhere else: the collapsed set of each sample, and
    the verdict that says whether a collapsed row scores anything better than printing one
    letter. The verdict is the whole point of the bundle, so it cannot be copied from the
    bundle — a stale verdict would keep claiming a mechanism after the numbers moved.
    """
    for field in ("schema", "z", "intervals", "constants", "rows", "collapsed",
                  "collapsed_vs_constants"):
        if field not in payload:
            raise ReproduceError(f"scaling check: missing {field!r}")

    rows = payload["rows"]
    if not isinstance(rows, dict) or len(rows) < 2:
        raise ReproduceError(f"scaling check: rows must cover at least two samples, got {rows!r}")
    samples = sorted(rows)
    constants = payload["constants"]
    collapsed: dict[str, list[int]] = {}
    per_sample: dict[str, dict[int, dict]] = {}

    for label in samples:
        entries = rows[label]
        if not isinstance(entries, list) or not entries:
            raise ReproduceError(f"scaling check: {label}: rows must be a non-empty list")
        sample_constants = constants.get(label)
        if not isinstance(sample_constants, dict) or sorted(sample_constants) != ["A", "B", "C", "D"]:
            raise ReproduceError(
                f"scaling check: {label}: constants must cover A/B/C/D, got {sample_constants!r}")
        for letter, value in sample_constants.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ReproduceError(f"scaling check: {label}: always-{letter} is {value!r}")
        # The four constants are the shares of the four gold letters, so they sum to one.
        _check(checks, f"scaling.{label}.constants_sum", sum(sample_constants.values()), 1.0, 1e-9)

        seen: list[int] = []
        for row in entries:
            layers = row.get("layers")
            questions = row.get("questions")
            histogram = row.get("histogram")
            if not isinstance(layers, int) or layers <= 0:
                raise ReproduceError(f"scaling check: {label}: layers must be a positive integer")
            if not isinstance(questions, int) or questions <= 0:
                raise ReproduceError(f"scaling check: {label}: {layers}: questions must be positive")
            if not isinstance(histogram, dict) or not histogram or sum(histogram.values()) != questions:
                raise ReproduceError(
                    f"scaling check: {label}: {layers}: the histogram does not sum to {questions}")
            if not isinstance(row.get("correct"), int) or row["correct"] > questions:
                raise ReproduceError(
                    f"scaling check: {label}: {layers}: {row.get('correct')!r} correct of {questions}")
            most = max(histogram, key=lambda letter: histogram[letter])
            if row.get("most_common") != most:
                raise ReproduceError(
                    f"scaling check: {label}: {layers}: most_common is {row.get('most_common')!r}, "
                    f"the histogram says {most!r}")
            interval = row.get("accuracy_ci")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ReproduceError(f"scaling check: {label}: {layers}: accuracy_ci must be [low, high]")
            _check(checks, f"scaling.{label}.{layers:02d}.accuracy",
                   row["accuracy"], row["correct"] / questions, 1e-5)
            _check(checks, f"scaling.{label}.{layers:02d}.most_common_share",
                   row["most_common_share"], histogram[most] / questions, 1e-5)
            seen.append(layers)

        if seen != sorted(set(seen)):
            raise ReproduceError(
                f"scaling check: {label}: the layers must ascend without repeats, got {seen}")
        per_sample[label] = {row["layers"]: row for row in entries}
        collapsed[label] = sorted(row["layers"] for row in entries
                                  if row["most_common_share"] >= COLLAPSE_SHARE)
        if not collapsed[label]:
            raise ReproduceError(
                f"scaling check: {label}: nothing collapsed at {COLLAPSE_SHARE}, "
                "so the bundle has no question to answer")

    stored_collapsed = payload["collapsed"]
    for label, computed in collapsed.items():
        if stored_collapsed.get(label) != computed:
            raise ReproduceError(
                f"scaling check: {label}: stored collapsed {stored_collapsed.get(label)}, "
                f"the shares say {computed}")
    same = all(collapsed[other] == collapsed[samples[0]] for other in samples[1:])
    if bool(stored_collapsed.get("same_set")) != same:
        raise ReproduceError(
            f"scaling check: stored same_set {stored_collapsed.get('same_set')!r}, recomputed {same}")

    expected: dict[tuple[str, int], tuple[str, str, float]] = {}
    for label in samples:
        for layers in collapsed[label]:
            row = per_sample[label][layers]
            letter = row["most_common"]
            constant = constants[label][letter]
            low, high = row["accuracy_ci"]
            if low <= constant <= high:
                verdict = f"not proven to differ from always-{letter}"
            elif row["accuracy"] > constant:
                verdict = f"above always-{letter}"
            else:
                verdict = f"below always-{letter}"
            expected[(label, layers)] = (letter, verdict, row["accuracy"])

    stored_verdicts = payload["collapsed_vs_constants"]
    if not isinstance(stored_verdicts, list):
        raise ReproduceError("scaling check: collapsed_vs_constants must be a list")
    stored: dict[tuple[str, int], dict] = {}
    for item in stored_verdicts:
        key = (item.get("sample"), item.get("layers"))
        if key in stored:
            raise ReproduceError(f"scaling check: the verdict table lists {key} twice")
        stored[key] = item
    if sorted(stored) != sorted(expected):
        raise ReproduceError(
            f"scaling check: the verdict table covers {sorted(stored)}, "
            f"but the collapsed rows are {sorted(expected)}")
    for key, item in stored.items():
        letter, verdict, accuracy = expected[key]
        if item.get("letter") != letter:
            raise ReproduceError(
                f"scaling check: {key[0]} questions at {key[1]} layers: letter "
                f"{item.get('letter')!r}, the histogram says {letter!r}")
        if item.get("verdict") != verdict:
            raise ReproduceError(
                f"scaling check: {key[0]} questions at {key[1]} layers: stored verdict "
                f"{item.get('verdict')!r}, recomputed {verdict!r}")
        _check(checks, f"scaling.{key[0]}.{key[1]:02d}.accuracy_vs_constant",
               item["accuracy"], accuracy, 1e-5)


def _fit_exponent(curve: list[float]) -> float:
    """Slope of ln(drift) against layer index, over the layers whose drift is measurable."""
    points = [(index, value) for index, value in enumerate(curve) if value > 1e-6]
    if len(points) < 3:
        return 0.0
    count = len(points)
    mean_x = sum(index for index, _ in points) / count
    mean_y = sum(math.log(value) for _, value in points) / count
    denominator = sum((index - mean_x) ** 2 for index, _ in points)
    if denominator == 0:
        return 0.0
    return sum((index - mean_x) * (math.log(value) - mean_y) for index, value in points) / denominator


def _check_amplification(payload: dict, checks: list[dict]) -> None:
    """Does the drift grow like independent per-layer damage, or like an amplified chain?"""
    for field in ("model", "protocol", "layers", "depth_axis", "prefixes", "single_layers",
                  "first_drift_crossing", "limits"):
        if field not in payload:
            raise ReproduceError(f"amplification: missing {field!r}")
    protocol = payload["protocol"]
    questions = protocol.get("questions")
    if not isinstance(questions, int) or questions <= 0:
        raise ReproduceError(f"amplification: questions must be a positive integer, got {questions!r}")
    layers = payload["layers"]
    if not isinstance(layers, int) or layers <= 0:
        raise ReproduceError(f"amplification: layers must be a positive integer, got {layers!r}")
    axis = payload["depth_axis"]
    if not isinstance(axis, int) or axis <= 0:
        raise ReproduceError(f"amplification: depth_axis must be a positive integer, got {axis!r}")

    singles = payload["single_layers"]
    if not isinstance(singles, list) or len(singles) != layers:
        raise ReproduceError(
            f"amplification: single_layers holds {len(singles) if isinstance(singles, list) else singles!r} rows, "
            f"one per layer was expected ({layers})")
    if [item.get("layer_index") for item in singles] != list(range(layers)):
        raise ReproduceError(f"amplification: single layer indices are not 0..{layers - 1}")
    for item in singles:
        if item.get("own_layer_drift", -1) < 0 or item.get("final_hidden_drift", -1) < 0:
            raise ReproduceError(f"amplification: layer {item['layer_index']} reports a negative drift")

    prefixes = payload["prefixes"]
    if not isinstance(prefixes, list) or not prefixes:
        raise ReproduceError("amplification: prefixes must be a non-empty list")
    depths = [block.get("layers_quantised") for block in prefixes]
    if depths != sorted(set(depths)):
        raise ReproduceError(f"amplification: prefix depths do not rise strictly: {depths}")
    for depth in depths:
        if not isinstance(depth, int) or depth <= 0 or depth > layers:
            raise ReproduceError(f"amplification: prefix depth {depth!r} is outside 1..{layers}")

    for block in prefixes:
        depth = block["layers_quantised"]
        if block.get("quantised_matrices") != depth * LINEAR_PER_LAYER:
            raise ReproduceError(
                f"amplification: {depth} layers report {block.get('quantised_matrices')!r} matrices, "
                f"not {depth * LINEAR_PER_LAYER}")
        curve = block.get("hidden_drift")
        if not isinstance(curve, list) or len(curve) != axis:
            raise ReproduceError(
                f"amplification: the drift curve at {depth} layers has "
                f"{len(curve) if isinstance(curve, list) else curve!r} points, not {axis}")
        if any(value < 0 for value in curve):
            raise ReproduceError(f"amplification: the drift curve at {depth} layers goes negative")
        _check(checks, f"amplification.prefix.{depth:02d}.final_hidden_drift",
               block["final_hidden_drift"], curve[-1], 1e-6)
        _check(checks, f"amplification.prefix.{depth:02d}.log_drift_exponent",
               block["log_drift_exponent"], _fit_exponent(curve), 1e-5)
        quadrature = sum(
            item["final_hidden_drift"] ** 2 for item in singles if item["layer_index"] < depth
        ) ** 0.5
        _check(checks, f"amplification.prefix.{depth:02d}.independent_quadrature",
               block["independent_quadrature"], quadrature, 1e-6)
        factor = block.get("interaction_factor")
        if quadrature > 0:
            _check(checks, f"amplification.prefix.{depth:02d}.interaction_factor",
                   factor, curve[-1] / quadrature, 1e-4)
        elif factor is not None:
            raise ReproduceError(
                f"amplification: {depth} layers report an interaction factor with no quadrature to divide by")

    crossing = [block["layers_quantised"] for block in prefixes if block["final_hidden_drift"] >= 1.0]
    expected = min(crossing) if crossing else None
    if payload["first_drift_crossing"] != expected:
        raise ReproduceError(
            f"amplification: first_drift_crossing is {payload['first_drift_crossing']!r}, "
            f"the curves say {expected!r}")


def _record(checks: list[dict], check_id: str, stored, recomputed,
            tolerance: float = 0.0) -> None:
    """Append one check in the shape the CLI prints and the report counts."""
    checks.append({
        "id": check_id,
        "stored": stored,
        "recomputed": recomputed,
        "delta": (abs(stored - recomputed)
                  if isinstance(stored, (int, float)) and isinstance(recomputed, (int, float))
                  else 0.0),
        "tolerance": tolerance,
        "ok": stored == recomputed if tolerance == 0.0
              else abs(stored - recomputed) <= tolerance,
    })


def _check_fair(payload: dict, checks: list[dict]) -> None:
    """The fair re-measurement: baseline vs the constant answer, and the bias spread.

    Every number here is re-derived from the integers in the bundle. The claims block is
    also recomputed rather than trusted, because those three booleans are what the page
    leans on when it says the baseline shows no measurable ability.
    """
    for field in ("model", "protocol", "baseline", "constant_answer", "conditions", "claims"):
        if field not in payload:
            raise ReproduceError(f"fair: missing {field!r}")

    baseline = payload["baseline"]
    constant = payload["constant_answer"]
    conditions = payload["conditions"]
    total = payload["protocol"]["questions"]

    _record(checks, "fair:baseline_accuracy",
            baseline["accuracy"], round(baseline["correct"] / total, 6), 1e-9)

    low, high = baseline["wilson95"]
    _record(checks, "fair:baseline_interval_contains_own_accuracy",
            True, low <= baseline["accuracy"] <= high)

    contains = low <= constant["share"] <= high
    _record(checks, "fair:baseline_contains_constant",
            payload["claims"]["baseline_contains_constant"], contains)

    all_below = all(block["wilson95"][1] < constant["share"] for block in conditions)
    _record(checks, "fair:all_quantised_below_constant",
            payload["claims"]["all_quantised_below_constant"], all_below)

    spreads = {block["position_bias_spread"] for block in conditions}
    spreads.add(baseline["position_bias_spread"])
    _record(checks, "fair:bias_identical_across_conditions",
            payload["claims"]["position_bias_identical_across_conditions"], len(spreads) == 1)

    for block in conditions:
        tag = f"fair:g{block['group_size']}"
        _record(checks, f"{tag}:accuracy",
                block["accuracy"], round(block["correct"] / block["count"], 6), 1e-9)
        lo, hi = block["wilson95"]
        _record(checks, f"{tag}:interval_contains_accuracy",
                True, lo <= block["accuracy"] <= hi)
        _record(checks, f"{tag}:histogram_sums_to_count",
                block["count"], sum(block["histogram"].values()))
        _record(checks, f"{tag}:most_common_matches_histogram",
                block["most_common"],
                max(block["histogram"], key=block["histogram"].get))

    _record(checks, "fair:no_prompt_truncated", 0, baseline["truncated"])


def _check_granularity(payload: dict, checks: list[dict]) -> None:
    """One depth, three group sizes: does the collapse survive a change of quantiser?

    A group size that does not collapse is not a failed experiment, so the checker never
    requires a collapse at every row. It requires the two stored summaries -- which classes
    break the collapse, and which group sizes collapsed at all -- to follow from the shares.
    """
    for field in ("model", "protocol", "rows", "collapsed_group_sizes", "limits"):
        if field not in payload:
            raise ReproduceError(f"granularity: missing {field!r}")
    protocol = payload["protocol"]
    collapse = protocol.get("collapse_share")
    if not isinstance(collapse, (int, float)) or not 0 < collapse <= 1:
        raise ReproduceError(f"granularity: collapse_share must be in (0, 1], got {collapse!r}")
    declared = protocol.get("group_sizes")
    if not isinstance(declared, list) or len(declared) < 2:
        raise ReproduceError(
            f"granularity: at least two group sizes are needed to compare, got {declared!r}")
    questions = protocol.get("questions")
    depth = protocol.get("quantised_layers")
    for name, value in (("questions", questions), ("quantised_layers", depth)):
        if not isinstance(value, int) or value <= 0:
            raise ReproduceError(f"granularity: {name} must be a positive integer, got {value!r}")

    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(declared):
        raise ReproduceError(
            f"granularity: rows must hold one entry per group size "
            f"({len(declared)}), got {len(rows) if isinstance(rows, list) else rows!r}")
    seen_sizes = []
    collapsed_sizes = []
    for row in rows:
        size = row.get("group_size")
        if size not in declared:
            raise ReproduceError(f"granularity: group size {size!r} is not one of {declared}")
        if size in seen_sizes:
            raise ReproduceError(f"granularity: group size {size!r} appears twice")
        seen_sizes.append(size)
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ReproduceError(f"granularity: group size {size}: conditions must be a non-empty list")
        shares = {}
        for item in conditions:
            name = item.get("condition")
            if name in shares:
                raise ReproduceError(f"granularity: group size {size}: {name!r} appears twice")
            count = item.get("count")
            histogram = item.get("histogram")
            if count != questions:
                raise ReproduceError(
                    f"granularity: group size {size}: {name!r} counts {count!r} questions, "
                    f"the protocol says {questions}")
            if not isinstance(histogram, dict) or not histogram or sum(histogram.values()) != count:
                raise ReproduceError(
                    f"granularity: group size {size}: {name!r} histogram does not sum to {count}")
            most = max(histogram, key=lambda letter: histogram[letter])
            if item.get("most_common") != most:
                raise ReproduceError(
                    f"granularity: group size {size}: {name!r} most_common is "
                    f"{item.get('most_common')!r}, the histogram says {most!r}")
            if not isinstance(item.get("correct"), int) or item["correct"] > count:
                raise ReproduceError(
                    f"granularity: group size {size}: {name!r} reports {item.get('correct')!r} correct")
            _check(checks, f"granularity.g{size}.{name}.most_common_share",
                   item["most_common_share"], histogram[most] / count, 1e-5)
            shares[name] = histogram[most] / count
        if "all_quantised" not in shares:
            raise ReproduceError(f"granularity: group size {size}: no all_quantised condition")

    # Second pass: the summaries, now that every share is known.
    by_size = {}
    for row in rows:
        size = row["group_size"]
        shares = {item["condition"]: item["histogram"][max(item["histogram"],
                  key=lambda letter: item["histogram"][letter])] / item["count"]
                  for item in row["conditions"]}
        by_size[size] = shares
        collapsed_here = shares["all_quantised"] >= collapse
        if bool(row.get("collapsed")) != collapsed_here:
            raise ReproduceError(
                f"granularity: group size {size}: stored collapsed {row.get('collapsed')!r}, "
                f"the share {shares['all_quantised']:.4f} says {collapsed_here}")
        if collapsed_here:
            collapsed_sizes.append(size)
        survivors = sorted(name for name, share in shares.items() if share < collapse)
        stored = sorted(row.get("survivors") or [])
        if stored != survivors:
            raise ReproduceError(
                f"granularity: group size {size}: survivors {stored} do not match the shares {survivors}")
        keeps = sorted(name for name in shares
                       if name != "all_quantised" and shares[name] >= collapse)
        stated = sorted(row.get("collapsed_keeps") or [])
        if stated != keeps:
            raise ReproduceError(
                f"granularity: group size {size}: collapsed_keeps {stated} do not match {keeps}")

    stored_collapsed = sorted(payload["collapsed_group_sizes"])
    if stored_collapsed != sorted(collapsed_sizes):
        raise ReproduceError(
            f"granularity: collapsed_group_sizes {stored_collapsed} do not match the shares "
            f"{sorted(collapsed_sizes)}")


def _bundle_paths(evidence_path=None) -> list[Path]:
    """Which bundles to check: the one asked for, or the whole evidence directory."""
    if evidence_path:
        return [Path(evidence_path)]
    paths = [DEFAULT_EVIDENCE]
    paths += sorted(EVIDENCE_DIR.glob("real_model_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("bench_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("packed_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("answer_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("packing_equivalence*.json"))
    paths += sorted(EVIDENCE_DIR.glob("prefix_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("nf4_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("layer_cliff*.json"))
    paths += sorted(EVIDENCE_DIR.glob("scaling_*.json"))
    paths += sorted(EVIDENCE_DIR.glob("amplification*.json"))
    paths += sorted(EVIDENCE_DIR.glob("granularity*.json"))
    paths += sorted(EVIDENCE_DIR.glob("fair_remeasure*.json"))
    return paths


def _check_bundle(payload: dict, checks: list[dict], name: str) -> None:
    schema = payload.get("schema")
    if schema == FF_SCHEMA:
        _check_ff(payload, checks)
    elif schema == REAL_MODEL_SCHEMA:
        _check_real_model(payload, checks)
    elif schema == BENCH_SCHEMA:
        _check_bench(payload, checks)
    elif schema == PACKED_SCHEMA:
        _check_packed(payload, checks)
    elif schema == FOOTPRINT_SCHEMA:
        _check_footprint(payload, checks)
    elif schema == EQUIVALENCE_SCHEMA:
        _check_equivalence(payload, checks)
    elif schema == ANSWER_SCHEMA:
        _check_answer_distribution(payload, checks)
    elif schema == PREFIX_SCHEMA:
        _check_prefix_scan(payload, checks)
    elif schema == NF4_SCHEMA:
        _check_nf4(payload, checks)
    elif schema == LAYER_CLIFF_SCHEMA:
        _check_layer_cliff(payload, checks)
    elif schema == SCALING_SCHEMA:
        _check_scaling(payload, checks)
    elif schema == AMP_SCHEMA:
        _check_amplification(payload, checks)
    elif schema == FAIR_SCHEMA:
        _check_fair(payload, checks)
    elif schema == GRANULARITY_SCHEMA:
        _check_granularity(payload, checks)
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

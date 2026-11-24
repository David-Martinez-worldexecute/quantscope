# -*- coding: utf-8 -*-
"""The layer-cliff checker: eighteen layers collapsed, then one class put back to float32.

The bundle is synthetic -- fourteen matrices, ten questions -- so that a wrong mean or a
mis-stated survivor list fails here instead of on the real 126-matrix bundle, where the same
mistake would be hidden by numbers that look plausible.
"""
from __future__ import annotations

import json
import shutil
import unittest

import support
from quantscope import reproduce as reproduce_mod

COUNT = 10
DEPTH = 2
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ALL_A = {"A": 10, "B": 0, "C": 0, "D": 0}
SPREAD = {"A": 4, "B": 2, "C": 2, "D": 2}
ALMOST_ALL_A = {"A": 9, "B": 1, "C": 0, "D": 0}


def condition(name: str, quantised: int, histogram: dict, correct: int) -> dict:
    most = max(histogram, key=lambda letter: histogram[letter])
    return {
        "condition": name,
        "quantised_matrices": quantised,
        "float32_matrices": DEPTH * 7 - quantised,
        "correct": correct,
        "count": COUNT,
        "histogram": dict(histogram),
        "most_common": most,
        "most_common_share": round(histogram[most] / COUNT, 6),
        "mean_gap": 1.0,
    }


def by_kind(copies: int = DEPTH) -> dict:
    return {
        kind: {
            "matrices": copies,
            "weights": 1000 * copies,
            "mean_zero_fraction": 0.31,
            "zero_fraction_sum": 0.31 * copies,
            "mean_relative_error": 0.51,
            "relative_error_sum": 0.51 * copies,
        }
        for kind in KINDS
    }


def cliff_bundle() -> dict:
    conditions = [
        condition("all_quantised", 14, ALL_A, 2),
        condition("keep_k_proj", 12, SPREAD, 4),
        condition("keep_q_proj", 12, ALMOST_ALL_A, 3),
    ]
    return {
        "schema": "quantscope-layer-cliff-v1",
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": {"name": "synthetic"},
        "protocol": {
            "questions": COUNT,
            "quantised_layers": DEPTH,
            "group_size": 64,
            "max_tokens": 96,
            "scoring": "last-position single-token A/B/C/D logits",
            "quantisation": "ternary per-group absmean (same formula as the lab quantiser)",
            "collapse_share": 0.9,
            "device": "cuda",
            "torch": "2.6.0+cu124",
            "cuda": "12.4",
            "threads": 4,
        },
        "baseline": {
            "correct": 2,
            "count": COUNT,
            "histogram": {"A": 5, "B": 2, "C": 1, "D": 2},
            "most_common": "A",
            "most_common_share": 0.5,
            "mean_gap": 2.0,
        },
        "conditions": conditions,
        "anchor": {
            "layers_quantised": DEPTH,
            "correct": 2,
            "count": COUNT,
            "histogram": dict(ALL_A),
            "most_common": "A",
            "most_common_share": 1.0,
        },
        "matrix_stats": {
            "matrices": 14,
            "by_kind": by_kind(),
            "worst_relative_error": [],
        },
        "survivors": ["keep_k_proj"],
        "collapsed_keeps": ["keep_q_proj"],
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
        "seconds": 1.0,
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class LayerCliffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("cliff-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict) -> "object":
        path = self.directory / "layer_cliff.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_shares_and_means(self) -> None:
        report = reproduce_mod.reproduce(self.write(cliff_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("cliff.all_quantised.most_common_share", ids)
        self.assertIn("cliff.keep_k_proj.most_common_share", ids)
        self.assertIn("cliff.by_kind.q_proj.mean_zero_fraction", ids)
        self.assertIn("cliff.by_kind.down_proj.mean_relative_error", ids)

    def test_share_that_disagrees_with_the_histogram_fails(self) -> None:
        bundle = cliff_bundle()
        bundle["conditions"][1]["most_common_share"] = 0.5
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["cliff.keep_k_proj.most_common_share"])

    def test_mean_that_disagrees_with_its_sum_fails(self) -> None:
        bundle = cliff_bundle()
        bundle["matrix_stats"]["by_kind"]["q_proj"]["mean_zero_fraction"] = 0.5
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["cliff.by_kind.q_proj.mean_zero_fraction"])

    def test_survivor_list_that_contradicts_the_shares_is_rejected(self) -> None:
        bundle = cliff_bundle()
        bundle["survivors"] = ["keep_k_proj", "keep_q_proj"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_collapsed_keeps_that_contradict_the_shares_are_rejected(self) -> None:
        bundle = cliff_bundle()
        bundle["collapsed_keeps"] = []
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_matrix_counts_that_forget_a_layer_are_rejected(self) -> None:
        bundle = cliff_bundle()
        bundle["conditions"][1]["quantised_matrices"] = 13
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_stats_that_count_a_different_set_of_matrices_are_rejected(self) -> None:
        bundle = cliff_bundle()
        bundle["matrix_stats"]["matrices"] = 12
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_kind_missing_from_the_stats_is_rejected(self) -> None:
        bundle = cliff_bundle()
        del bundle["matrix_stats"]["by_kind"]["up_proj"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_an_uncollapsed_all_quantised_row_is_rejected(self) -> None:
        bundle = cliff_bundle()
        row = bundle["conditions"][0]
        row["histogram"] = dict(SPREAD)
        row["most_common"] = "A"
        row["most_common_share"] = 0.4
        bundle["anchor"]["histogram"] = dict(SPREAD)
        bundle["anchor"]["most_common_share"] = 0.4
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_histogram_that_does_not_sum_to_the_count_is_rejected(self) -> None:
        bundle = cliff_bundle()
        bundle["conditions"][1]["histogram"]["A"] = 5
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
